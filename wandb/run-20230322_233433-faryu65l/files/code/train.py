#!/usr/bin/env python

import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
import math
import argparse
import importlib

# torchim:
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
# from tensorboardX import SummaryWriter
import numpy as np
import pytorch_warmup as warmup

# data:
import data
from data.collate import collate_fn, gpu_collate, no_pad_collate
from data.transforms import (
    Compose, AddLengths, AudioSqueeze, TextPreprocess,
    MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
    ToGpu, Pad, NormalizedMelSpectrogram
)
import youtokentome as yttm

import torchaudio
from audiomentations import (
    TimeStretch, PitchShift, AddGaussianNoise
)
from functools import partial

# model:
from model import configs as quartznet_configs
from model.quartznet import QuartzNet

# utils:
import yaml
from easydict import EasyDict as edict
from utils import fix_seeds, remove_from_dict, prepare_bpe
import wandb
from decoder import GreedyDecoder, BeamCTCDecoder

import youtokentome as yttm


# TODO: wrap to trainer class


def train(config):
    fix_seeds(seed=config.train.get('seed', 42))
    dataset_module = importlib.import_module(
        f'.{config.dataset.name}', data.__name__)
    bpe = prepare_bpe(config)

    transforms_train = Compose([
        TextPreprocess(),
        ToNumpy(),
        BPEtexts(bpe=bpe, dropout_prob=config.bpe.get('dropout_prob', 0.05)),
        AudioSqueeze(),
        AddGaussianNoise(
            min_amplitude=0.001,
            max_amplitude=0.015,
            p=0.5
        ),
        TimeStretch(
            min_rate=0.8,
            max_rate=1.25,
            p=0.5
        ),
        PitchShift(
            min_semitones=-4,
            max_semitones=4,
            p=0.5
        )
        # AddLengths()
    ])

    batch_transforms_train = Compose([
        ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
        NormalizedMelSpectrogram(
            sample_rate=config.dataset.get('sample_rate', 16000),
            n_mels=config.model.feat_in,
            normalize=config.dataset.get('normalize', None)
        ).to('cuda' if torch.cuda.is_available() else 'cpu'),
        MaskSpectrogram(
            probability=0.5,
            time_mask_max_percentage=0.05,
            frequency_mask_max_percentage=0.15
        ),
        AddLengths(),
        Pad()
    ])

    transforms_val = Compose([
        TextPreprocess(),
        ToNumpy(),
        BPEtexts(bpe=bpe),
        AudioSqueeze()
    ])

    batch_transforms_val = Compose([
        ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
        NormalizedMelSpectrogram(
            sample_rate=config.dataset.get(
                'sample_rate', 16000),  # for LJspeech
            n_mels=config.model.feat_in,
            normalize=config.dataset.get('normalize', None)
        ).to('cuda' if torch.cuda.is_available() else 'cpu'),
        AddLengths(),
        Pad()
    ])

    # load datasets
    train_dataset = dataset_module.get_dataset(
        config, transforms=transforms_train, part='train')
    val_dataset = dataset_module.get_dataset(
        config, transforms=transforms_val, part='val')
    print("!!!", config.train.get('num_workers', 4))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.train.get('batch_size', 1), collate_fn=no_pad_collate)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1, collate_fn=no_pad_collate)

    model = QuartzNet(
        model_config=getattr(
            quartznet_configs, config.model.name, '_quartznet5x5_config'),
        **remove_from_dict(config.model, ['name'])
    )

    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), **config.train.get('optimizer', {}))
    num_steps = len(train_dataloader) * config.train.get('epochs', 10)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    if config.train.get('from_checkpoint', None) is not None:
        model.load_weights(config.train.from_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    # criterion = nn.CTCLoss(blank=config.model.vocab_size)
    decoder = GreedyDecoder(bpe=bpe)

    prev_wer = 1000
    wandb.init(project=config.wandb.project, config=config)
    wandb.watch(model, log="all", log_freq=config.wandb.get(
        'log_interval', 5000))
    for epoch_idx in tqdm(range(config.train.get('epochs', 10))):
        # train:
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # print(batch)
            batch = batch_transforms_train(batch)
            # print(batch)

            optimizer.zero_grad()
            logits = model(batch['audio'])
            output_length = torch.ceil(
                batch['input_lengths'].float() / model.stride).int()
            loss = criterion(logits.permute(2, 0, 1).log_softmax(
                dim=2), batch['text'], output_length, batch['target_lengths'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.get('clip_grad_norm', 15))
            optimizer.step()
            lr_scheduler.step()
            # warmup_scheduler.dampen()

            if batch_idx % config.wandb.get('log_interval', 5000) == 0:
                target_strings = decoder.convert_to_strings(batch['text'])
                decoded_output = decoder.decode(
                    logits.permute(0, 2, 1).softmax(dim=2))
                wer = np.mean([decoder.wer(true, pred)
                              for true, pred in zip(target_strings, decoded_output)])
                cer = np.mean([decoder.cer(true, pred)
                              for true, pred in zip(target_strings, decoded_output)])
                step = epoch_idx * \
                    len(train_dataloader) * train_dataloader.batch_size + \
                    batch_idx * train_dataloader.batch_size
                wandb.log({
                    "train_loss": loss.item(),
                    "train_wer": wer,
                    "train_cer": cer,
                    "train_samples": wandb.Table(
                        columns=['gt_text', 'pred_text'],
                        data=zip(target_strings, decoded_output)
                    )
                }, step=step)

        # validate:
        model.eval()
        val_stats = defaultdict(list)
        for batch_idx, batch in enumerate(val_dataloader):
            batch = batch_transforms_val(batch)
            with torch.no_grad():
                logits = model(batch['audio'])
                output_length = torch.ceil(
                    batch['input_lengths'].float() / model.stride).int()
                loss = criterion(logits.permute(2, 0, 1).log_softmax(
                    dim=2), batch['text'], output_length, batch['target_lengths'])

            target_strings = decoder.convert_to_strings(batch['text'])
            decoded_output = decoder.decode(
                logits.permute(0, 2, 1).softmax(dim=2))
            wer = np.mean([decoder.wer(true, pred)
                          for true, pred in zip(target_strings, decoded_output)])
            cer = np.mean([decoder.cer(true, pred)
                          for true, pred in zip(target_strings, decoded_output)])
            val_stats['val_loss'].append(loss.item())
            val_stats['wer'].append(wer)
            val_stats['cer'].append(cer)
        for k, v in val_stats.items():
            val_stats[k] = np.mean(v)
        val_stats['val_samples'] = wandb.Table(
            columns=['gt_text', 'pred_text'], data=zip(target_strings, decoded_output))
        wandb.log(val_stats, step=step)

        # save model, TODO: save optimizer:
        if val_stats['wer'] < prev_wer:
            os.makedirs(config.train.get(
                'checkpoint_path', 'checkpoints'), exist_ok=True)
            prev_wer = val_stats['wer']
            torch.save(
                model.state_dict(),
                os.path.join(config.train.get(
                    'checkpoint_path', 'checkpoints'), f'model_{epoch_idx}_{prev_wer}.pth')
            )
            wandb.save(os.path.join(config.train.get(
                'checkpoint_path', 'checkpoints'), f'model_{epoch_idx}_{prev_wer}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='configs/train_LJSpeech.yaml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(f))
    train(config)
