# %% [markdown]
# # letters: 0-255, spacer:256, blank(e):257

# %%
#! constant
from os import walk
from IPython.display import clear_output
from gen_data_func import *
from decoder import GreedyDecoder, BeamCTCDecoder
import wandb
from utils import fix_seeds, remove_from_dict, prepare_bpe
from easydict import EasyDict as edict
import yaml
from model.quartznet import QuartzNet
from model import configs as quartznet_configs
from functools import partial
from audiomentations import (
    TimeStretch, PitchShift, AddGaussianNoise
)
import torchaudio
import youtokentome as yttm
from data.transforms import (
    Compose, AddLengths, AudioSqueeze, TextPreprocess,
    MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
    ToGpu, Pad, NormalizedMelSpectrogram
)
from data.collate import collate_fn, gpu_collate, no_pad_collate
import data
import pytorch_warmup as warmup
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch import nn
import torch
import importlib
from collections import defaultdict
from tqdm import tqdm
import argparse
import json
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math
import ScrappySequence as ss
import randomizeSequence as rSeq
BLANK_VAL = 257
dna_vocab = [f"_{x}_" for x in range(256)]  # .append(['spacer','blank'])
dna_vocab.append('spacer')
dna_vocab.append('blank')
dna_vocab.append('_')

#! Tin

#!/usr/bin/env python

# torchim:
# from tensorboardX import SummaryWriter

# data:


# model:

# utils:


# %%

class DNA_vocab:
    # character and index is the same for this task
    def __init__(self, dna_vocab):
        self.dna_vocab = dna_vocab

    def vocab(self):
        return dna_vocab

    def id_to_subword(self, id):
        return dna_vocab[id]

# %%


class AudioUnsqueeze:
    def __call__(self, data):
        data['audio'] = data['audio'].unsqueeze(1)
        return data


# %%
# choosing file
print(os.getcwd())
data_file_name_train = "out_csv/full_spacer_detection_10k.csv"
data_file_name_val = "out_csv/full_spacer_detection.csv"
samples_per_sequence_train = 2
samples_per_sequence_val = 1


# %%
start = time.monotonic()
signals, index, spacer_labels, letter_labels, barcode_labels, ctc_labels = prepare_train2(
    data_file_name_train, samples_per_sequence_train)
end = time.monotonic()
print(
    f"Generated {len(signals)} signals and {len(spacer_labels)} labels", f"in {end-start}")
# Sequential: 26.099690708000026b

# %%
start = time.monotonic()
signals_v, index_v, spacer_labels_v, letter_labels_v, barcode_labels_v, ctc_labels_v = prepare_train2(
    data_file_name_val, samples_per_sequence_val)
end = time.monotonic()
print(
    f"Generated {len(signals)} signals and {len(spacer_labels)} labels", f"in {end-start}")
#Sequential: 26.099690708000026

# %%
plt.figure(figsize=(30, 10))
i = 0
plt.plot(signals[i])
plt.plot(spacer_labels[i])
print("Letters:", letter_labels[i], "\nbarcodes:",
      barcode_labels[i], '\nctc_labels:', ctc_labels[i])

# print(len(signals))
# print(len(signals[i]))
print(([len(x) for x in signals]))
print(max([len(x) for x in signals]))

# %%
# Create a "SignalDataset" class by importing the Pytorch Dataset class


class Dataset_ctc(Dataset):
    """ Data noisey sinewave dataset
        num_datapoints - the number of datapoints you want
    """

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    # called by the dataLOADER class whenever it wants a new mini-batch
    # returns corresponding input datapoints AND the corresponding labels
    def __getitem__(self, index):
        return {'audio': np.array(self.x_data[index]), 'text': np.array(self.y_data[index]), 'sample_rate': 22050}

    # length of the dataset
    def __len__(self):
        return len(self.x_data)

# %%

# TODO: wrap to trainer class


parser = argparse.ArgumentParser(description='Training model.')
parser.add_argument('--config', default='configs/train_LJSpeech.yaml',
                    help='path to config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = edict(yaml.safe_load(f))


def write_to_file(str_w, file_name='sth.txt', mode='w'):
    with open(file_name, mode) as f:
        f.write(str(str_w))


# %%
config

# %%
fix_seeds(seed=config.train.get('seed', 42))
dataset_module = importlib.import_module(
    f'.{config.dataset.name}', data.__name__)
bpe = prepare_bpe(config)
bpe = DNA_vocab(dna_vocab)

transforms_train = Compose([
    # TextPreprocess(),# removing punctuation in text - might not needed
    # ToNumpy(), # convert audio to numpy
    # BPEtexts(bpe=bpe, dropout_prob=config.bpe.get('dropout_prob', 0.05)),
    # AudioSqueeze(), # remove 1st dimension if it is 1 [1,...]
    # AddGaussianNoise(
    #     min_amplitude=0.001,
    #     max_amplitude=0.015,
    #     p=0.5
    # ),
    # TimeStretch(
    #     min_rate=0.8,
    #     max_rate=1.25,
    #     p=0.5
    # ),
    # PitchShift(
    #     min_semitones=-4,
    #     max_semitones=4,
    #     p=0.5
    # )
    # AddLengths()
])

batch_transforms_train = Compose([
    ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
    # NormalizedMelSpectrogram(
    #     sample_rate=config.dataset.get('sample_rate', 16000),
    #     n_mels=config.model.feat_in,
    #     normalize=config.dataset.get('normalize', None)
    # ).to('cuda' if torch.cuda.is_available() else 'cpu'),
    # MaskSpectrogram(
    #     probability=0.5,
    #     time_mask_max_percentage=0.05,
    #     frequency_mask_max_percentage=0.15
    # ),
    AddLengths(),
    Pad(),
    AudioUnsqueeze()  # ! Tin add
])

transforms_val = Compose([
    #     TextPreprocess(),
    #     ToNumpy(),
    #     BPEtexts(bpe=bpe),
    #     AudioSqueeze()
])

batch_transforms_val = Compose([
    ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
    # NormalizedMelSpectrogram(
    #     sample_rate=config.dataset.get(
    #         'sample_rate', 16000),  # for LJspeech
    #     n_mels=config.model.feat_in,
    #     normalize=config.dataset.get('normalize', None)
    # ).to('cuda' if torch.cuda.is_available() else 'cpu'),
    AddLengths(),
    Pad(),  # pad both audio and text
    AudioUnsqueeze()  # ! Tin add
])

# load datasets
# train_dataset = dataset_module.get_dataset(
#     config, transforms=transforms_train, part='train')
# val_dataset = dataset_module.get_dataset(
#     config, transforms=transforms_val, part='val')

# ! TIN Dataset
train_dataset = Dataset_ctc(signals, ctc_labels)
val_dataset = Dataset_ctc(signals_v, ctc_labels_v)

# print("!!!", config.train.get('num_workers', 4))
train_dataloader = DataLoader(train_dataset,
                              batch_size=config.train.get('batch_size', 1), collate_fn=no_pad_collate)

val_dataloader = DataLoader(val_dataset,
                            batch_size=1, collate_fn=no_pad_collate)

# print(getattr(quartznet_configs, config.model.name, '_quartznet5x5_config'))

model = QuartzNet(
    model_config=getattr(
        quartznet_configs, config.model.name, '_quartznet5x5_config'),
    **remove_from_dict(config.model, ['name'])
)
# model = QuartzNet(
#     model_config=getattr(
#         quartznet_configs, config.model.name, '_quartznet5x5_config'),**config)


# print(model)
write_to_file(model, 'model_structure.txt')

optimizer = torch.optim.Adam(
    model.parameters(), **config.train.get('optimizer', {}))
num_steps = len(train_dataloader) * config.train.get('epochs', 10)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_steps)
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)


f = []
mypath = 'checkpoints'
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)

best_model_epoch = [int(x.split('_')[1]) for x in f]
# print(best_model_epoch)
best_model_name = 'checkpoints/' + \
    f[best_model_epoch.index(max(best_model_epoch))]

if config.train.get('from_checkpoint', None) is not None:
    model.load_weights(config.train.from_checkpoint)
    print(f'load from checkpoint {best_model_name}')
else:
    model.load_weights(best_model_name)
    print(f'load from checkpoint {best_model_name}')

if torch.cuda.is_available():
    model = model.cuda()


criterion = nn.CTCLoss(blank=BLANK_VAL, reduction='mean', zero_infinity=True)
# criterion = nn.CTCLoss(blank=config.model.vocab_size)
decoder = GreedyDecoder(bpe=bpe, blank_index=BLANK_VAL, space_simbol='_')

prev_wer = 1000
#! TIN CHNAGE
wandb.init(project=config.wandb.project, config=config)
wandb.watch(model, log="all", log_freq=config.wandb.get(
    'log_interval', 5000))

# %%

for epoch_idx in tqdm(range(config.train.get('epochs', 10))):
    # train:
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        batch1 = batch
        # print(len(batch1['audio'][0]))
        batch = batch_transforms_train(batch)
        batch2 = batch
        # print(len(batch2['audio'][0]))

        optimizer.zero_grad()
        logits = model(batch['audio'].float())

        output_length = torch.ceil(
            batch['input_lengths'].float() / model.stride).int()

        # print('batch audio shape:', batch['audio'].shape, batch['audio'].dtype )
        # print('logits shape:', logits.shape)
        # print('output_length shape:',output_length)
        # print('target_lengths:',batch['target_lengths'])

        loss = criterion(logits.permute(2, 0, 1).log_softmax(
            dim=2), batch['text'], output_length, batch['target_lengths'])  # target_length is the length of text of batch1 (before batch aug)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.get('clip_grad_norm', 15))
        optimizer.step()
        lr_scheduler.step()
        # warmup_scheduler.dampen()
        # break
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
            #! TIN change
            wandb.log({
                "train_loss": loss.item(),
                "train_wer": wer,
                "train_cer": cer,
                "train_samples": wandb.Table(
                    columns=['gt_text', 'pred_text'],
                    data=list(zip(target_strings, decoded_output))
                )
            }, step=step)
        # print(decoded_output)
    #!
    # validate:
    model.eval()
    val_stats = defaultdict(list)
    for batch_idx, batch in enumerate(val_dataloader):
        batch = batch_transforms_val(batch)
        with torch.no_grad():
            logits = model(batch['audio'].float())
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
        columns=['gt_text', 'pred_text'], data=list(zip(target_strings, decoded_output)))
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
        try:
            wandb.save(os.path.join(config.train.get(
                'checkpoint_path', 'checkpoints'), f'model_{epoch_idx}_{prev_wer}.pth'))
        except:
            print('could not use wandb save')
    # break
