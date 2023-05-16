#! constant
try:
    from IPython.display import clear_output
except:
    pass
from gen_data_func2 import *
from decoder import GreedyDecoder, BeamCTCDecoder
import wandb
from utils import fix_seeds, remove_from_dict, prepare_bpe
from easydict import EasyDict as edict
import yaml
from model.quartznet import QuartzNet
from model import configs as quartznet_configs
from functools import partial
from audiomentations import TimeStretch, PitchShift, AddGaussianNoise
import torchaudio
import youtokentome as yttm
from data.transforms import (
    Compose,
    AddLengths,
    AudioSqueeze,
    TextPreprocess,
    MaskSpectrogram,
    ToNumpy,
    BPEtexts,
    MelSpectrogram,
    ToGpu,
    Pad,
    NormalizedMelSpectrogram,
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

# BLANK_VAL = 257
# dna_vocab = [f"_{x}_" for x in range(256)]  # .append(['spacer','blank'])
# space_symbol = "spacer"
# dna_vocab.append(space_symbol)
# dna_vocab.append("blank")
# spacer_idx = 256

BLANK_VAL = 0
dna_vocab = [f"_{x+1}_" for x in range(256)]  # .append(['spacer','blank'])
space_symbol = "spacer"
dna_vocab.append(space_symbol)
dna_vocab.insert(BLANK_VAL, "blank")
spacer_idx = 257


def evaluate(config, args):
    fix_seeds(seed=config.train.get("seed", 42))
    dataset_module = importlib.import_module(f".{config.dataset.name}", data.__name__)
    bpe = prepare_bpe(config)

    transforms_val = Compose(
        [TextPreprocess(), ToNumpy(), BPEtexts(bpe=bpe), AudioSqueeze()]
    )

    batch_transforms_val = Compose(
        [
            ToGpu("cuda" if torch.cuda.is_available() else "cpu"),
            NormalizedMelSpectrogram(
                sample_rate=config.dataset.get("sample_rate", 16000),  # for LJspeech
                n_mels=config.model.feat_in,
                normalize=config.dataset.get("normalize", None),
            ).to("cuda" if torch.cuda.is_available() else "cpu"),
            AddLengths(),
            Pad(),
        ]
    )
    #! ############
    # ! TIN Dataset
    if args.mode == "word":
        val_dataset = Dataset_ctc(signals_v, ctc_labels_v)
        print("using word mode")
    elif args.mode == "sample":
        val_dataset = Dataset_ctc(signals_v, spacer_labels2_v)
        print("using sample mode")

    elif args.mode == "compress":
        [signal_qs_v, label_qs_v, all_states_v] = compress_sig(
            spacer_labels2_v, signals_v
        )
        val_dataset = Dataset_ctc(signal_qs_v, ctc_labels_v)
        print("using compress mode")

        idx = 0
        states = all_states[idx]
        signal = signals[idx]
        plt.figure()
        plt.plot(signal, "-.")
        for state in states:
            plt.plot(state, np.mean(signal[state]) * np.ones(state.shape), "r")
        plt.savefig("fig/compress_states")
        # raise Exception("STOP TESTING")

        plt.figure()
        plt.plot(signal_qs[idx])
        plt.savefig("fig/compress_signal")

    elif args.mode == "compress_sample":
        [signal_qs_v, label_qs_v, all_states_v] = compress_sig(
            spacer_labels2_v, signals_v
        )
        val_dataset = Dataset_ctc(signal_qs_v, label_qs_v)
        print("using compress sample mode")

        idx = 0
        states = all_states[idx]
        signal = signals[idx]
        plt.plot(signal, "-.")
        for state in states:
            plt.plot(state, np.mean(signal[state]) * np.ones(state.shape), "r")
        plt.savefig("fig/compress_states")

        plt.figure()
        plt.plot(signal_qs[idx])
        plt.savefig("fig/compress_signal")
    else:
        raise Exception("NON EXISTING MODE INPUT")

    #! ############
    val_dataset = dataset_module.get_dataset(
        config, transforms=transforms_val, part="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=config.train.get("num_workers", 4),
        batch_size=1,
        collate_fn=no_pad_collate,
    )

    model = QuartzNet(
        model_config=getattr(
            quartznet_configs, config.model.name, "_quartznet5x5_config"
        ),
        **remove_from_dict(config.model, ["name"]),
    )
    print(model)

    if config.train.get("from_checkpoint", None) is not None:
        model.load_weights(config.train.from_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    decoder = BeamCTCDecoder(bpe=bpe)

    model.eval()
    val_stats = defaultdict(list)
    for batch_idx, batch in enumerate(val_dataloader):
        batch = batch_transforms_val(batch)
        with torch.no_grad():
            logits = model(batch["audio"].float())
            output_length = torch.ceil(
                batch["input_lengths"].float() / model.stride
            ).int()
            loss = criterion(
                logits.permute(2, 0, 1).log_softmax(dim=2),
                batch["text"],
                output_length,
                batch["target_lengths"],
            )

        target_strings = decoder.convert_to_strings(batch["text"].int())
        decoded_output = decoder.decode(logits.permute(0, 2, 1).softmax(dim=2))
        wer = np.mean(
            [
                decoder.wer(true, pred)
                for true, pred in zip(target_strings, decoded_output)
            ]
        )
        cer = np.mean(
            [
                decoder.cer(true, pred)
                for true, pred in zip(target_strings, decoded_output)
            ]
        )
        val_stats["val_loss"].append(loss.item())
        val_stats["wer"].append(wer)
        val_stats["cer"].append(cer)
    for k, v in val_stats.items():
        val_stats[k] = np.mean(v)
    val_stats["val_samples"] = wandb.Table(
        columns=["gt_text", "pred_text"], data=list(zip(target_strings, decoded_output))
    )
    print(val_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation model.")
    parser.add_argument(
        "--config", default="configs/train_LJSpeech.yml", help="path to config file"
    )
    parser.add_argument("--mode", default="word", help="model name")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f))
    evaluate(config)
