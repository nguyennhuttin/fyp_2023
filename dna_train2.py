# %% [markdown]
# # letters: 0-255, spacer:256, blank(e):257

# %%
#! constant
try:
    from IPython.display import clear_output
except:
    pass
from gen_data_func2_new import *
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

parser = argparse.ArgumentParser(description="Training model.")
parser.add_argument(
    "--config", default="configs/train_LJSpeech.yaml", help="path to config file"
)
parser.add_argument(
    "--checkpoint_path", default="checkpoints_free", help="path to checkpoint file"
)

# full=True, using_barcode=True, num_letter=7
parser.add_argument("--seg", action="store_false", help="using only 1 seq s-l-s")
parser.add_argument("--no_bc", action="store_false", help="specify data has no barcode")
parser.add_argument("--numletter", default="7", help="specify number of letter")

# data_file_name_train = "out_csv/full_spacer_detection_10k.csv"
# data_file_name_val = "out_csv/full_spacer_detection.csv"
parser.add_argument(
    "--train_data",
    default="out_csv/full_spacer_detection_10k.csv",
    help="path to train data",
)
parser.add_argument(
    "--valid_data",
    default="out_csv/full_spacer_detection.csv",
    help="path to train data",
)
parser.add_argument("--new", action="store_true", help="start training from scratch")
parser.add_argument("--model_name", default="model", help="model name")
parser.add_argument("--mode", default="word", help="model name")
parser.add_argument(
    "--eval", action="store_true", help="skip training and do evaluation"
)
parser.add_argument("--custom", action="store_true", help="use custom testing case")
parser.add_argument("--customfile", default="temp_data/segs_val_dataset.pt")
parser.add_argument("--customfiletrain", default="temp_data/segs_train_dataset.pt")
args = parser.parse_args()
# print(args.seg)
print("\n!!!!args!!!")
print(args)
#! Tin

from remove_amp_noise_6 import remove_amp_noise_6

# time-compression parameters
state_len = 1
amp_diff_st = 0.8
min_samples_st = 2

#!/usr/bin/env python

# torchim:
# from tensorboardX import SummaryWriter

# data:


# model:


# utils:
def remove_bc(signal, spacer_labels2):
    val = spacer_labels2[0]
    for i, value in enumerate(spacer_labels2):
        if value != val:
            start_index = i
            break

    val = spacer_labels2[-1]
    for i, value in enumerate(spacer_labels2[::-1]):
        if value != val:
            end_index = i
            break
    end_index = len(signal) - end_index
    # print(start_index,end_index)
    # print(start_index,end_index)
    return signal[start_index:end_index], spacer_labels2[start_index:end_index]


def split_signal(signal, spacer_label2, segment_size=200, overlap=25):
    segments = []
    i = 0
    # print(len(signal))
    while i < len(signal):
        segment = signal[i : i + segment_size]
        if len(segment) == segment_size:
            segments.append(segment)
        i += segment_size - overlap
        # print(i)

    ctc_segments = []
    spacer_label2_seg = []
    i = 0
    while i < len(spacer_label2):
        segment = spacer_label2[i : i + segment_size]
        if len(segment) == segment_size:
            ctc_segments.append(list(set(segment)))
            spacer_label2_seg.append(segment)
        i += segment_size - overlap
    return segments, ctc_segments, spacer_label2_seg


def compress_sig(spacer_labels2, signals):
    result = []
    label_qs = []
    signal_qs = []
    all_states = []
    for i in range(len(spacer_labels2)):
        signal = signals[i]
        lst = spacer_labels2[i]
        [signal_q, states, _] = remove_amp_noise_6(
            signal, amp_diff_st, state_len, min_samples_st
        )
        start = 0
        for i in range(1, len(lst)):
            if lst[i] != lst[i - 1]:
                result.append((lst[i - 1], start, i - 1))
                start = i
        result.append((lst[-1], start, len(lst) - 1))

        letter_reg = result[1]
        label_q = []

        l = result[0][0]
        for state in states:
            if state[0] <= letter_reg[1] and state[1] >= letter_reg[1]:
                l = result[1][0]
            label_q.append(l)
            if state[1] >= letter_reg[2]:
                l = result[2][0]
        label_qs.append(label_q)
        signal_qs.append(signal_q)
        all_states.append(states)
    return [signal_qs, label_qs, all_states]


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
        data["audio"] = data["audio"].unsqueeze(1)
        return data


# %%
# choosing file
print("current directory:", os.getcwd())
# data_file_name_train = "out_csv/full_spacer_detection_10k.csv"
# data_file_name_val = "out_csv/full_spacer_detection.csv"
data_file_name_train = args.train_data
data_file_name_val = args.valid_data


samples_per_sequence_train = 3
samples_per_sequence_val = 1

# ! load train
print(f"loading train data from {data_file_name_train}")
# %%
start = time.monotonic()
(
    signals,
    index,
    spacer_labels,
    letter_labels,
    barcode_labels,
    ctc_labels,
    spacer_labels2,
) = prepare_train2(
    data_file_name_train,
    samples_per_sequence_train,
    args.seg,
    args.no_bc,
    int(args.numletter),
    spacer_idx=spacer_idx,
)
end = time.monotonic()
print(
    f"Generated {len(signals)} signals and {len(spacer_labels)} labels",
    f"in {end-start}",
)
# Sequential: 26.099690708000026b

# ! load valid
print(f"loading val data from {data_file_name_val}")
# %%

start = time.monotonic()
(
    signals_v,
    index_v,
    spacer_labels_v,
    letter_labels_v,
    barcode_labels_v,
    ctc_labels_v,
    spacer_labels2_v,
) = prepare_train2(
    data_file_name_val,
    samples_per_sequence_val,
    args.seg,
    args.no_bc,
    int(args.numletter),
    spacer_idx=spacer_idx,
)
end = time.monotonic()
print(
    f"Generated {len(signals)} signals and {len(spacer_labels)} labels",
    f"in {end-start}",
)
# Sequential: 26.099690708000026

# %%
plt.figure()
i = 0
plt.plot(signals[i])
plt.plot(spacer_labels[i])
plt.title(
    f"letters:{letter_labels[i]}, barcodes:{barcode_labels[i]}, ctc_labels:{ctc_labels[i]}"
)
print(
    "Letters:",
    letter_labels[i],
    "\nbarcodes:",
    barcode_labels[i],
    "\nctc_labels:",
    ctc_labels[i],
)
plt.savefig("fig/signal_vis")

plt.figure()
plt.plot(spacer_labels2[i])
plt.savefig("fig/spacer_labels2")
# # print(len(signals))
# # print(len(signals[i]))
# print(([len(x) for x in signals]))
# print(max([len(x) for x in signals]))

# %%
# Create a "SignalDataset" class by importing the Pytorch Dataset class


class Dataset_ctc(Dataset):
    """Data noisey sinewave dataset
    num_datapoints - the number of datapoints you want
    """

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    # called by the dataLOADER class whenever it wants a new mini-batch
    # returns corresponding input datapoints AND the corresponding labels
    def __getitem__(self, index):
        return {
            "audio": np.array(self.x_data[index]),
            "text": np.array(self.y_data[index]),
            "sample_rate": 22050,
        }

    # length of the dataset
    def __len__(self):
        return len(self.x_data)


# %%

# TODO: wrap to trainer class


with open(args.config, "r") as f:
    config = edict(yaml.safe_load(f))


def write_to_file(str_w, file_name="sth.txt", mode="w"):
    with open(file_name, mode) as f:
        f.write(str(str_w))


# %%
config

# %%
fix_seeds(seed=config.train.get("seed", 42))
dataset_module = importlib.import_module(f".{config.dataset.name}", data.__name__)
# bpe = prepare_bpe(config)
bpe = DNA_vocab(dna_vocab)
print("DNA vocab:", dna_vocab)

transforms_train = Compose(
    [
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
    ]
)

batch_transforms_train = Compose(
    [
        ToGpu("cuda" if torch.cuda.is_available() else "cpu"),
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
        AudioUnsqueeze(),  # ! Tin add
    ]
)

transforms_val = Compose(
    [
        #     TextPreprocess(),
        #     ToNumpy(),
        #     BPEtexts(bpe=bpe),
        #     AudioSqueeze()
    ]
)

batch_transforms_val = Compose(
    [
        ToGpu("cuda" if torch.cuda.is_available() else "cpu"),
        # NormalizedMelSpectrogram(
        #     sample_rate=config.dataset.get(
        #         'sample_rate', 16000),  # for LJspeech
        #     n_mels=config.model.feat_in,
        #     normalize=config.dataset.get('normalize', None)
        # ).to('cuda' if torch.cuda.is_available() else 'cpu'),
        AddLengths(),
        Pad(),  # pad both audio and text
        AudioUnsqueeze(),  # ! Tin add
    ]
)

# load datasets
# train_dataset = dataset_module.get_dataset(
#     config, transforms=transforms_train, part='train')
# val_dataset = dataset_module.get_dataset(
#     config, transforms=transforms_val, part='val')

# ! TIN Dataset
print("####### MODE: ########")
if args.mode == "word":
    train_dataset = Dataset_ctc(signals, ctc_labels)
    val_dataset = Dataset_ctc(signals_v, ctc_labels_v)
    print("using word mode")
elif args.mode == "sample":
    train_dataset = Dataset_ctc(signals, spacer_labels2)
    val_dataset = Dataset_ctc(signals_v, spacer_labels2_v)
    print("using sample mode")

elif args.mode == "compress":
    [signal_qs, label_qs, all_states] = compress_sig(spacer_labels2, signals)
    [signal_qs_v, label_qs_v, all_states_v] = compress_sig(spacer_labels2_v, signals_v)
    train_dataset = Dataset_ctc(signal_qs, ctc_labels)
    val_dataset = Dataset_ctc(signal_qs_v, ctc_labels_v)

    if args.custom:
        sig_segs_all = []
        sp_segs_all = []
        a = []
        for sig, sp in zip(signals, spacer_labels2):
            sig, sp = remove_bc(sig, sp)
            sig_segs, sp_segs, a1 = split_signal(sig, sp)
            sig_segs_all.extend(
                sig_segs
            )  # use append to have array of segments belongs to the original signal
            sp_segs_all.extend(sp_segs)
            a.extend(a1)

        sig_segs_all_v = []
        sp_segs_all_v = []
        a_v = []
        for sig, sp in zip(signals_v, spacer_labels2_v):
            sig, sp = remove_bc(sig, sp)
            sig_segs, sp_segs, a1_v = split_signal(sig, sp)
            sig_segs_all_v.extend(
                sig_segs
            )  # use append to have array of segments belongs to the original signal
            sp_segs_all_v.extend(sp_segs)
            a_v.extend(a1_v)

        # print(sig_segs_all[0])

        [sig_segs_all_qs, sp_segs_all_qs, all_states] = compress_sig(sig_segs_all, a)
        [sig_segs_all_qs_v, sp_segs_all_qs_v, all_states] = compress_sig(
            sig_segs_all_v, a_v
        )
        # print(a_v)
        # print(sig_segs_all_qs_v)
        plt.figure()
        plt.plot(sig_segs_all_qs_v[0])
        plt.savefig(f"fig/ctc_{sp_segs_all_v[0]}")

        train_dataset = Dataset_ctc(sig_segs_all_qs, sp_segs_all)
        val_dataset = Dataset_ctc(sig_segs_all_qs_v, sp_segs_all_v)

        # train_dataset = torch.load(args.customfiletrain)
        val_dataset = torch.load(args.customfile)
        print("#### USING CUSTOM TEST FILE ####")

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
    [signal_qs, label_qs, all_states] = compress_sig(spacer_labels2, signals)
    [signal_qs_v, label_qs_v, all_states_v] = compress_sig(spacer_labels2_v, signals_v)
    train_dataset = Dataset_ctc(signal_qs, label_qs)
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


# print("!!!", config.train.get('num_workers', 4))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train.get("batch_size", 1),
    collate_fn=no_pad_collate,
)


val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=no_pad_collate)

# print(getattr(quartznet_configs, config.model.name, '_quartznet5x5_config'))

model = QuartzNet(
    model_config=getattr(quartznet_configs, config.model.name, "_quartznet5x5_config"),
    **remove_from_dict(config.model, ["name"]),
)
# model = QuartzNet(
#     model_config=getattr(
#         quartznet_configs, config.model.name, '_quartznet5x5_config'),**config)


# print(model)
write_to_file(model, "model_structure.txt")

optimizer = torch.optim.Adam(model.parameters(), **config.train.get("optimizer", {}))
num_steps = len(train_dataloader) * config.train.get("epochs", 10)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)


f = []
mypath = args.checkpoint_path
for dirpath, dirnames, filenames in os.walk(mypath):
    f.extend(filenames)


if args.new:  # ! if new is specified then start from scratch
    print("##### FROM SCRATCH ######")
else:
    #! load best model by epoch
    best_model_epoch = [
        int(x.split("_")[1]) for x in f if x.split("_")[0] == args.model_name
    ]
    # print(best_model_epoch)
    best_model_name = mypath + "/" + f[best_model_epoch.index(max(best_model_epoch))]
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # ! load from checkpoint specified in the yaml
    print("##### CHECKPOINT ######")
    if config.train.get("from_checkpoint", None) is not None:
        print("load file specified in yaml file")
        # ! checkpoints folder contain checkpoint that is constraint by performance
        defined_model_name = "checkpoints/" + config.train.from_checkpoint
        model.load_weights(defined_model_name)
        print(f"load from checkpoint {defined_model_name}")
    else:  # ! if no checkpoint specified in yaml load the best checkpoint from epoch
        model.load_weights(best_model_name)  # best by epoch
        print(f"load from checkpoint {best_model_name}")


if torch.cuda.is_available():
    model = model.cuda()


criterion = nn.CTCLoss(blank=BLANK_VAL, reduction="mean", zero_infinity=True)
# criterion = nn.CTCLoss(blank=config.model.vocab_size)
decoder = GreedyDecoder(bpe=bpe, blank_index=BLANK_VAL, space_simbol=space_symbol)

prev_wer = 1000
#! TIN CHNAGE
if not args.eval:
    wandb.init(project=config.wandb.project, config=config, name=args.model_name)
    wandb.watch(model, log="all", log_freq=config.wandb.get("log_interval", 5000))
else:
    print("Only evaluating no logging")

# %%

if not args.eval:
    print("##### START TRAINING ######")
    for epoch_idx in tqdm(range(config.train.get("epochs", 10))):
        #! train: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # batch1 = batch
            # print(len(batch1['audio'][0]))
            batch = batch_transforms_train(batch)
            # batch2 = batch
            # print(len(batch2['audio'][0]))

            optimizer.zero_grad()
            logits = model(batch["audio"].float())

            output_length = torch.ceil(
                batch["input_lengths"].float() / model.stride
            ).int()

            # print('batch audio shape:', batch['audio'].shape, batch['audio'].dtype )
            # print('logits shape:', logits.shape)
            # print('output_length shape:',output_length)
            # print('target_lengths:',batch['target_lengths'])

            loss = criterion(
                logits.permute(2, 0, 1).log_softmax(dim=2),
                batch["text"],
                output_length,
                batch["target_lengths"],
            )  # target_length is the length of text of batch1 (before batch aug)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.get("clip_grad_norm", 15)
            )
            optimizer.step()
            lr_scheduler.step()
            # warmup_scheduler.dampen()
            # break
            if batch_idx % config.wandb.get("log_interval", 5000) == 0:
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
                step = (
                    epoch_idx * len(train_dataloader) * train_dataloader.batch_size
                    + batch_idx * train_dataloader.batch_size
                )
                #! TIN change
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_wer": wer,
                        "train_cer": cer,
                        "train_samples": wandb.Table(
                            columns=["gt_text", "pred_text"],
                            data=list(zip(target_strings, decoded_output)),
                        ),
                    },
                    step=step,
                )
            # print(decoded_output)
        #! #########################
        # validate:
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
            columns=["gt_text", "pred_text"],
            data=list(zip(target_strings, decoded_output)),
        )
        wandb.log(val_stats, step=step)

        # save model, TODO: save optimizer:

        #! unless defined in the yaml file with save in checkpoints
        if val_stats["wer"] < prev_wer:
            os.makedirs(
                config.train.get("checkpoint_path", "checkpoints"), exist_ok=True
            )
            prev_wer = val_stats["wer"]
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.train.get("checkpoint_path", "checkpoints"),
                    f"{args.model_name}_{epoch_idx}_{prev_wer}.pth",
                ),
            )
            # try:
            #     wandb.save(
            #         os.path.join(
            #             config.train.get("checkpoint_path", "checkpoints"),
            #             f"{args.model_name}_{epoch_idx}_{prev_wer}.pth",
            #         )
            #     )
            # except:
            #     print("could not use wandb save")

        #! no restriction saving
        os.makedirs(mypath, exist_ok=True)
        prev_wer = val_stats["wer"]
        torch.save(
            model.state_dict(),
            os.path.join(mypath, f"{args.model_name}_{epoch_idx}_{prev_wer}.pth"),
        )
    # try:
    #     wandb.save(
    #         os.path.join(mypath, f"{args.model_name}_{epoch_idx}_{prev_wer}.pth")
    #     )
    # except:
    #     print("could not use wandb save")
    # break
else:
    all_gt = []
    all_predict = []
    all_gt_predict = {}
    all_logits = []

    print("##### ONLY DO EVALUATION #####")
    # validate:
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

        # for checking
        all_gt.append(target_strings)
        all_predict.append(decoded_output)
        all_logits.append(logits.permute(0, 2, 1).softmax(dim=2))

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
        columns=["gt_text", "pred_text"],
        data=list(zip(target_strings, decoded_output)),
    )

    # print(val_stats["wer"])
    # raise Exception("STOP FOR CHECKING")

    torch.save(all_logits, "temp_data/logits.pt")
    all_gt_predict["gt"] = all_gt
    all_gt_predict["predict"] = all_predict

    letter_correct_count = 0
    seq_correct_count = 0
    for gt, predict in zip(all_gt, all_predict):
        gt = gt[0]
        predict = predict[0]
        letter_gt = re.findall(r"spacer_(.*?)_spacer", gt)
        letter_predict = re.findall(r"spacer_(.*?)_spacer", predict)
        if letter_gt == letter_predict:
            letter_correct_count += 1
        if gt == predict:
            seq_correct_count += 1

    print(f"letter acc: {letter_correct_count/len(all_gt)}")
    print(f"entire seq acc: {letter_correct_count/len(all_gt)}")

    df = pd.DataFrame.from_dict(all_gt_predict)
    df.to_csv(f"logfile/{best_model_name.split('/')[-1]}_eval.csv", index=False)
    print(val_stats)
