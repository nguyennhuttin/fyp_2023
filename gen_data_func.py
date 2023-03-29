# adding noise to the signal, python version of the MATLAB code which was sent to me
# necessary imports
from scipy import signal as scipy_signal
try:
    from IPython.display import clear_output
except:
    pass
import re
import numpy as np
import pandas as pd
# from sklearn.utils import shuffle
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import randomizeSequence as rSeq
import ScrappySequence as ss
import math
from re import L
from tqdm import tqdm


def generate_samples(mean_curr, stdv_curr, dwell):
    signal_s = []
    index_s = []
    for i in range(0, len(mean_curr)):
        p = 1/dwell[i]
        if p > 1:
            p = 1
        r = np.random.geometric(p=p) + 1
        signal_s = np.concatenate(
            (signal_s, stdv_curr[i]*np.random.normal(size=r)+mean_curr[i]))
        index_s = np.concatenate((index_s, i*np.ones(r)))
    return signal_s, index_s

# reading the reference signals from the scrappie squiggle outputs


def read_data(datafile_path):  # 'datafile_path' refers to the squiggle output file in .csv format

    currents = []  # list of all reference currents
    std_dev = []
    dwells = []
    comments = []
    sequences = []

    sequence = []
    curr = []
    dev = []
    dw = []

    with open(datafile_path, 'r') as f:
        for line in f:

            if line.startswith('#'):
                comments.append(line)

            elif line.startswith('pos'):
                if len(curr) == 0:
                    continue
                sequences.append(''.join(sequence))
                currents.append(curr)
                std_dev.append(dev)
                dwells.append(dw)

                sequence = []
                curr = []
                dev = []
                dw = []

            else:
                parts = line.split(',')
                assert len(parts) == 5
                sequence.append(parts[1])
                curr.append(float(parts[2]))
                dev.append(float(parts[3]))
                dw.append(float(parts[4]))

    sequences.append(''.join(sequence))
    currents.append(curr)
    std_dev.append(dev)
    dwells.append(dw)

    return sequences, comments, currents, std_dev, dwells


def generate_label(indexes, spacer_len=6, letter_len=7):
    label = np.zeros(len(indexes))

    is_in_letter = True
    remaining = letter_len
    previous = 0
    for i, v in enumerate(indexes):  # indexes 00001111

        if v != previous:
            remaining -= 1
            previous = v
            if remaining == 0:
                if is_in_letter:
                    remaining = spacer_len
                else:
                    remaining = letter_len
                is_in_letter = not is_in_letter

        if is_in_letter:
            label[i] = 0
        else:
            label[i] = 1

#is_in_letter = T
#remaining = 6
#previous = -1
#  L     L    L      L L L L S S S S  S  S  L  L  L  L  L  L  L  S  S  S  S  S  S  L  L  L  L  L  L  L  S
    # 00000 1111 222222 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
    # 0     0000 000000 0 0 0 0 1 1 1 1  1  1  0
    return label


def generate_spacer_label(indexes, spacer_len=6, letter_len=7):
    indexes = list(indexes)

    first_spacer_base = indexes.index(56)
    last_spacer_base = len(indexes) - 1 - indexes[::-1].index(152)

    label = np.zeros(len(indexes))

    is_in_spacer = True
    remaining = spacer_len
    previous = 56
    for i, v in enumerate(indexes[first_spacer_base:last_spacer_base+1]):

        if v != previous:
            remaining -= 1
            previous = v
            if remaining == 0:
                if is_in_spacer:
                    remaining = letter_len
                else:
                    remaining = spacer_len
                is_in_spacer = not is_in_spacer

        if is_in_spacer:
            label[first_spacer_base + i] = 1
        else:
            label[first_spacer_base + i] = 0

    return label


# a comment is of this shape
#  #Seq[1. 4.3.42.19.61.13.26.40.9],,,,
#  where '1.' is the sequence number
#  '4' is the forward barcode
#  '3' is the reverse barcode
#  and the remaining numbers are letters
def parse_comment(comment):

    match = re.search(
        "^#Seq\[(\d+\.) (\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\],,,,$", comment)
    assert match is not None, "The comment string is not in the correct format"
    groups = list(match.groups())

    assert len(groups) == 10, f"{groups} {comment}"

    sequence_no = groups[0]
    barcodes = groups[1:3]
    letters = groups[3:10]
    assert len(barcodes) == 2 and len(
        letters) == 7, f"groups: {groups} barcodes: {len(barcodes)} letters: {len(letters)}"
    return sequence_no, [int(i) for i in barcodes], [int(i) for i in letters]


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    result.append(item)
    result.insert(0, item)
    return result

# 'num_samples_per_reference' refers to number of noisy signals generated from a single reference signal


def generate_data2(sequences, comments, currents, std_dev, dwells, num_samples_per_reference):
    assert len(sequences) == len(comments) == len(
        currents) == len(std_dev) == len(dwells)

    signals = [0]*(len(currents)*num_samples_per_reference)
    index = []
    spacer_labels = [0]*(len(currents)*num_samples_per_reference)
    letter_labels = [0]*(len(currents)*num_samples_per_reference)
    barcode_labels = [0]*(len(currents)*num_samples_per_reference)
    ctc_labels = [0]*(len(currents)*num_samples_per_reference)

    for i in tqdm(range(0, len(currents))):  # 'len(currents)' -> number of reference signals
        seq = ss.ScrappySequence(
            sequences[i], currents[i], std_dev[i], dwells[i])
        # remove initial and final 8As
        seq = ss.ScrappySequence.fromSequenceItems(seq.slice(8, len(seq) - 8))

        # clear_output()
        # print("Sequence", i + 1, "of", len(currents))

        _, barcode_label, letter_label = parse_comment(comments[i])
        ctc_label = intersperse(letter_label, 256)

        for j in range(0, num_samples_per_reference):
            # noisy signal generation for each instance
            signal_s, index_s = generate_samples(
                seq.currents, seq.sds, seq.dwells)
            spacer_label = generate_spacer_label(index_s)

            signals[i * num_samples_per_reference + j] = signal_s
            spacer_labels[i * num_samples_per_reference + j] = spacer_label
            letter_labels[i * num_samples_per_reference + j] = letter_label
            barcode_labels[i * num_samples_per_reference + j] = barcode_label
            ctc_labels[i * num_samples_per_reference + j] = ctc_label

    return signals, index, spacer_labels, letter_labels, barcode_labels, ctc_labels

# generating training data


def prepare_train2(dataset_path, num_samples_per_reference):
    sequences, comments, currents, std_dev, dwells = read_data(
        dataset_path)  # read the reference signals
    signals, index, spacer_labels, letter_labels, barcode_labels, ctc_labels = generate_data2(
        sequences, comments, currents, std_dev, dwells, num_samples_per_reference)  # generate the noisy dataset from reference signals
    return signals, index, spacer_labels, letter_labels, barcode_labels, ctc_labels

# processed the noisy signals to obtain fixed size vectors for each instance
# truncated the signals to size of minimum length signal


def resize(sig, size):
    return scipy_signal.resample(sig, size, t=None, axis=0, window=None)


avg_signal_len = 0


def process_data(signals, labels):
    global avg_signal_len

    # randomizer = rSeq.SignalRandomizer().add_samples_in_big_jumps()
    # randomizer.build()
    # print("Randomizing signals...")
    # for i, signal in enumerate(signals):
    #   signals[i] = randomizer.randomize(signal)

    min = 1000
    print("Calculating average length...")

    sizeSum = sum((len(i) for i in signals))
    avgLen = round(sizeSum / len(signals))
    print("Average signal length:", avgLen)
    # avg_signal_len = 2342
    avg_signal_len = 1928

    print("Resizing signals to", avg_signal_len, "...")
    resizer = rSeq.SignalRandomizer().resize(length=avg_signal_len)
    resizer.build()
    for i, signal in enumerate(signals):
        signals[i] = resizer.randomize(signal)
        labels[i] = resizer.randomize(labels[i])


avg_signal_len2 = 0


def process_data2(signals):
    global avg_signal_len2

    # randomizer = rSeq.SignalRandomizer().add_samples_in_big_jumps()
    # randomizer.build()
    # print("Randomizing signals...")
    # for i, signal in enumerate(signals):
    #   signals[i] = randomizer.randomize(signal)

    min = 1000
    print("Calculating average length...")

    sizeSum = sum((len(i) for i in signals))
    avgLen = round(sizeSum / len(signals))
    print("Average signal length:", avgLen)
    avg_signal_len2 = avgLen
    # avg_signal_len = avgLen

    print("Resizing signals to", avg_signal_len2, "...")
    resizer = rSeq.SignalRandomizer().resize(length=avg_signal_len2)
    resizer.build()
    for i, signal in enumerate(signals):
        signals[i] = resizer.randomize(signal)
        # labels[i] = resizer.randomize(labels[i])
