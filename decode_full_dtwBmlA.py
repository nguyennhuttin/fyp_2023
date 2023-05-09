import sys
#this is necessary for the helpers module to be correctly imported
sys.path.append("../")

import numpy as np
import scipy.io
import os
from os import listdir
from os.path import isfile, join
from helpers import *
import time
#from dtw import dtw
from dtaidistance import dtw as dtwfast
import json
import multiprocessing
import cProfile
import matplotlib.pyplot as plt
import copy

from pp.pp_get_sig_set_1 import pp_get_sig_set_1
from pp.remove_amp_noise_6 import remove_amp_noise_6
from dec.decode_spiky_spacer_2 import decode_spiky_spacer_2
from dec.decode_barcode_x import decode_barcode_x
from dec.decode_barcode_y import decode_barcode_y

import alphabet_classifier.alphabet_classifier as ac
import numpy as np
import random
from scipy.io import loadmat

import cython
import argparse

parser = argparse.ArgumentParser(description='Nucleotrace Decoder')
parser.add_argument("--fast5_subdir", help="The supplied fast5 sub directories", default="")
parser.add_argument("--github_subdir", help="The supplied github nanopore decoder directory", default="")
args = parser.parse_args()
print(args.fast5_subdir)

# load ML 
model_filename = args.github_subdir + "/alphabet_classifier/alphabet_classifier_model.pt"
classifier = ac.AlphabetClassifier(model_filename, "cpu")

# load barcode alphabets
data = scipy.io.loadmat(args.github_subdir + '/code/code_alphabets/L24_F_BX_8.mat')
#data = scipy.io.loadmat('/Users/zacc/github_repo/nanopore_decoder/code/code_alphabets/L24_F_BX_8.mat')
f_bx_set = [arr[0] for arr in data['F_BX_AVG'][0]]
data = scipy.io.loadmat(args.github_subdir + '/code/code_alphabets/L24_R_BX_8.mat')
#data = scipy.io.loadmat('/Users/zacc/github_repo/nanopore_decoder/code/code_alphabets/L24_R_BX_8.mat')
r_bx_set = [arr[0] for arr in data['R_BX_SHORT'][0]]

# load data alphabet (create with new signatures)
curr_alph = scipy.io.loadmat(args.github_subdir + '/code/code_alphabets/F256_K10_AA_23_N.mat')['curr_alph'][0]

# load data and barcode sequences (create with all variables)
codewords = scipy.io.loadmat(args.github_subdir + '/code/codewords/PROD_210927_F256_AA_N.mat')['t_seq']
#codewords = scipy.io.loadmat('/Users/zacc/github_repo/nanopore_decoder/code/codewords/PROD_210927_F256_AA_N.mat')['t_seq']

# output files for parallel processing
dir1 = args.fast5_subdir
filename1 = os.path.basename(dir1)
decode1_filename =  str('decode_results_' + str(filename1) + '.txt')
decode2_filename =  str('temp_' + str(filename1) + '.txt')
op_dir = '../read_test_files/temp_decoded/'
op_file1 = op_dir + decode1_filename
op_file2 = op_dir + decode2_filename

os.makedirs(os.path.dirname(op_file1), exist_ok=True)
os.makedirs(os.path.dirname(op_file2), exist_ok=True)

# decoding parameters, variables for statistics
# main parameters
normalize = 1
preprocess = 1
use_bump_flat = 1

# pre-processing parameters
lim_gap = 40
bump_range = 30
sig_lim = 1000
cont_thres = 4
mid_gap = 10
pp_par = [lim_gap, bump_range, sig_lim, cont_thres, mid_gap]

# flat region detection parameters
bp_amp_range = 15
bp_amp_gap = 10
bp_noise_len = 0
bp_req_spc_cnt = 1
bp_pld_l_lim = 0
bp_spc_l_lim = 40
bp_plot_req = 0
bp_par = [bp_amp_range, bp_amp_gap, bp_noise_len, bp_req_spc_cnt, bp_pld_l_lim, bp_spc_l_lim, bp_plot_req]

# time-compression parameters
state_len = 1
amp_diff_st = 0.8
min_samples_st = 2

# spiky spacer detection parameters
level_diff = 1.8
filt_st = 3
pld_l_lim = 60
pld_u_lim = 50000
N = 6

# barcode decoding parametes
bx_min_len = 10

# variables for decoding statistics
f_cnt = 0 
miss_cnt = 0
dec_est_cnt = 0
dec_est = np.array([])
bx_est_cnt = 0
bx_est = np.array([])
stat_cnt = 10

def add_estimates_to_list(f_bx_est, r_bx_est, spc_est, cdw_est):
    global bx_est_cnt, bx_est, dec_est_cnt, dec_est
    est_len = 13
	
    # locate barcode estimate (not used anywhere else, could be useful for debugging)
    ch = 0
    for i in range(bx_est_cnt):
        if f_bx_est == bx_est[i, 0] and r_bx_est == bx_est[i, 1]:
            ch = 1
            bx_est[i, 2] = bx_est[i, 2] + 1
    # save new estimate
    if ch == 0:
        bx_est = vstack(bx_est, np.array([f_bx_est, r_bx_est, 1]))
        if bx_est.ndim == 1:
            bx_est = np.array([bx_est])
        bx_est_cnt = bx_est_cnt + 1

    # locate full decoding estimate
    full_est = np.concatenate((np.array([f_bx_est, r_bx_est]), np.squeeze(spc_est), np.array(cdw_est)))
    ch = 0
    for i in range(dec_est_cnt):
        truth_arr = np.equal(full_est, dec_est[i, 0 : est_len])
        if all(truth_arr):
            ch = 1
            dec_est[i, est_len] = dec_est[i, est_len] + 1
    # save new estimate
    if ch == 0:
        dec_est = vstack(dec_est, np.concatenate((full_est, np.array([1]))))
        if dec_est.ndim == 1:
            dec_est = np.array([dec_est])
        dec_est_cnt = dec_est_cnt + 1		
		


# helper functions for decoding symbols
def decode_information_symbols(signal, sections, states):
    #manhattan_distance = lambda x, y: np.abs(x - y)
	
    # decode information symbols
    payloads = []
    for i in range(5):
        start_pos = states[int(sections[i, 1]), 1] + 1
        end_pos = states[int(sections[i + 1, 0]), 0] - 1
        if start_pos <= end_pos:
            payloads.append(signal[start_pos: end_pos + 1])
        else:
            payloads.append([])

    best_est = []
    for i in range(5):
        est_pos = 0
        est_cost = 10 ** 10
        for j in range(len(curr_alph)):
            payload = payloads[i]
            signature = curr_alph[j]['sig'][0][0][0]
            #[temp_cost, _, _, _] = dtw(payload, signature, dist=manhattan_distance)
            temp_cost = dtwfast.distance_fast(payload, signature, use_pruning=True)

            if temp_cost < est_cost:
                est_cost = temp_cost
                est_pos = j

        # since python starts indexing with zero
        best_est.append(est_pos + 1)
    return best_est

# helper functions for decoding symbols using ML classifier
def decode_information_symbols_ml(signal, sections, states):

    # decode information symbols
    payloads = []
    for i in range(5):
        start_pos = states[int(sections[i, 1]), 1] + 1
        end_pos = states[int(sections[i + 1, 0]), 0] - 1
        if start_pos <= end_pos:
            payloads.append(signal[start_pos: end_pos + 1])
        else:
            payloads.append([])

    best_est = classifier.classify_batch(payloads) + 1
    
    return best_est


# decode a raw read
def decode_raw_reads(read_dir):
    # read directory
    mat_files = [f for f in listdir(read_dir) if isfile(join(read_dir, f))]
	
    global f_cnt, miss_cnt, dec_est_cnt, dec_est, bx_est_cnt, bx_est
    est_len = 13

    for n in range(len(mat_files)):
        # load raw read from file
        file_dir = read_dir + mat_files[n]
        print(file_dir)

        data = np.load(file_dir, allow_pickle=True).item()
        signal = np.array(data['signal'])

        # preprocess
        if preprocess == 1:
            sig_set = pp_get_sig_set_1(signal, use_bump_flat, normalize, pp_par, bp_par)
        else:
            if normalize == 1:
                signal = (signal - np.mean(signal)) / unbiased_std(signal)
            sig_set = [signal]

        for signal in sig_set:
            f_cnt = f_cnt + 1
			
            # locate spacers
            [signal_q, states, _] = remove_amp_noise_6(signal, amp_diff_st, state_len, min_samples_st)
            [sections, spc_est, states, temp_err] = decode_spiky_spacer_2(signal_q, states, level_diff, pld_l_lim,
                                                                           pld_u_lim, filt_st)
            if temp_err <= 0:
                spc_est = np.reshape(spc_est, (1, 6))
                #print(sections)
				
                # decode forward barcode
                f_bx_est = 0
                if states[int(sections[0, 0]), 0] > 1:
                    f_bx_sig = signal[0 : states[int(sections[0, 0]), 0]]
                    f_bx_set_cpy = copy.deepcopy(f_bx_set)
                    [f_bx_est, _] = decode_barcode_x(f_bx_sig, f_bx_set_cpy, bx_min_len, 1)
				
                # decode reverse barcode
                r_bx_est = 0
                if states[int(sections[5, 1]), 1] < len(signal):
                    r_bx_sig = signal[states[int(sections[5, 1]), 1] + 1: len(signal)]
                    [r_bx_est, _] = decode_barcode_x(r_bx_sig, r_bx_set, bx_min_len, 0)

                # decode data symbols
                # cdw_est = decode_information_symbols(signal, sections, states)
                cdw_est = decode_information_symbols_ml(signal, sections, states)

                # add estimates to list
                add_estimates_to_list(f_bx_est, r_bx_est, np.squeeze(spc_est), cdw_est)

                # full decoding estimate
                full_est = np.concatenate((np.array([f_bx_est, r_bx_est]), np.squeeze(spc_est), cdw_est))
                print("Full Estimate:", full_est)
            else:
                miss_cnt = miss_cnt + 1

        if (n + 1) % stat_cnt == 0:
            print_stats(n + 1)

    print_stats(n + 1)


# print decoding statistics
def print_stats(n):
    global f_cnt, miss_cnt, dec_est, dec_est_cnt, codewords
    est_len = 13
	
    corr_cnt = 0
    nbxc_cnt = 0
    t_seq_cnt = np.zeros(codewords.shape[0])
    b_seq_cnt = np.zeros(codewords.shape[0])

    for i in range(dec_est_cnt):
        for j in range(codewords.shape[0]):
            truth_arr = np.equal(dec_est[i, 0 : 13], codewords[j, 0 : 13])
            if all(truth_arr):
                t_seq_cnt[j] = t_seq_cnt[j] + dec_est[i, 13]
                corr_cnt = corr_cnt + dec_est[i, 13]
				
            truth_arr = np.equal(dec_est[i, 2 : 13], codewords[j, 2 : 13])
            if all(truth_arr):
                b_seq_cnt[j] = b_seq_cnt[j] + dec_est[i, 13]
                nbxc_cnt = nbxc_cnt + dec_est[i, 13]

    corr_perc = corr_cnt / f_cnt
    nbxc_perc = nbxc_cnt / f_cnt

    print('Progress: (', n, ') files, (', f_cnt, ') signals, (', miss_cnt, ') misses ')
    if miss_cnt > 0:
        print('[', f_cnt / miss_cnt, ']')
    print('Corr: (', corr_cnt, ')[', corr_perc, ']')
    print('NBXC: (', nbxc_cnt, ')[', nbxc_perc, ']')
    print(dec_est)

    ind_corr_perc = []
    for i in range(codewords.shape[0]):
        ind_corr_perc.append(t_seq_cnt[i] / f_cnt)

    # save results
    #with open('decode_results.txt', 'a') as outfile:
    with open(op_file1, 'a') as outfile:
        json.dump({
            'f_cnt': f_cnt,
            'miss_cnt': miss_cnt,
            'corr_cnt': corr_cnt,
            'corr_perc': corr_perc,
            'nbxc_perc': nbxc_perc
        }, outfile)

    original_stdout = sys.stdout
    #with open('temp.txt', 'w') as f:
    with open(op_file2, 'w') as f:    
        sys.stdout = f
        for i in range(dec_est_cnt):
            for j in range(14):
                print(dec_est[i, j], end='')
            print()
        print('===============')
        sys.stdout = original_stdout

if __name__ == "__main__":
    t0 = time.time()
    decode_raw_reads(args.fast5_subdir)
    t1 = time.time()
    print('Time elapsed:', round(t1 - t0, 2), ' seconds')