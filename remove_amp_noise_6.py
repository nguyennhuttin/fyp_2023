import numpy as np
from helpers import vstack

# Input data types:
# signal: (1 x n) np.array
# amp_range: float
# state_len: int
# min_samples: int

# Output data types:
# signal_q: (1 x n) np.array
# states: (n x 2) np.array
# sections_x: (n x 2) np.array


def remove_amp_noise_6(signal, amp_range, state_len, min_samples):
    signal_q = np.array([])
    states = np.array([])       # does not contain 'noisy' states
    sections_x = np.array([])   # contains all states

    curr_str = 0
    curr_len = 1
    curr_sum = signal[0]
    i = 1  # start from second sample

    while i < len(signal):
        # calculate mean of current section, without noisy outliers
        curr_mean = curr_sum / curr_len

        # check if within range
        if np.abs(curr_mean - signal[i]) < amp_range:
            # add to section
            curr_sum = curr_sum + signal[i]
            curr_len = curr_len + 1
        else:
            # add section to array
            if curr_len >= min_samples:
                signal_q = np.concatenate((signal_q, curr_mean*np.ones(state_len)))
                states = vstack(states, np.array([curr_str, curr_str + curr_len - 1]))

                # start new section
                sections_x = vstack(sections_x, np.array([curr_str, curr_str + curr_len - 1]))
                curr_str = i
                curr_len = 1
                curr_sum = signal[i]
				
                if states.ndim == 1:
                    states = np.array([states])
                if sections_x.ndim == 1:
                    sections_x = np.array([sections_x])
            else:
                sections_x = vstack(sections_x, np.array([curr_str, curr_str + curr_len - 1]))
                # should continue the previous section
                if len(signal_q) > 0 and np.abs(signal_q[-1] - signal[i]) < amp_range:
                    curr_str = states[len(signal_q) - 1, 0]
                    temp_len = states[len(signal_q) - 1, 1] - states[len(signal_q) - 1, 0] + 1
                    curr_sum = signal_q[-1] * (temp_len + curr_len) + signal[i]
                    curr_len = temp_len + curr_len + 1

                    # remove last section
                    signal_q = signal_q[:-1]
                    states = states[:-1]
                else:  # start new section
                    curr_str = i
                    curr_len = 1
                    curr_sum = signal[i]
        i = i + 1

    # last section
    if curr_len > 0:
        sections_x = vstack(sections_x, np.array([curr_str, curr_str + curr_len - 1]))
        if curr_len >= min_samples:
            curr_mean = curr_sum / curr_len
            states = vstack(states, np.array([curr_str, curr_str + curr_len - 1]))
            signal_q = np.concatenate((signal_q, curr_mean*np.ones(state_len)))

    return signal_q, states, sections_x
