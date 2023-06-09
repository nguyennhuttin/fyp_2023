#
# This code copied from https://github.com/SeanNaren/deepspeech.pytorch.
#

#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.
    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, bpe, blank_index=0, space_simbol="▁", decoder_vis=False):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels = bpe.vocab()
        self.int_to_char = bpe.id_to_subword
        self.blank_index = blank_index
        self.space_simbol = space_simbol
        # To prevent errors in decode, we add an out of bounds index for the space
        space_index = None
        if self.space_simbol in labels:
            space_index = labels.index(self.space_simbol)
        else:
            raise ValueError("I wanna break free!!!")
        self.space_index = space_index

        if decoder_vis:
            print("label")
            print(self.labels)

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split(self.space_simbol) + s2.split(self.space_simbol))
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split(self.space_simbol)]
        w2 = [chr(word2char[w]) for w in s2.split(self.space_simbol)]

        return Lev.distance("".join(w1), "".join(w2)) / len(w1)

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1_concated, s2_concated = s1.replace(self.space_simbol, ""), s2.replace(
            self.space_simbol, ""
        )
        return Lev.distance(s1_concated, s2_concated) / len(s1)

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1_concated, s2_concated = s1.replace(self.space_simbol, ""), s2.replace(
            self.space_simbol, ""
        )
        return Lev.distance(s1_concated, s2_concated) / len(s1)

    def convert_to_strings(
        self, sequences, sizes=None, remove_repetitions=False, return_offsets=False
    ):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(
                sequences[x], seq_len, remove_repetitions
            )
            strings.append(string)  # We only return one path
            if return_offsets:
                offsets.append(string_offsets)
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ""
        offsets = []
        for i in range(size):
            char = self.int_to_char(sequence[i].item())
            if char != self.int_to_char(self.blank_index):
                # if this char is a repetition and remove_repetitions=true, then skip
                if (
                    remove_repetitions
                    and i != 0
                    and char == self.int_to_char(sequence[i - 1].item())
                ):
                    pass
                elif char == self.labels[self.space_index]:
                    string += self.space_simbol
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(
        self,
        bpe,
        lm_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_index=0,
        space_simbol="spacer",
    ):
        self.labels = labels = bpe.vocab()
        # print(self.labels)

        super(BeamCTCDecoder, self).__init__(
            bpe=bpe, blank_index=blank_index, space_simbol=space_simbol
        )
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(
            labels,
            model_path=lm_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob,
            beam_width=beam_width,
            num_processes=num_processes,
            blank_id=self.blank_index,
        )

    def convert_to_strings_ctc(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = "".join(
                        map(lambda x: self.int_to_char(x.item()), utt[0:size])
                    )
                else:
                    transcript = ""
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor_ctc(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)
        # print(scores)
        strings = self.convert_to_strings_ctc(out, seq_lens)
        strings = [item[0] for item in strings]
        # offsets = self.convert_tensor_ctc(offsets, seq_lens)
        return strings


class GreedyDecoder(Decoder):
    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.
        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(
            max_probs.view(max_probs.size(0), max_probs.size(1)),
            sizes,
            remove_repetitions=True,
            return_offsets=True,
        )
        return strings
