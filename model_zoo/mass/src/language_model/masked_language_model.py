# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Masked language model."""
import numpy as np

from .base import LanguageModel


class MaskedLanguageModel(LanguageModel):
    """
    Do mask operation on sentence.

    If k is assigned, then mask sentence with length k.
    Otherwise, use mask_ratio.

    Args:
        k (int): Length of fragment.
        mask_ratio (float): Mask ratio.
    """

    def __init__(self, k: int = None, mask_ratio=0.5,
                 mask_all_prob=None):
        super(MaskedLanguageModel, self).__init__()
        self.mask_ratio = mask_ratio
        self._k = k
        self._threshold = mask_all_prob

    def emit(self, sentence: np.ndarray, vocabulary):
        """
        Mask mono source sentence.

        A sample used to train model is processed with following step:

        encoder input (source): [x1, x2, x3, x4, x5, x6, x7, x8, </eos>]
        masked encoder input:   [x1, x2,  _,  _,  _, x6, x7, x8, </eos>]
        decoder input:          [  _, x3, x4]
                                  |   |   |
                                  V   V   V
        decoder output:         [ x3, x4, x5]

        Notes:
            A simple rule is made that source sentence starts without <BOS>
            but end with <EOS>.

        Args:
            vocabulary (Dictionary): Vocabulary.
            sentence (np.ndarray): Raw sentence instance.

        Returns:
            dict, an example.
        """
        encoder_input = sentence.copy()
        seq_len = encoder_input.shape[0]

        # If v=0, then u must equal to 0. [u, v)
        u, v = self._get_masked_interval(len(encoder_input),
                                         self._k, self._threshold)

        if u == 0:
            _len = v - u if v - u != 0 else seq_len
            decoder_input = np.array([vocabulary.mask_index] * _len, dtype=np.int32)
            decoder_input[1:] = encoder_input[:_len - 1].copy()
        else:
            decoder_input = np.array([vocabulary.mask_index] * (v - u), dtype=np.int32)
            decoder_input[1:] = encoder_input[u:v - 1].copy()

        if v == 0:
            decoder_output = encoder_input.copy()
            encoder_input[:] = vocabulary.mask_index
        else:
            decoder_output = encoder_input[u:v].copy()
            encoder_input[np.arange(start=u, stop=v)] = vocabulary.mask_index

        if u != v and u > 0:
            padding = np.array([vocabulary.padding_index] * u, dtype=np.int32)
            decoder_input = np.concatenate((padding, decoder_input))
            decoder_output = np.concatenate((padding, decoder_output))

        assert decoder_input.shape[0] == decoder_output.shape[0], "seq len must equal."

        return {
            "sentence_length": seq_len,
            "tgt_sen_length": decoder_output.shape[0],
            "encoder_input": encoder_input,  # end with </eos>
            "decoder_input": decoder_input,
            "decoder_output": decoder_output  # end with </eos>
        }

    def _get_masked_interval(self, length, fix_length=None,
                             threshold_to_mask_all=None):
        """
        Generate a sequence length according to length and mask_ratio.

        Args:
            length (int): Sequence length.

        Returns:
            Tuple[int, int], [start position, end position].
        """
        # Can not larger than sequence length.
        # Mask_length belongs to [0, length].
        if fix_length is not None:
            interval_length = min(length, fix_length)
        else:
            interval_length = min(length, round(self.mask_ratio * length))

        _magic = np.random.random()
        if threshold_to_mask_all is not None and _magic <= threshold_to_mask_all:
            return 0, length

        # If not sequence to be masked, then return 0, 0.
        if interval_length == 0:
            return 0, 0
        # Otherwise, return start position and interval length.
        start_pos = np.random.randint(low=0, high=length - interval_length + 1)
        return start_pos, start_pos + interval_length
