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
"""Noise channel language model."""
import numpy as np

from .base import LanguageModel


class NoiseChannelLanguageModel(LanguageModel):
    """Do mask on bilingual data."""

    def __init__(self, add_noise_prob: float = 0.1):
        super(NoiseChannelLanguageModel, self).__init__()
        self._noisy_prob = add_noise_prob

    def emit(self, sentence: np.ndarray, target: np.ndarray,
             mask_symbol_idx: int,
             bos_symbol_idx: int):
        """
        Add noise to sentence randomly.

        For example, given a sentence pair:
        source sentence:    [x1, x2, x3, x4, x5, x6, </eos>]
        target sentence:    [y1, y2, y3, y4, </eos>]

        After do random mask, data is looked like:
        encoder input (source): [x1, x2,  _, x4, x5,  _, </eos>]
        decoder input:          [<bos>,  y1,  y2,  y3,  y4]
                                   |    |    |    |    |
                                   V    V    V    V    V
        decoder output:         [ y1,  y2,  y3,  y4, </eos>]

        Args:
            sentence (np.ndarray): Raw sentence.
            target (np.ndarray): Target output (prediction).
            mask_symbol_idx (int): Index of MASK symbol.
            bos_symbol_idx (int): Index of bos symbol.

        Returns:
            dict, an example.
        """
        encoder_input = sentence.copy()
        tgt_seq_len = target.shape[0]

        for i, _ in enumerate(encoder_input):
            _prob = np.random.random()
            if _prob < self._noisy_prob:
                encoder_input[i] = mask_symbol_idx

        decoder_input = np.empty(shape=tgt_seq_len, dtype=np.int64)
        decoder_input[1:] = target[:-1]
        decoder_input[0] = bos_symbol_idx

        return {
            "sentence_length": encoder_input.shape[0],
            "tgt_sen_length": tgt_seq_len,
            "encoder_input": encoder_input,  # end with </eos>
            "decoder_input": decoder_input,  # start with <bos>
            "decoder_output": target  # end with </eos>
        }
