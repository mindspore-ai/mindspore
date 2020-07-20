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
"""Bilingual data loader."""
import numpy as np

from src.utils import Dictionary
from .base import DataLoader
from ..language_model.base import LanguageModel
from ..language_model.noise_channel_language_model import NoiseChannelLanguageModel


class BiLingualDataLoader(DataLoader):
    """Loader for bilingual data."""

    def __init__(self, src_filepath: str, tgt_filepath: str,
                 src_dict: Dictionary, tgt_dict: Dictionary,
                 src_lang: str, tgt_lang: str,
                 language_model: LanguageModel = NoiseChannelLanguageModel(add_noise_prob=0),
                 max_sen_len=66,
                 merge_dict=True):
        super(BiLingualDataLoader, self).__init__(max_sen_len)
        self._src_filepath = src_filepath
        self._tgt_filepath = tgt_filepath
        self._src_dict = src_dict
        self._tgt_dict = tgt_dict
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._lm = language_model
        self.max_sen_len = max_sen_len
        self.share_dict = merge_dict
        self._merge_dict()

    def _merge_dict(self):
        if self.share_dict:
            merged_dict = self._src_dict.merge_dict(self._tgt_dict,
                                                    new_dict=True)
            self._src_dict = merged_dict
            self._tgt_dict = merged_dict

    @property
    def src_dict(self):
        return self._src_dict

    @property
    def tgt_dict(self):
        return self._tgt_dict

    def _load(self):
        _min_len = 9999999999
        _max_len = 0
        unk_count = 0
        tokens_count = 0
        count = 0
        with open(self._src_filepath, "r") as _src_file:
            print(f" | Processing corpus {self._src_filepath}.")
            print(f" | Processing corpus {self._tgt_filepath}.")
            with open(self._tgt_filepath, "r") as _tgt_file:
                _min, _max = 9999999, -1
                for _, _pair in enumerate(zip(_src_file, _tgt_file)):
                    src_tokens = [
                        self._src_dict.index(t)
                        for t in _pair[0].strip().split(" ") if t
                    ]
                    tgt_tokens = [
                        self._tgt_dict.index(t)
                        for t in _pair[1].strip().split(" ") if t
                    ]
                    src_tokens.append(self._src_dict.eos_index)
                    tgt_tokens.append(self._tgt_dict.eos_index)
                    opt = self._lm.emit(
                        sentence=np.array(src_tokens, dtype=np.int64),
                        target=np.array(tgt_tokens, dtype=np.int64),
                        mask_symbol_idx=self._src_dict.mask_index,
                        bos_symbol_idx=self._tgt_dict.bos_index
                    )
                    src_len = opt["sentence_length"]
                    tgt_len = opt["tgt_sen_length"]

                    _min_len = min(_min_len, opt["sentence_length"], opt["tgt_sen_length"])
                    _max_len = max(_max_len, opt["sentence_length"], opt["tgt_sen_length"])

                    if src_len > self.max_sen_len or tgt_len > self.max_sen_len:
                        continue

                    src_padding = np.zeros(shape=self.max_sen_len, dtype=np.int64)
                    tgt_padding = np.zeros(shape=self.max_sen_len, dtype=np.int64)
                    for i in range(src_len):
                        src_padding[i] = 1
                    for j in range(tgt_len):
                        tgt_padding[j] = 1

                    tokens_count += opt["encoder_input"].shape[0]
                    tokens_count += opt["decoder_input"].shape[0]
                    tokens_count += opt["decoder_output"].shape[0]
                    unk_count += np.where(opt["encoder_input"] == self._src_dict.unk_index)[0].shape[0]
                    unk_count += np.where(opt["decoder_input"] == self._src_dict.unk_index)[0].shape[0]
                    unk_count += np.where(opt["decoder_output"] == self._src_dict.unk_index)[0].shape[0]

                    encoder_input = self.padding(opt["encoder_input"],
                                                 self._src_dict.padding_index)
                    decoder_input = self.padding(opt["decoder_input"],
                                                 self._tgt_dict.padding_index)
                    decoder_output = self.padding(opt["decoder_output"],
                                                  self._tgt_dict.padding_index)
                    if encoder_input is None or decoder_input is None or decoder_output is None:
                        continue

                    _min = np.min([np.min(encoder_input),
                                   np.min(decoder_input),
                                   np.min(decoder_output), _min])
                    _max = np.max([np.max(encoder_input),
                                   np.max(decoder_input),
                                   np.max(decoder_output), _max])

                    example = {
                        "src_padding": src_padding,
                        "tgt_padding": tgt_padding,
                        "src": encoder_input,
                        "prev_opt": decoder_input,
                        "prev_padding": tgt_padding,
                        "target": decoder_output
                    }
                    self._add_example(example)
                    count += 1

                print(f" | Shortest len = {_min_len}.")
                print(f" | Longest  len = {_max_len}.")
                print(f" | Total    sen = {count}.")
                print(f" | Total token num={tokens_count}, "
                      f"{unk_count / tokens_count * 100}% replaced by <unk>.")
