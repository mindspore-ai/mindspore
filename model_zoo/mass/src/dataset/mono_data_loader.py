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
"""Mono data loader."""
import numpy as np

from src.utils import Dictionary

from .base import DataLoader
from .schema import SCHEMA
from ..language_model.base import LanguageModel
from ..language_model import LooseMaskedLanguageModel


class MonoLingualDataLoader(DataLoader):
    """Loader for monolingual data."""
    _SCHEMA = SCHEMA

    def __init__(self, src_filepath: str, lang: str, dictionary: Dictionary,
                 language_model: LanguageModel = LooseMaskedLanguageModel(mask_ratio=0.3),
                 max_sen_len=66, min_sen_len=16):
        super(MonoLingualDataLoader, self).__init__(max_sen_len=max_sen_len)
        self._file_path = src_filepath
        self._lang = lang
        self._dictionary = dictionary
        self._lm = language_model
        self.max_sen_len = max_sen_len
        self.min_sen_len = min_sen_len

    @property
    def dict(self):
        return self._dictionary

    def generate_padding_mask(self, sentence, length, exclude_mask=False):
        """Generate padding mask vector."""
        src_padding = np.zeros(shape=self.max_sen_len, dtype=np.int64)
        if exclude_mask:
            pos = np.where(sentence == self._dictionary.padding_index)[0]
        else:
            pos = np.where((sentence == self._dictionary.padding_index) | (sentence == self._dictionary.mask_index))[0]
        src_padding[0:length] = 1
        if pos.shape[0] != 0:
            src_padding[pos] = 0
        return src_padding

    def _load(self):
        _min_len = 9999999999
        _max_len = 0
        count = 0
        with open(self._file_path, "r") as _file:
            print(f" | Processing corpus {self._file_path}.")
            for _, _line in enumerate(_file):
                tokens = [self._dictionary.index(t.replace(" ", ""))
                          for t in _line.strip().split(" ") if t]
                # In mass code, it doesn't add <BOS> to sen.
                tokens.append(self._dictionary.eos_index)
                opt = self._lm.emit(sentence=np.array(tokens, dtype=np.int32),
                                    vocabulary=self._dictionary)

                src_len = opt["sentence_length"]
                _min_len = min(_min_len, opt["sentence_length"], opt["tgt_sen_length"])
                _max_len = max(_max_len, opt["sentence_length"], opt["tgt_sen_length"])

                if src_len > self.max_sen_len:
                    continue
                if src_len < self.min_sen_len:
                    continue

                src_padding = self.generate_padding_mask(opt["encoder_input"],
                                                         opt["sentence_length"],
                                                         exclude_mask=False)
                tgt_padding = self.generate_padding_mask(opt["decoder_input"],
                                                         opt["tgt_sen_length"],
                                                         exclude_mask=True)

                encoder_input = self.padding(opt["encoder_input"],
                                             self._dictionary.padding_index)
                decoder_input = self.padding(opt["decoder_input"],
                                             self._dictionary.padding_index)
                decoder_output = self.padding(opt["decoder_output"],
                                              self._dictionary.padding_index)
                if encoder_input is None or decoder_input is None or decoder_output is None:
                    continue

                example = {
                    "src": encoder_input,
                    "src_padding": src_padding,
                    "prev_opt": decoder_input,
                    "prev_padding": tgt_padding,
                    "target": decoder_output,
                    "tgt_padding": tgt_padding,
                }
                self._add_example(example)
                count += 1

        print(f" | Shortest len = {_min_len}.")
        print(f" | Longest  len = {_max_len}.")
        print(f" | Total    sen = {count}.")
