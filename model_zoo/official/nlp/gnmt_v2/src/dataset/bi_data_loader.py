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

from .base import DataLoader
from .tokenizer import Tokenizer


class BiLingualDataLoader(DataLoader):
    """Loader for bilingual data."""

    def __init__(self,
                 src_filepath: str,
                 tgt_filepath: str,
                 tokenizer: Tokenizer,
                 min_sen_len=0,
                 source_max_sen_len=None,
                 target_max_sen_len=80,
                 schema_address=None):
        super(BiLingualDataLoader, self).__init__()
        self._src_filepath = src_filepath
        self._tgt_filepath = tgt_filepath
        self.tokenizer = tokenizer
        self.min_sen_len = min_sen_len
        self.source_max_sen_len = source_max_sen_len
        self.target_max_sen_len = target_max_sen_len
        self.schema_address = schema_address

    def _load(self):
        count = 0
        if self.source_max_sen_len is None:
            with open(self._src_filepath, "r") as _src_file:
                print(f" | count the max_sen_len of corpus {self._src_filepath}.")
                max_src = 0
                for _, _pair in enumerate(_src_file):
                    src_tokens = [
                        int(self.tokenizer.tok2idx[t])
                        for t in _pair.strip().split(" ") if t
                    ]
                    src_len = len(src_tokens)
                    if src_len > max_src:
                        max_src = src_len
                self.source_max_sen_len = max_src + 2

        if self.target_max_sen_len is None:
            with open(self._src_filepath, "r") as _tgt_file:
                print(f" | count the max_sen_len of corpus {self._src_filepath}.")
                max_tgt = 0
                for _, _pair in enumerate(_tgt_file):
                    src_tokens = [
                        int(self.tokenizer.tok2idx[t])
                        for t in _pair.strip().split(" ") if t
                    ]
                    tgt_len = len(src_tokens)
                    if tgt_len > max_tgt:
                        max_tgt = tgt_len
                self.target_max_sen_len = max_tgt + 1

        with open(self._src_filepath, "r") as _src_file:
            print(f" | Processing corpus {self._src_filepath}.")
            print(f" | Processing corpus {self._tgt_filepath}.")
            with open(self._tgt_filepath, "r") as _tgt_file:
                for _, _pair in enumerate(zip(_src_file, _tgt_file)):

                    src_tokens = [
                        int(self.tokenizer.tok2idx[t])
                        for t in _pair[0].strip().split(" ") if t
                    ]
                    tgt_tokens = [
                        int(self.tokenizer.tok2idx[t])
                        for t in _pair[1].strip().split(" ") if t
                    ]
                    src_tokens.insert(0, self.tokenizer.bos_index)
                    src_tokens.append(self.tokenizer.eos_index)
                    tgt_tokens.insert(0, self.tokenizer.bos_index)
                    tgt_tokens.append(self.tokenizer.eos_index)
                    src_tokens = np.array(src_tokens)
                    tgt_tokens = np.array(tgt_tokens)
                    src_len = src_tokens.shape[0]
                    tgt_len = tgt_tokens.shape[0]

                    if (src_len > self.source_max_sen_len) or (src_len < self.min_sen_len) or (
                            tgt_len > (self.target_max_sen_len + 1)) or (tgt_len < self.min_sen_len):
                        print(f"+++++ delete! src_len={src_len}, tgt_len={tgt_len - 1}, "
                              f"source_max_sen_len={self.source_max_sen_len},"
                              f"target_max_sen_len={self.target_max_sen_len}")
                        continue
                    # encoder inputs
                    encoder_input = self.padding(src_tokens, self.tokenizer.padding_index, self.source_max_sen_len)
                    src_padding = np.zeros(shape=self.source_max_sen_len, dtype=np.int64)
                    for i in range(src_len):
                        src_padding[i] = 1
                    # decoder inputs
                    decoder_input = self.padding(tgt_tokens[:-1], self.tokenizer.padding_index, self.target_max_sen_len)
                    # decoder outputs
                    decoder_output = self.padding(tgt_tokens[1:], self.tokenizer.padding_index, self.target_max_sen_len)
                    tgt_padding = np.zeros(shape=self.target_max_sen_len + 1, dtype=np.int64)
                    for j in range(tgt_len):
                        tgt_padding[j] = 1
                    tgt_padding = tgt_padding[1:]
                    decoder_input = np.array(decoder_input, dtype=np.int64)
                    decoder_output = np.array(decoder_output, dtype=np.int64)
                    tgt_padding = np.array(tgt_padding, dtype=np.int64)

                    example = {
                        "src": encoder_input,
                        "src_padding": src_padding,
                        "prev_opt": decoder_input,
                        "target": decoder_output,
                        "tgt_padding": tgt_padding
                    }
                    self._add_example(example)
                    count += 1

                print(f" | source padding_len = {self.source_max_sen_len}.")
                print(f" | target padding_len = {self.target_max_sen_len}.")
                print(f" | Total  activate  sen = {count}.")
                print(f" | Total  sen = {count}.")

                if self.schema_address is not None:
                    provlist = [count, self.source_max_sen_len, self.source_max_sen_len,
                                self.target_max_sen_len, self.target_max_sen_len, self.target_max_sen_len]
                    columns = ["src", "src_padding", "prev_opt", "target", "tgt_padding"]
                    with open(self.schema_address, "w", encoding="utf-8") as  f:
                        f.write("{\n")
                        f.write('  "datasetType":"MS",\n')
                        f.write('  "numRows":%s,\n' % provlist[0])
                        f.write('  "columns":{\n')
                        t = 1
                        for name in columns:
                            f.write('    "%s":{\n' % name)
                            f.write('      "type":"int64",\n')
                            f.write('      "rank":1,\n')
                            f.write('      "shape":[%s]\n' % provlist[t])
                            f.write('    }')
                            if t < len(columns):
                                f.write(',')
                            f.write('\n')
                            t += 1
                        f.write('  }\n}\n')
                        print(" | Write to " + self.schema_address)


class TextDataLoader(DataLoader):
    """Loader for text data."""

    def __init__(self,
                 src_filepath: str,
                 tokenizer: Tokenizer,
                 min_sen_len=0,
                 source_max_sen_len=None,
                 schema_address=None):
        super(TextDataLoader, self).__init__()
        self._src_filepath = src_filepath
        self.tokenizer = tokenizer
        self.min_sen_len = min_sen_len
        self.source_max_sen_len = source_max_sen_len
        self.schema_address = schema_address

    def _load(self):
        count = 0
        if self.source_max_sen_len is None:
            with open(self._src_filepath, "r") as _src_file:
                print(f" | count the max_sen_len of corpus {self._src_filepath}.")
                max_src = 0
                for _, _pair in enumerate(_src_file):
                    src_tokens = self.tokenizer.tokenize(_pair)
                    src_len = len(src_tokens)
                    if src_len > max_src:
                        max_src = src_len
                self.source_max_sen_len = max_src

        with open(self._src_filepath, "r") as _src_file:
            print(f" | Processing corpus {self._src_filepath}.")
            for _, _pair in enumerate(_src_file):
                src_tokens = self.tokenizer.tokenize(_pair)
                src_len = len(src_tokens)
                src_tokens = np.array(src_tokens)
                # encoder inputs
                encoder_input = self.padding(src_tokens, self.tokenizer.padding_index, self.source_max_sen_len)
                src_padding = np.zeros(shape=self.source_max_sen_len, dtype=np.int64)
                for i in range(src_len):
                    src_padding[i] = 1

                example = {
                    "src": encoder_input,
                    "src_padding": src_padding
                }
                self._add_example(example)
                count += 1

            print(f" | source padding_len = {self.source_max_sen_len}.")
            print(f" | Total  activate  sen = {count}.")
            print(f" | Total  sen = {count}.")

            if self.schema_address is not None:
                provlist = [count, self.source_max_sen_len, self.source_max_sen_len]
                columns = ["src", "src_padding"]
                with open(self.schema_address, "w", encoding="utf-8") as  f:
                    f.write("{\n")
                    f.write('  "datasetType":"MS",\n')
                    f.write('  "numRows":%s,\n' % provlist[0])
                    f.write('  "columns":{\n')
                    t = 1
                    for name in columns:
                        f.write('    "%s":{\n' % name)
                        f.write('      "type":"int64",\n')
                        f.write('      "rank":1,\n')
                        f.write('      "shape":[%s]\n' % provlist[t])
                        f.write('    }')
                        if t < len(columns):
                            f.write(',')
                        f.write('\n')
                        t += 1
                    f.write('  }\n}\n')
                    print(" | Write to " + self.schema_address)
