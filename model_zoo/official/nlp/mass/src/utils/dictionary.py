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
"""Vocabulary."""
from typing import List
import numpy as np

CUBE_SIZE = 16
REPLACE_THRESHOLD = 200


class Dictionary:
    """Dictionary for mono lingual dataset."""

    def __init__(self, max_size=46000, bos="<s>", eos="</s>", unk="<unk>",
                 mask="<mask>", padding="<pad>"):
        self._bos = bos
        self._eos = eos
        self._unk = unk
        self._mask = mask
        self._padding = padding
        self._symbols = []
        self._frequency = []
        self._mapping = {}
        self._init_symbols()
        self.is_learning = False
        self.max_vocab_size = max_size

    def shrink(self, threshold=50):
        """
        Shrink dataset into a small one.

        Args:
            threshold (int): Threshold that determinate whether to
                drop the word.

        Returns:
            Dictionary, a new dict.
        """
        _new_dict = Dictionary()

        freq_idx = [(f, i) for i, f in enumerate(self._frequency)]
        freq_idx = sorted(freq_idx, key=lambda x: x[0], reverse=True)

        freqs = np.array(self._frequency, dtype=np.int)
        filtered_count = np.where(freqs <= threshold)[0].shape[0]

        left_count = self.size - filtered_count
        if left_count % CUBE_SIZE != 0:
            supplement = CUBE_SIZE - left_count % CUBE_SIZE
            if supplement <= filtered_count:
                filtered_count -= supplement

        for f, i in freq_idx:
            if f <= threshold and filtered_count > 0:
                filtered_count -= 1
                continue
            _new_dict.add_symbol(self._symbols[i], f)

        return _new_dict

    def set_to_learn(self, learn: bool):
        self.is_learning = learn

    def is_empty(self):
        if self.size <= 4:
            if sum(self._frequency) == 0:
                return True
        return False

    @property
    def symbols(self):
        return self._symbols

    @property
    def frequency(self):
        return self._frequency

    @property
    def size(self):
        return len(self._symbols)

    @property
    def mask(self):
        return self._mask

    @property
    def eos(self):
        return self._eos

    @property
    def bos(self):
        return self._bos

    @property
    def unk(self):
        return self._unk

    @property
    def padding(self):
        return self._padding

    @property
    def padding_index(self):
        return self._padding_index

    @property
    def mask_index(self):
        return self._mask_index

    @property
    def eos_index(self):
        return self._eos_index

    @property
    def bos_index(self):
        return self._bos_index

    @property
    def unk_index(self):
        return self._unk_index

    def _init_symbols(self):
        self._padding_index = self.add_symbol(self._padding, 0)  # 0
        self._bos_index = self.add_symbol(self._bos, 0)  # 1
        self._eos_index = self.add_symbol(self._eos, 0)  # 2
        self._unk_index = self.add_symbol(self._unk, 0)  # 3
        self._mask_index = self.add_symbol(self._mask, 0)  # 4

    def __contains__(self, symbol):
        return symbol in self._mapping

    def __getitem__(self, idx):
        if 0 <= idx < self.size:
            return self._symbols[idx]
        return self._unk

    def __len__(self):
        return self.size

    def index(self, symbol: str):
        """
        Return id according to symbol.

        Args:
            symbol (str): Symbol.

        Returns:
            int, id.
        """
        idx = self._mapping.get(symbol)
        if idx is None:
            if self.is_learning and symbol.isalpha():
                if self.max_vocab_size <= self.size:
                    return self.add_symbol(symbol)

                if symbol.lower() in self._mapping:
                    return self._mapping.get(symbol.lower())

            idx = self._mapping.get(symbol.lower())
            if idx is not None:
                freq = self._frequency[idx]
                # If lower symbol in vocabulary and
                # its frequency larger than `REPLACE_THRESHOLD`,
                # then replace symbol by lower symbol.
                if freq >= REPLACE_THRESHOLD:
                    return idx
            return self.unk_index
        return idx

    def add_symbol(self, symbol, times=1):
        """
        Add symbol to dict.

        Args:
            symbol (str): Symbol.
            times (int): Frequency.

        Returns:
            int, token id.
        """
        if symbol in self._mapping:
            idx = self._mapping[symbol]
            self._frequency[idx] = self._frequency[idx] + times
            return idx

        idx = len(self._symbols)
        self._mapping[symbol] = idx
        self._symbols.append(symbol)
        self._frequency.append(times)
        return idx

    @classmethod
    def load_from_text(cls, filepaths: List[str]):
        """
        Load dict from text which is in format of [word, freq].

        Args:
            filepaths (str): Dict list.

        Returns:
            Dictionary, dict instance.
        """
        _dict = cls()
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for _, line in enumerate(f):
                    line = line.strip()
                    if line is None:
                        continue
                    try:
                        word, freq = line.split(" ")
                        _dict.add_symbol(word, times=int(freq))
                    except ValueError:
                        continue

        return _dict

    @classmethod
    def load_from_persisted_dict(cls, filepath):
        """
        Load dict from binary file.

        Args:
            filepath (str): File path.

        Returns:
            Dictionary, dict instance.
        """
        import pickle
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def persistence(self, path):
        """Save dict to binary file."""
        import pickle
        with open(path, "wb") as _dict:
            pickle.dump(self, _dict, protocol=1)

    def merge_dict(self, other, new_dict=False):
        """Merge two dict."""
        if other.is_empty():
            return self

        if new_dict:
            _dict = Dictionary()

            for s, f in zip(self.symbols, self.frequency):
                _dict.add_symbol(s, times=f)
            for s, f in zip(other.symbols, other.frequency):
                _dict.add_symbol(s, times=f)
            return _dict

        for s, f in zip(other.symbols, other.frequency):
            self.add_symbol(s, times=f)

        return self

    def export(self, path):
        """Save text-like vocabulary."""
        _lines = []
        for token, freq in zip(self._symbols, self._frequency):
            _lines.append(f"{token} {freq}")
        with open(path, "w") as f:
            f.writelines(_lines)
