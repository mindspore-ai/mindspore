# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Tokenization classes for CPM."""

from functools import lru_cache
from io import open
import logging
import json
import jieba
import sentencepiece as spm

jieba.setLogLevel(logging.INFO)


class CPMTokenizer():
    """Tokenizer for CPM network."""

    def __init__(self, vocab_json, chinese_vocab_file):
        self.tok2idx = json.load(open(vocab_json))
        self.idx2tok = {token_idx: token_key for token_key, token_idx in self.tok2idx.items()}
        self.spp = spm.SentencePieceProcessor(model_file=chinese_vocab_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.eod_id = self.tok2idx['<eod>']
        self.unk_id = self.tok2idx['<unk>']
        self.mask_id = self.tok2idx['<mask>']
        self.pad_id = self.tok2idx['<pad>']

    @property
    def vocab_size(self):
        return len(self.tok2idx)

    def __len__(self):
        return len(self.tok2idx)

    @property
    def unk(self):
        return self.unk_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def eod(self):
        return self.eod_id

    @lru_cache()
    def tokenize_op(self, input_text):
        input_tokens = [x_item.translate(self.translator) for x_item in jieba.cut(input_text, cut_all=False)]
        return self.spp.encode(" ".join(input_tokens))

    def encode(self, input_text):
        result_encode = self.tokenize_op(input_text)
        return result_encode

    def decode(self, input_tokens):
        text_decoder = self.spp.decode(input_tokens)
        text_decoder = text_decoder.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text_decoder
