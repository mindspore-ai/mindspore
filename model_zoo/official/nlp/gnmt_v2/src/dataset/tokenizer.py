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
"""Tokenizer."""
import os
from collections import defaultdict
from functools import partial
import subword_nmt.apply_bpe
import sacremoses


class Tokenizer:
    """
    Tokenizer class.
    """

    def __init__(self, vocab_address=None, bpe_code_address=None,
                 src_en='en', tgt_de='de', vocab_pad=8, isolator='@@'):
        """
        Constructor for the Tokenizer class.

        Args:
            vocab_address: vocabulary address.
            bpe_code_address: path to the file with bpe codes.
            vocab_pad: pads vocabulary to a multiple of 'vocab_pad' tokens.
            isolator: tokenization isolator.
        """
        self.padding_index = 0
        self.unk_index = 1
        self.bos_index = 2
        self.eos_index = 3
        self.pad_word = '<pad>'
        self.unk_word = '<unk>'
        self.bos_word = '<s>'
        self.eos_word = r'<\s>'
        self.isolator = isolator
        self.init_bpe(bpe_code_address)
        self.vocab_establist(vocab_address, vocab_pad)
        self.sacremoses_tokenizer = sacremoses.MosesTokenizer(src_en)
        self.sacremoses_detokenizer = sacremoses.MosesDetokenizer(tgt_de)

    def init_bpe(self, bpe_code_address):
        """Init bpe."""
        if (bpe_code_address is not None) and os.path.exists(bpe_code_address):
            with open(bpe_code_address, 'r') as f1:
                self.bpe = subword_nmt.apply_bpe.BPE(f1)

    def vocab_establist(self, vocab_address, vocab_pad):
        """Establish vocabulary."""
        if (vocab_address is None) or (not os.path.exists(vocab_address)):
            return
        vocab_words = [self.pad_word, self.unk_word, self.bos_word, self.eos_word]
        with open(vocab_address) as f1:
            for sentence in f1:
                vocab_words.append(sentence.strip())
        vocab_size = len(vocab_words)
        padded_vocab_size = (vocab_size + vocab_pad - 1) // vocab_pad * vocab_pad
        for idx in range(0, padded_vocab_size - vocab_size):
            fil_token = f'filled{idx:04d}'
            vocab_words.append(fil_token)
        self.vocab_size = len(vocab_words)
        self.tok2idx = defaultdict(partial(int, self.unk_index))
        for idx, token in enumerate(vocab_words):
            self.tok2idx[token] = idx
        self.idx2tok = {}
        self.idx2tok = defaultdict(partial(str, ","))
        for token, idx in self.tok2idx.items():
            self.idx2tok[idx] = token

    def tokenize(self, sentence):
        """Tokenize sentence."""
        tokenized = self.sacremoses_tokenizer.tokenize(sentence, return_str=True)
        bpe = self.bpe.process_line(tokenized)
        sentence = bpe.strip().split()
        inputs = [self.tok2idx[i] for i in sentence]
        inputs = [self.bos_index] + inputs + [self.eos_index]
        return inputs

    def detokenize(self, indexes, gap=' '):
        """Detokenizes single sentence and removes token isolator characters."""
        reconstruction_bpe = gap.join([self.idx2tok[idx] for idx in indexes])
        reconstruction_bpe = reconstruction_bpe.replace(self.isolator + ' ', '')
        reconstruction_bpe = reconstruction_bpe.replace(self.isolator, '')
        reconstruction_bpe = reconstruction_bpe.replace(self.bos_word, '')
        reconstruction_bpe = reconstruction_bpe.replace(self.eos_word, '')
        reconstruction_bpe = reconstruction_bpe.replace(self.unk_word, '')
        reconstruction_bpe = reconstruction_bpe.replace(self.pad_word, '')
        reconstruction_bpe = reconstruction_bpe.strip()
        reconstruction_words = self.sacremoses_detokenizer.detokenize(reconstruction_bpe.split())
        return reconstruction_words
