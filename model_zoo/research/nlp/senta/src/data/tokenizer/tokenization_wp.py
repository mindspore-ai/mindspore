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
""" Configuration base class and utilities."""
from src.common.register import RegisterSet
from src.data.tokenizer.tokenizer import Tokenizer

from src.data.tokenizer.tokenization_utils import BpeEncoder, convert_by_vocab


@RegisterSet.tokenizer.register
class GptBpeTokenizer(Tokenizer):
    """Gpt bpe tokenizer
    """

    def __init__(self, vocab_file, split_char=" ",
                 unk_token="[UNK]", params=None):
        Tokenizer.__init__(
            self,
            vocab_file,
            split_char,
            unk_token,
            params=params)

        assert params.get(
            "bpe_json_file", False), "params must have encoder_json_file"
        assert params.get(
            "bpe_vocab_file", False), "params must have vocab_bpe_file"
        vocab_bpe_file = params.get("bpe_vocab_file", False)
        encoder_json_file = params.get("bpe_json_file", False)

        self.gptbpe_tokenizer = BpeEncoder(encoder_json_file, vocab_bpe_file)

    def tokenize(self, text):
        """
        tokenize
        """
        return [str(token) for token in self.gptbpe_tokenizer.encode(text)]

    def convert_tokens_to_ids(self, tokens):
        """
        convert_tokens_to_ids
        """
        return convert_by_vocab(self.vocabulary.vocab_dict, tokens)

    def convert_ids_to_tokens(self, ids):
        """
        convert_ids_to_tokens
        """
        return convert_by_vocab(self.vocabulary.id_dict, ids)
