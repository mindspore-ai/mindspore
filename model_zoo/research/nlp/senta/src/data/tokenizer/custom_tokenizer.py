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
"""
:py:class:`CustomTokenizer`
"""
from src.common.register import RegisterSet
from src.data.tokenizer.tokenizer import Tokenizer
from src.utils.util_helper import convert_to_unicode


@RegisterSet.tokenizer.register
class CustomTokenizer(Tokenizer):
    """CustomTokenizer:"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file:
        :param split_char:
        """
        Tokenizer.__init__(self, vocab_file, split_char, unk_token, params)
        self.split_char = split_char

    def tokenize(self, text):
        """
        :param text:
        :return:
        """
        text = convert_to_unicode(text)
        split_tokens = text.split(self.split_char)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        return self.vocabulary.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        return self.vocabulary.convert_ids_to_tokens(ids)
