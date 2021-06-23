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
:py:class:`Tokenizer`
"""
from src.common.register import RegisterSet
from src.data.vocabulary import Vocabulary


@RegisterSet.tokenizer.register
class Tokenizer():
    """Tokenizer"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file:
        :param split_char:
        :param unk_token:
        :param params:
        """
        self.vocabulary = Vocabulary(vocab_file, unk_token)
        self.split_char = split_char
        self.unk_token = unk_token
        self.params = params
        self.vocab_size = self.vocabulary.vocab_size

    def tokenize(self, text):
        """
        :param text:
        :return: tokens
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        raise NotImplementedError

    def covert_id_to_token(self, tid):
        """
        :param tid:
        :return: token
        """
        return self.vocabulary.covert_id_to_token(tid)

    def covert_token_to_id(self, token):
        """
        :param token:
        :return: id
        """
        return self.vocabulary.covert_token_to_id(token)
