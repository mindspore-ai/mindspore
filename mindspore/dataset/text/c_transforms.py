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
"""
c transforms for all text related operators
"""

import mindspore._c_dataengine as cde
from .validators import check_lookup, check_from_list, check_from_dict, check_from_file


class Vocab(cde.Vocab):
    """
        Vocab object that is used for lookup word
    Args:
    """

    def __init__(self):
        pass

    @classmethod
    @check_from_list
    def from_list(cls, word_list):
        """
           build a vocab object from a list of word
        Args:
            word_list(list): a list of string where each element is a word
        """
        return super().from_list(word_list)

    @classmethod
    @check_from_file
    def from_file(cls, file_path, delimiter=None, vocab_size=None):
        """
            build a vocab object from a list of word
        Args:
            file_path(str): path to the file which contains the vocab list
            delimiter(None, str): a delimiter to break up each line in file, the first element is taken to be the word
            vocab_size(None, int): number of words to read from file_path
        """
        return super().from_file(file_path, delimiter, vocab_size)

    @classmethod
    @check_from_dict
    def from_dict(cls, word_dict):
        """
            build a vocab object from a dict.
        Args:
            word_dict(dict): dict contains word, id pairs. id should start from 2 and continuous
        """
        return super().from_dict(word_dict)


class Lookup(cde.LookupOp):
    """
        Lookup operator that looks up a word to an id
    Args:
        vocab(Vocab): a Vocab object
        unknown(None,int): default id to lookup a word that is out of vocab
    """

    @check_lookup
    def __init__(self, vocab, unknown=None):
        if unknown is None:
            super().__init__(vocab)
        else:
            super().__init__(vocab, unknown)
