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
Some basic function for nlp
"""
from enum import IntEnum

import mindspore._c_dataengine as cde
import numpy as np

from .validators import check_from_file, check_from_list, check_from_dict


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


def to_str(array, encoding='utf8'):
    """
    Convert numpy array of `bytes` to array of `str` by decoding each element based on charset `encoding`.

    Args:
        array (numpy array): Array of type `bytes` representing strings.
        encoding (string): Indicating the charset for decoding.
    Returns:
        Numpy array of `str`.

    """

    if not isinstance(array, np.ndarray):
        raise ValueError('input should be a numpy array')

    return np.char.decode(array, encoding)


def to_bytes(array, encoding='utf8'):
    """
    Convert numpy array of `str` to array of `bytes` by encoding each element based on charset `encoding`.

    Args:
        array (numpy array): Array of type `str` representing strings.
        encoding (string): Indicating the charset for encoding.
    Returns:
        Numpy array of `bytes`.

    """

    if not isinstance(array, np.ndarray):
        raise ValueError('input should be a numpy array')

    return np.char.encode(array, encoding)


class JiebaMode(IntEnum):
    MIX = 0
    MP = 1
    HMM = 2
