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

import copy
import numpy as np
import mindspore._c_dataengine as cde

from .validators import check_from_file, check_from_list, check_from_dict, check_from_dataset


class Vocab(cde.Vocab):
    """
        Vocab object that is used for lookup word
    """

    @classmethod
    @check_from_dataset
    def from_dataset(cls, dataset, columns=None, freq_range=None, top_k=None):
        """
        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified)
        Args:
            dataset(Dataset): dataset to build vocab from.
            columns(str or list, optional): column names to get words from. It can be a list of column names.
                (Default is None where all columns will be used. If any column isn't string type, will return error)
            freq_range(tuple, optional): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                can be None, which corresponds to 0/total_words separately (default is None, all words are included)
            top_k(int, optional): top_k > 0. Number of words to be built into vocab. top_k most frequent words are
                taken. top_k is taken after freq_range. If not enough top_k, all words will be taken. (default is None
                all words are included)
        return:
            text.Vocab: vocab object built from dataset.
        """
        vocab = Vocab()
        root = copy.deepcopy(dataset).build_vocab(vocab, columns, freq_range, top_k)
        for d in root.create_dict_iterator():
            if d is not None:
                raise ValueError("from_dataset should receive data other than None")
        return vocab

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
