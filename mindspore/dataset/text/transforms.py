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

import os
import re

import mindspore._c_dataengine as cde

from .utils import JiebaMode
from .validators import check_lookup, check_jieba_add_dict, \
    check_jieba_add_word, check_jieba_init, check_ngram


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


class Ngram(cde.NgramOp):
    """
    TensorOp to generate n-gram from a 1-D string Tensor
    Refer to https://en.wikipedia.org/wiki/N-gram#Examples for an explanation of what n-gram is.
    Args:
        n(int or list):  n in n-gram, n >= 1. n is a list of positive integers, for e.g. n=[4,3], The result
        would be a 4-gram followed by a 3-gram in the same tensor.
        left_pad(tuple, optional): ("pad_token",pad_width). Padding performed on left side of the sequence. pad_width
        will be capped at n-1. left_pad=("_",2) would pad left side of the sequence with "__". (Default is None)
        right_pad(tuple, optional): ("pad_token",pad_width). Padding performed on right side of the sequence. pad_width
        will be capped at n-1. right_pad=("-":2) would pad right side of the sequence with "--". (Default is None)
        separator(str,optional): symbol used to join strings together. for e.g. if 2-gram the ["mindspore", "amazing"]
        with separator="-" the result would be ["mindspore-amazing"]. (Default is None which means whitespace is used)
    """

    @check_ngram
    def __init__(self, n, left_pad=None, right_pad=None, separator=None):
        super().__init__(ngrams=n, l_pad_len=left_pad[1], r_pad_len=right_pad[1], l_pad_token=left_pad[0],
                         r_pad_token=right_pad[0], separator=separator)


DE_C_INTER_JIEBA_MODE = {
    JiebaMode.MIX: cde.JiebaMode.DE_JIEBA_MIX,
    JiebaMode.MP: cde.JiebaMode.DE_JIEBA_MP,
    JiebaMode.HMM: cde.JiebaMode.DE_JIEBA_HMM
}


class JiebaTokenizer(cde.JiebaTokenizerOp):
    """
    Tokenize Chinese string into words based on dictionary.

    Args:
        hmm_path (str): the dictionary file is used by  HMMSegment algorithm,
            the dictionary can be obtained on the official website of cppjieba.
        mp_path(str): the dictionary file is used by MPSegment algorithm,
            the dictionary can be obtained on the official website of cppjieba.
        mode (Enum):  [Default "MIX"], "MP" model will tokenize with MPSegment algorithm,
            "HMM" mode will tokenize with Hiddel Markov Model Segment algorithm,
            "MIX" model will tokenize with a mix of MPSegment and HMMSegment algorithm.
    """

    @check_jieba_init
    def __init__(self, hmm_path, mp_path, mode=JiebaMode.MIX):
        self.mode = mode
        self.__check_path__(hmm_path)
        self.__check_path__(mp_path)
        super().__init__(hmm_path, mp_path,
                         DE_C_INTER_JIEBA_MODE[mode])

    @check_jieba_add_word
    def add_word(self, word, freq=None):
        """
        Add user defined word to JiebaTokenizer's dictionary
        Args:
            word(required, string): The word to be added to the JiebaTokenizer instance.
                The added word will not be written into the built-in dictionary on disk.
            freq(optional, int): The frequency of the word to be added, The higher the frequency,
                the better change the word will be tokenized(default None, use default frequency).
        """
        if freq is None:
            super().add_word(word, 0)
        else:
            super().add_word(word, freq)

    @check_jieba_add_dict
    def add_dict(self, user_dict):
        """
        Add user defined word to JiebaTokenizer's dictionary
        Args:
            user_dict(path/dict):Dictionary to be added, file path or Python dictionary,
            Python Dict format: {word1:freq1, word2:freq2,...}
            Jieba dictionary format : word(required), freq(optional), such as:
                word1 freq1
                word2
                word3 freq3
        """
        if isinstance(user_dict, str):
            self.__add_dict_py_file(user_dict)
        elif isinstance(user_dict, dict):
            for k, v in user_dict.items():
                self.add_word(k, v)
        else:
            raise ValueError("the type of user_dict must str or dict")

    def __add_dict_py_file(self, file_path):
        """Add user defined word by file"""
        words_list = self.__parser_file(file_path)
        for data in words_list:
            if data[1] is None:
                freq = 0
            else:
                freq = int(data[1])
            self.add_word(data[0], freq)

    def __parser_file(self, file_path):
        """parser user defined word by file"""
        if not os.path.exists(file_path):
            raise ValueError(
                "user dict file {} is not exist".format(file_path))
        file_dict = open(file_path)
        data_re = re.compile('^(.+?)( [0-9]+)?$', re.U)
        words_list = []
        for item in file_dict:
            data = item.strip()
            if not isinstance(data, str):
                data = self.__decode(data)
            words = data_re.match(data).groups()
            if len(words) != 2:
                raise ValueError(
                    "user dict file {} format error".format(file_path))
            words_list.append(words)
        return words_list

    def __decode(self, data):
        """decode the dict file to utf8"""
        try:
            data = data.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("user dict file must utf8")
        return data.lstrip('\ufeff')

    def __check_path__(self, model_path):
        """check model path"""
        if not os.path.exists(model_path):
            raise ValueError(
                " jieba mode file {} is not exist".format(model_path))


class UnicodeCharTokenizer(cde.UnicodeCharTokenizerOp):
    """
    Tokenize a scalar tensor of UTF-8 string to Unicode characters.
    """
