# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
validators for text ops
"""

from functools import wraps

import mindspore._c_dataengine as cde

from ..transforms.validators import check_uint32


def check_lookup(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        vocab, unknown = (list(args) + 2 * [None])[:2]
        if "vocab" in kwargs:
            vocab = kwargs.get("vocab")
        if "unknown" in kwargs:
            unknown = kwargs.get("unknown")
        if unknown is not None:
            if not (isinstance(unknown, int) and unknown >= 0):
                raise ValueError("unknown needs to be a non-negative integer")

        if not isinstance(vocab, cde.Vocab):
            raise ValueError("vocab is not an instance of cde.Vocab")

        kwargs["vocab"] = vocab
        kwargs["unknown"] = unknown
        return method(self, **kwargs)

    return new_method


def check_from_file(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        file_path, delimiter, vocab_size = (list(args) + 3 * [None])[:3]
        if "file_path" in kwargs:
            file_path = kwargs.get("file_path")
        if "delimiter" in kwargs:
            delimiter = kwargs.get("delimiter")
        if "vocab_size" in kwargs:
            vocab_size = kwargs.get("vocab_size")

        if not isinstance(file_path, str):
            raise ValueError("file_path needs to be str")

        if delimiter is not None:
            if not isinstance(delimiter, str):
                raise ValueError("delimiter needs to be str")
        else:
            delimiter = ""
        if vocab_size is not None:
            if not (isinstance(vocab_size, int) and vocab_size > 0):
                raise ValueError("vocab size needs to be a positive integer")
        else:
            vocab_size = -1
        kwargs["file_path"] = file_path
        kwargs["delimiter"] = delimiter
        kwargs["vocab_size"] = vocab_size
        return method(self, **kwargs)

    return new_method


def check_from_list(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word_list, = (list(args) + [None])[:1]
        if "word_list" in kwargs:
            word_list = kwargs.get("word_list")
        if not isinstance(word_list, list):
            raise ValueError("word_list needs to be a list of words")
        for word in word_list:
            if not isinstance(word, str):
                raise ValueError("each word in word list needs to be type str")

        kwargs["word_list"] = word_list
        return method(self, **kwargs)

    return new_method


def check_from_dict(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word_dict, = (list(args) + [None])[:1]
        if "word_dict" in kwargs:
            word_dict = kwargs.get("word_dict")
        if not isinstance(word_dict, dict):
            raise ValueError("word_dict needs to be a list of word,id pairs")
        for word, word_id in word_dict.items():
            if not isinstance(word, str):
                raise ValueError("each word in word_dict needs to be type str")
            if not (isinstance(word_id, int) and word_id >= 0):
                raise ValueError("each word id needs to be positive integer")
        kwargs["word_dict"] = word_dict
        return method(self, **kwargs)

    return new_method


def check_jieba_init(method):
    """Wrapper method to check the parameters of jieba add word."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        hmm_path, mp_path, model = (list(args) + 3 * [None])[:3]

        if "hmm_path" in kwargs:
            hmm_path = kwargs.get("hmm_path")
        if "mp_path" in kwargs:
            mp_path = kwargs.get("mp_path")
        if hmm_path is None:
            raise ValueError(
                "the dict of HMMSegment in cppjieba is not provided")
        kwargs["hmm_path"] = hmm_path
        if mp_path is None:
            raise ValueError(
                "the dict of MPSegment in cppjieba is not provided")
        kwargs["mp_path"] = mp_path
        if model is not None:
            kwargs["model"] = model
        return method(self, **kwargs)

    return new_method


def check_jieba_add_word(method):
    """Wrapper method to check the parameters of jieba add word."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word, freq = (list(args) + 2 * [None])[:2]

        if "word" in kwargs:
            word = kwargs.get("word")
        if "freq" in kwargs:
            freq = kwargs.get("freq")
        if word is None:
            raise ValueError("word is not provided")
        kwargs["word"] = word
        if freq is not None:
            check_uint32(freq)
            kwargs["freq"] = freq
        return method(self, **kwargs)

    return new_method


def check_jieba_add_dict(method):
    """Wrapper method to check the parameters of add dict"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        user_dict = (list(args) + [None])[0]
        if "user_dict" in kwargs:
            user_dict = kwargs.get("user_dict")
        if user_dict is None:
            raise ValueError("user_dict is not provided")
        kwargs["user_dict"] = user_dict
        return method(self, **kwargs)

    return new_method


def check_ngram(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        n, left_pad, right_pad, separator = (list(args) + 4 * [None])[:4]
        if "n" in kwargs:
            n = kwargs.get("n")
        if "left_pad" in kwargs:
            left_pad = kwargs.get("left_pad")
        if "right_pad" in kwargs:
            right_pad = kwargs.get("right_pad")
        if "separator" in kwargs:
            separator = kwargs.get("separator")

        if isinstance(n, int):
            n = [n]

        if not (isinstance(n, list) and n != []):
            raise ValueError("n needs to be a non-empty list of positive integers")

        for gram in n:
            if not (isinstance(gram, int) and gram > 0):
                raise ValueError("n in ngram needs to be a positive number\n")

        if left_pad is None:
            left_pad = ("", 0)

        if right_pad is None:
            right_pad = ("", 0)

        if not (isinstance(left_pad, tuple) and len(left_pad) == 2 and isinstance(left_pad[0], str) and isinstance(
                left_pad[1], int)):
            raise ValueError("left_pad needs to be a tuple of (str, int) str is pad token and int is pad_width")

        if not (isinstance(right_pad, tuple) and len(right_pad) == 2 and isinstance(right_pad[0], str) and isinstance(
                right_pad[1], int)):
            raise ValueError("right_pad needs to be a tuple of (str, int) str is pad token and int is pad_width")

        if not (left_pad[1] >= 0 and right_pad[1] >= 0):
            raise ValueError("padding width need to be positive numbers")

        if separator is None:
            separator = " "

        if not isinstance(separator, str):
            raise ValueError("separator needs to be a string")

        kwargs["n"] = n
        kwargs["left_pad"] = left_pad
        kwargs["right_pad"] = right_pad
        kwargs["separator"] = separator

        return method(self, **kwargs)

    return new_method
