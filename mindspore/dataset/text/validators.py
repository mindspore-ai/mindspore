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
import mindspore.common.dtype as mstype

from mindspore._c_expression import typing
from ..transforms.validators import check_uint32, check_pos_int64


def check_unique_list_of_words(words, arg_name):
    """Check that words is a list and each element is a str without any duplication"""

    if not isinstance(words, list):
        raise ValueError(arg_name + " needs to be a list of words of type string.")
    words_set = set()
    for word in words:
        if not isinstance(word, str):
            raise ValueError("each word in " + arg_name + " needs to be type str.")
        if word in words_set:
            raise ValueError(arg_name + " contains duplicate word: " + word + ".")
        words_set.add(word)
    return words_set


def check_lookup(method):
    """A wrapper that wrap a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        vocab, unknown = (list(args) + 2 * [None])[:2]
        if "vocab" in kwargs:
            vocab = kwargs.get("vocab")
        if "unknown" in kwargs:
            unknown = kwargs.get("unknown")
        if unknown is not None:
            if not (isinstance(unknown, int) and unknown >= 0):
                raise ValueError("unknown needs to be a non-negative integer.")

        if not isinstance(vocab, cde.Vocab):
            raise ValueError("vocab is not an instance of cde.Vocab.")

        kwargs["vocab"] = vocab
        kwargs["unknown"] = unknown
        return method(self, **kwargs)

    return new_method


def check_from_file(method):
    """A wrapper that wrap a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        file_path, delimiter, vocab_size, special_tokens, special_first = (list(args) + 5 * [None])[:5]
        if "file_path" in kwargs:
            file_path = kwargs.get("file_path")
        if "delimiter" in kwargs:
            delimiter = kwargs.get("delimiter")
        if "vocab_size" in kwargs:
            vocab_size = kwargs.get("vocab_size")
        if "special_tokens" in kwargs:
            special_tokens = kwargs.get("special_tokens")
        if "special_first" in kwargs:
            special_first = kwargs.get("special_first")

        if not isinstance(file_path, str):
            raise ValueError("file_path needs to be str.")

        if delimiter is not None:
            if not isinstance(delimiter, str):
                raise ValueError("delimiter needs to be str.")
        else:
            delimiter = ""
        if vocab_size is not None:
            if not (isinstance(vocab_size, int) and vocab_size > 0):
                raise ValueError("vocab size needs to be a positive integer.")
        else:
            vocab_size = -1

        if special_first is None:
            special_first = True

        if not isinstance(special_first, bool):
            raise ValueError("special_first needs to be a boolean value")

        if special_tokens is None:
            special_tokens = []

        check_unique_list_of_words(special_tokens, "special_tokens")

        kwargs["file_path"] = file_path
        kwargs["delimiter"] = delimiter
        kwargs["vocab_size"] = vocab_size
        kwargs["special_tokens"] = special_tokens
        kwargs["special_first"] = special_first

        return method(self, **kwargs)

    return new_method


def check_from_list(method):
    """A wrapper that wrap a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word_list, special_tokens, special_first = (list(args) + 3 * [None])[:3]
        if "word_list" in kwargs:
            word_list = kwargs.get("word_list")
        if "special_tokens" in kwargs:
            special_tokens = kwargs.get("special_tokens")
        if "special_first" in kwargs:
            special_first = kwargs.get("special_first")
        if special_tokens is None:
            special_tokens = []
        word_set = check_unique_list_of_words(word_list, "word_list")
        token_set = check_unique_list_of_words(special_tokens, "special_tokens")

        intersect = word_set.intersection(token_set)

        if intersect != set():
            raise ValueError("special_tokens and word_list contain duplicate word :" + str(intersect) + ".")

        if special_first is None:
            special_first = True

        if not isinstance(special_first, bool):
            raise ValueError("special_first needs to be a boolean value.")

        kwargs["word_list"] = word_list
        kwargs["special_tokens"] = special_tokens
        kwargs["special_first"] = special_first
        return method(self, **kwargs)

    return new_method


def check_from_dict(method):
    """A wrapper that wrap a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word_dict, = (list(args) + [None])[:1]
        if "word_dict" in kwargs:
            word_dict = kwargs.get("word_dict")
        if not isinstance(word_dict, dict):
            raise ValueError("word_dict needs to be a list of word,id pairs.")
        for word, word_id in word_dict.items():
            if not isinstance(word, str):
                raise ValueError("Each word in word_dict needs to be type string.")
            if not (isinstance(word_id, int) and word_id >= 0):
                raise ValueError("Each word id needs to be positive integer.")
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
                "The dict of HMMSegment in cppjieba is not provided.")
        kwargs["hmm_path"] = hmm_path
        if mp_path is None:
            raise ValueError(
                "The dict of MPSegment in cppjieba is not provided.")
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
            raise ValueError("word is not provided.")
        kwargs["word"] = word
        if freq is not None:
            check_uint32(freq)
            kwargs["freq"] = freq
        return method(self, **kwargs)

    return new_method


def check_jieba_add_dict(method):
    """Wrapper method to check the parameters of add dict."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        user_dict = (list(args) + [None])[0]
        if "user_dict" in kwargs:
            user_dict = kwargs.get("user_dict")
        if user_dict is None:
            raise ValueError("user_dict is not provided.")
        kwargs["user_dict"] = user_dict
        return method(self, **kwargs)

    return new_method


def check_from_dataset(method):
    """A wrapper that wrap a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        dataset, columns, freq_range, top_k, special_tokens, special_first = (list(args) + 6 * [None])[:6]
        if "dataset" in kwargs:
            dataset = kwargs.get("dataset")
        if "columns" in kwargs:
            columns = kwargs.get("columns")
        if "freq_range" in kwargs:
            freq_range = kwargs.get("freq_range")
        if "top_k" in kwargs:
            top_k = kwargs.get("top_k")
        if "special_tokens" in kwargs:
            special_tokens = kwargs.get("special_tokens")
        if "special_first" in kwargs:
            special_first = kwargs.get("special_first")

        if columns is None:
            columns = []

        if not isinstance(columns, list):
            columns = [columns]

        for column in columns:
            if not isinstance(column, str):
                raise ValueError("columns need to be a list of strings.")

        if freq_range is None:
            freq_range = (None, None)

        if not isinstance(freq_range, tuple) or len(freq_range) != 2:
            raise ValueError("freq_range needs to be either None or a tuple of 2 integers or an int and a None.")

        for num in freq_range:
            if num is not None and (not isinstance(num, int)):
                raise ValueError("freq_range needs to be either None or a tuple of 2 integers or an int and a None.")

        if isinstance(freq_range[0], int) and isinstance(freq_range[1], int):
            if freq_range[0] > freq_range[1] or freq_range[0] < 0:
                raise ValueError("frequency range [a,b] should be 0 <= a <= b (a,b are inclusive).")

        if top_k is not None and (not isinstance(top_k, int)):
            raise ValueError("top_k needs to be a positive integer.")

        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError("top_k needs to be a positive integer.")

        if special_first is None:
            special_first = True

        if special_tokens is None:
            special_tokens = []

        if not isinstance(special_first, bool):
            raise ValueError("special_first needs to be a boolean value.")

        check_unique_list_of_words(special_tokens, "special_tokens")

        kwargs["dataset"] = dataset
        kwargs["columns"] = columns
        kwargs["freq_range"] = freq_range
        kwargs["top_k"] = top_k
        kwargs["special_tokens"] = special_tokens
        kwargs["special_first"] = special_first

        return method(self, **kwargs)

    return new_method


def check_ngram(method):
    """A wrapper that wrap a parameter checker to the original function."""

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
            raise ValueError("n needs to be a non-empty list of positive integers.")

        for gram in n:
            if not (isinstance(gram, int) and gram > 0):
                raise ValueError("n in ngram needs to be a positive number.")

        if left_pad is None:
            left_pad = ("", 0)

        if right_pad is None:
            right_pad = ("", 0)

        if not (isinstance(left_pad, tuple) and len(left_pad) == 2 and isinstance(left_pad[0], str) and isinstance(
                left_pad[1], int)):
            raise ValueError("left_pad needs to be a tuple of (str, int) str is pad token and int is pad_width.")

        if not (isinstance(right_pad, tuple) and len(right_pad) == 2 and isinstance(right_pad[0], str) and isinstance(
                right_pad[1], int)):
            raise ValueError("right_pad needs to be a tuple of (str, int) str is pad token and int is pad_width.")

        if not (left_pad[1] >= 0 and right_pad[1] >= 0):
            raise ValueError("padding width need to be positive numbers.")

        if separator is None:
            separator = " "

        if not isinstance(separator, str):
            raise ValueError("separator needs to be a string.")

        kwargs["n"] = n
        kwargs["left_pad"] = left_pad
        kwargs["right_pad"] = right_pad
        kwargs["separator"] = separator

        return method(self, **kwargs)

    return new_method


def check_pair_truncate(method):
    """Wrapper method to check the parameters of number of pair truncate."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        max_length = (list(args) + [None])[0]
        if "max_length" in kwargs:
            max_length = kwargs.get("max_length")
        if max_length is None:
            raise ValueError("max_length is not provided.")

        check_pos_int64(max_length)
        kwargs["max_length"] = max_length

        return method(self, **kwargs)

    return new_method


def check_to_number(method):
    """A wrapper that wraps a parameter check to the original function (ToNumber)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        data_type = (list(args) + [None])[0]
        if "data_type" in kwargs:
            data_type = kwargs.get("data_type")

        if data_type is None:
            raise ValueError("data_type is a mandatory parameter but was not provided.")

        if not isinstance(data_type, typing.Type):
            raise TypeError("data_type is not a MindSpore data type.")

        if data_type not in mstype.number_type:
            raise TypeError("data_type is not numeric data type.")

        kwargs["data_type"] = data_type

        return method(self, **kwargs)

    return new_method


def check_python_tokenizer(method):
    """A wrapper that wraps a parameter check to the original function (PythonTokenizer)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        tokenizer = (list(args) + [None])[0]
        if "tokenizer" in kwargs:
            tokenizer = kwargs.get("tokenizer")

        if tokenizer is None:
            raise ValueError("tokenizer is a mandatory parameter.")

        if not callable(tokenizer):
            raise TypeError("tokenizer is not a callable python function")

        kwargs["tokenizer"] = tokenizer

        return method(self, **kwargs)

    return new_method
