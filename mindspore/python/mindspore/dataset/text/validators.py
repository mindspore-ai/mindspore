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
import numpy as np

import mindspore._c_dataengine as cde
import mindspore.common.dtype as mstype
from mindspore._c_expression import typing

import mindspore.dataset.text as text
from ..core.validator_helpers import parse_user_args, type_check, type_check_list, check_uint32, \
    INT32_MAX, check_value, check_positive, check_pos_int32, check_filename, check_non_negative_int32


def check_add_token(method):
    """Wrapper method to check the parameters of add token."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [token, begin], _ = parse_user_args(method, *args, **kwargs)
        type_check(token, (str,), "token")
        type_check(begin, (bool,), "begin")
        return method(self, *args, **kwargs)

    return new_method


def check_unique_list_of_words(words, arg_name):
    """Check that words is a list and each element is a str without any duplication"""

    type_check(words, (list,), arg_name)
    words_set = set()
    for word in words:
        type_check(word, (str,), arg_name)
        if word in words_set:
            raise ValueError(arg_name + " contains duplicate word: " + word + ".")
        words_set.add(word)
    return words_set


def check_lookup(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [vocab, unknown_token, data_type], _ = parse_user_args(method, *args, **kwargs)

        if unknown_token is not None:
            type_check(unknown_token, (str,), "unknown_token")

        type_check(vocab, (text.Vocab,), "vocab is not an instance of text.Vocab.")
        type_check(vocab.c_vocab, (cde.Vocab,), "vocab.c_vocab is not an instance of cde.Vocab.")
        type_check(data_type, (typing.Type,), "data_type")

        return method(self, *args, **kwargs)

    return new_method


def check_from_file(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [file_path, delimiter, vocab_size, special_tokens, special_first], _ = parse_user_args(method, *args,
                                                                                               **kwargs)
        if special_tokens is not None:
            check_unique_list_of_words(special_tokens, "special_tokens")
        type_check_list([file_path, delimiter], (str,), ["file_path", "delimiter"])
        if vocab_size is not None:
            check_positive(vocab_size, "vocab_size")
        type_check(special_first, (bool,), special_first)

        return method(self, *args, **kwargs)

    return new_method


def check_vocab(c_vocab):
    """Check the c_vocab of Vocab is initialized or not"""

    if not isinstance(c_vocab, cde.Vocab):
        error = "The Vocab has not built yet, got type {0}. ".format(type(c_vocab))
        suggestion = "Use Vocab.from_dataset(), Vocab.from_list(), Vocab.from_file() or Vocab.from_dict() " \
                     "to build a Vocab."
        raise RuntimeError(error + suggestion)


def check_tokens_to_ids(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [tokens], _ = parse_user_args(method, *args, **kwargs)
        type_check(tokens, (str, list, np.ndarray), "tokens")
        if isinstance(tokens, list):
            param_names = ["tokens[{0}]".format(i) for i in range(len(tokens))]
            type_check_list(tokens, (str, np.str_), param_names)

        return method(self, *args, **kwargs)

    return new_method


def check_ids_to_tokens(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ids], _ = parse_user_args(method, *args, **kwargs)
        type_check(ids, (int, list, np.ndarray), "ids")
        if isinstance(ids, int):
            check_value(ids, (0, INT32_MAX), "ids")
        if isinstance(ids, list):
            for index, id_ in enumerate(ids):
                type_check(id_, (int, np.int_), "ids[{}]".format(index))
                check_value(id_, (0, INT32_MAX), "ids[{}]".format(index))

        return method(self, *args, **kwargs)

    return new_method


def check_from_list(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [word_list, special_tokens, special_first], _ = parse_user_args(method, *args, **kwargs)

        word_set = check_unique_list_of_words(word_list, "word_list")
        if special_tokens is not None:
            token_set = check_unique_list_of_words(special_tokens, "special_tokens")

            intersect = word_set.intersection(token_set)

            if intersect != set():
                raise ValueError("special_tokens and word_list contain duplicate word :" + str(intersect) + ".")

        type_check(special_first, (bool,), "special_first")

        return method(self, *args, **kwargs)

    return new_method


def check_from_dict(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [word_dict], _ = parse_user_args(method, *args, **kwargs)

        type_check(word_dict, (dict,), "word_dict")

        for word, word_id in word_dict.items():
            type_check(word, (str,), "word")
            type_check(word_id, (int,), "word_id")
            check_value(word_id, (0, INT32_MAX), "word_id")
        return method(self, *args, **kwargs)

    return new_method


def check_jieba_init(method):
    """Wrapper method to check the parameters of jieba init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [hmm_path, mp_path, _, with_offsets], _ = parse_user_args(method, *args, **kwargs)

        if hmm_path is None:
            raise ValueError("The dict of HMMSegment in cppjieba is not provided.")
        if not isinstance(hmm_path, str):
            raise TypeError("Wrong input type for hmm_path, should be string.")
        if mp_path is None:
            raise ValueError("The dict of MPSegment in cppjieba is not provided.")
        if not isinstance(mp_path, str):
            raise TypeError("Wrong input type for mp_path, should be string.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_jieba_add_word(method):
    """Wrapper method to check the parameters of jieba add word."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [word, freq], _ = parse_user_args(method, *args, **kwargs)
        if word is None:
            raise ValueError("word is not provided.")
        if freq is not None:
            check_uint32(freq)
        return method(self, *args, **kwargs)

    return new_method


def check_jieba_add_dict(method):
    """Wrapper method to check the parameters of add dict."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        parse_user_args(method, *args, **kwargs)
        return method(self, *args, **kwargs)

    return new_method


def check_with_offsets(method):
    """Wrapper method to check if with_offsets is the only one parameter."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [with_offsets], _ = parse_user_args(method, *args, **kwargs)
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_unicode_script_tokenizer(method):
    """Wrapper method to check the parameter of UnicodeScriptTokenizer."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [keep_whitespace, with_offsets], _ = parse_user_args(method, *args, **kwargs)
        if not isinstance(keep_whitespace, bool):
            raise TypeError("Wrong input type for keep_whitespace, should be boolean.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_wordpiece_tokenizer(method):
    """Wrapper method to check the parameter of WordpieceTokenizer."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [vocab, suffix_indicator, max_bytes_per_token, unknown_token, with_offsets], _ = \
            parse_user_args(method, *args, **kwargs)
        if vocab is None:
            raise ValueError("vocab is not provided.")
        if not isinstance(vocab, text.Vocab):
            raise TypeError("Wrong input type for vocab, should be text.Vocab object.")
        if not isinstance(suffix_indicator, str):
            raise TypeError("Wrong input type for suffix_indicator, should be string.")
        if not isinstance(unknown_token, str):
            raise TypeError("Wrong input type for unknown_token, should be string.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        check_uint32(max_bytes_per_token)
        return method(self, *args, **kwargs)

    return new_method


def check_regex_replace(method):
    """Wrapper method to check the parameter of RegexReplace."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [pattern, replace, replace_all], _ = parse_user_args(method, *args, **kwargs)
        type_check(pattern, (str,), "pattern")
        type_check(replace, (str,), "replace")
        type_check(replace_all, (bool,), "replace_all")
        return method(self, *args, **kwargs)

    return new_method


def check_regex_tokenizer(method):
    """Wrapper method to check the parameter of RegexTokenizer."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [delim_pattern, keep_delim_pattern, with_offsets], _ = parse_user_args(method, *args, **kwargs)
        if delim_pattern is None:
            raise ValueError("delim_pattern is not provided.")
        if not isinstance(delim_pattern, str):
            raise TypeError("Wrong input type for delim_pattern, should be string.")
        if not isinstance(keep_delim_pattern, str):
            raise TypeError("Wrong input type for keep_delim_pattern, should be string.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_basic_tokenizer(method):
    """Wrapper method to check the parameter of RegexTokenizer."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [lower_case, keep_whitespace, _, preserve_unused, with_offsets], _ = \
            parse_user_args(method, *args, **kwargs)
        if not isinstance(lower_case, bool):
            raise TypeError("Wrong input type for lower_case, should be boolean.")
        if not isinstance(keep_whitespace, bool):
            raise TypeError("Wrong input type for keep_whitespace, should be boolean.")
        if not isinstance(preserve_unused, bool):
            raise TypeError("Wrong input type for preserve_unused_token, should be boolean.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_bert_tokenizer(method):
    """Wrapper method to check the parameter of BertTokenizer."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [vocab, suffix_indicator, max_bytes_per_token, unknown_token, lower_case, keep_whitespace, _,
         preserve_unused_token, with_offsets], _ = parse_user_args(method, *args, **kwargs)
        if vocab is None:
            raise ValueError("vacab is not provided.")
        if not isinstance(vocab, text.Vocab):
            raise TypeError("Wrong input type for vocab, should be text.Vocab object.")
        if not isinstance(suffix_indicator, str):
            raise TypeError("Wrong input type for suffix_indicator, should be string.")
        if not isinstance(max_bytes_per_token, int):
            raise TypeError("Wrong input type for max_bytes_per_token, should be int.")
        check_uint32(max_bytes_per_token)

        if not isinstance(unknown_token, str):
            raise TypeError("Wrong input type for unknown_token, should be string.")
        if not isinstance(lower_case, bool):
            raise TypeError("Wrong input type for lower_case, should be boolean.")
        if not isinstance(keep_whitespace, bool):
            raise TypeError("Wrong input type for keep_whitespace, should be boolean.")
        if not isinstance(preserve_unused_token, bool):
            raise TypeError("Wrong input type for preserve_unused_token, should be boolean.")
        if not isinstance(with_offsets, bool):
            raise TypeError("Wrong input type for with_offsets, should be boolean.")
        return method(self, *args, **kwargs)

    return new_method


def check_from_dataset(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        [_, columns, freq_range, top_k, special_tokens, special_first], _ = parse_user_args(method, *args,
                                                                                            **kwargs)
        if columns is not None:
            if not isinstance(columns, list):
                columns = [columns]
                type_check_list(columns, (str,), "col")

        if freq_range is not None:
            type_check(freq_range, (tuple,), "freq_range")

            if len(freq_range) != 2:
                raise ValueError("freq_range needs to be a tuple of 2 element.")

            for num in freq_range:
                if num is not None and (not isinstance(num, int)):
                    raise ValueError(
                        "freq_range needs to be either None or a tuple of 2 integers or an int and a None.")

            if isinstance(freq_range[0], int) and isinstance(freq_range[1], int):
                if freq_range[0] > freq_range[1] or freq_range[0] < 0:
                    raise ValueError("frequency range [a,b] should be 0 <= a <= b (a,b are inclusive).")

        type_check(top_k, (int, type(None)), "top_k")

        if isinstance(top_k, int):
            check_positive(top_k, "top_k")
        type_check(special_first, (bool,), "special_first")

        if special_tokens is not None:
            check_unique_list_of_words(special_tokens, "special_tokens")

        return method(self, *args, **kwargs)

    return new_method


def check_slidingwindow(method):
    """A wrapper that wraps a parameter checker to the original function(sliding window operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [width, axis], _ = parse_user_args(method, *args, **kwargs)
        check_pos_int32(width, "width")
        type_check(axis, (int,), "axis")
        return method(self, *args, **kwargs)

    return new_method


def check_ngram(method):
    """A wrapper that wraps a parameter checker to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [n, left_pad, right_pad, separator], _ = parse_user_args(method, *args, **kwargs)

        if isinstance(n, int):
            n = [n]

        if not (isinstance(n, list) and n != []):
            raise ValueError("n needs to be a non-empty list of positive integers.")

        for i, gram in enumerate(n):
            type_check(gram, (int,), "gram[{0}]".format(i))
            check_positive(gram, "gram_{}".format(i))

        if not (isinstance(left_pad, tuple) and len(left_pad) == 2 and isinstance(left_pad[0], str) and isinstance(
                left_pad[1], int)):
            raise ValueError("left_pad needs to be a tuple of (str, int) str is pad token and int is pad_width.")

        if not (isinstance(right_pad, tuple) and len(right_pad) == 2 and isinstance(right_pad[0], str) and isinstance(
                right_pad[1], int)):
            raise ValueError("right_pad needs to be a tuple of (str, int) str is pad token and int is pad_width.")

        if not (left_pad[1] >= 0 and right_pad[1] >= 0):
            raise ValueError("padding width need to be positive numbers.")

        type_check(separator, (str,), "separator")

        kwargs["n"] = n
        kwargs["left_pad"] = left_pad
        kwargs["right_pad"] = right_pad
        kwargs["separator"] = separator

        return method(self, **kwargs)

    return new_method


def check_truncate(method):
    """Wrapper method to check the parameters of number of truncate."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [max_seq_len], _ = parse_user_args(method, *args, **kwargs)
        check_pos_int32(max_seq_len, "max_seq_len")
        return method(self, *args, **kwargs)

    return new_method


def check_pair_truncate(method):
    """Wrapper method to check the parameters of number of pair truncate."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        parse_user_args(method, *args, **kwargs)
        return method(self, *args, **kwargs)

    return new_method


def check_to_number(method):
    """A wrapper that wraps a parameter check to the original function (ToNumber)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(data_type, (typing.Type,), "data_type")

        if data_type not in mstype.number_type:
            raise TypeError("data_type: " + str(data_type) + " is not numeric data type.")

        return method(self, *args, **kwargs)

    return new_method


def check_python_tokenizer(method):
    """A wrapper that wraps a parameter check to the original function (PythonTokenizer)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [tokenizer], _ = parse_user_args(method, *args, **kwargs)

        if not callable(tokenizer):
            raise TypeError("tokenizer is not a callable Python function.")

        return method(self, *args, **kwargs)

    return new_method


def check_from_dataset_sentencepiece(method):
    """A wrapper that wraps a parameter checker to the original function (from_dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [_, col_names, vocab_size, character_coverage, model_type, params], _ = parse_user_args(method, *args, **kwargs)

        if col_names is not None:
            type_check_list(col_names, (str,), "col_names")

        if vocab_size is not None:
            check_uint32(vocab_size, "vocab_size")
        else:
            raise TypeError("vocab_size must be provided.")

        if character_coverage is not None:
            type_check(character_coverage, (float,), "character_coverage")

        if model_type is not None:
            from .utils import SentencePieceModel
            type_check(model_type, (str, SentencePieceModel), "model_type")

        if params is not None:
            type_check(params, (dict,), "params")

        return method(self, *args, **kwargs)

    return new_method


def check_from_file_sentencepiece(method):
    """A wrapper that wraps a parameter checker to the original function (from_file)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [file_path, vocab_size, character_coverage, model_type, params], _ = parse_user_args(method, *args, **kwargs)

        if file_path is not None:
            type_check(file_path, (list,), "file_path")

        if vocab_size is not None:
            check_uint32(vocab_size, "vocab_size")

        if character_coverage is not None:
            type_check(character_coverage, (float,), "character_coverage")

        if model_type is not None:
            from .utils import SentencePieceModel
            type_check(model_type, (str, SentencePieceModel), "model_type")

        if params is not None:
            type_check(params, (dict,), "params")

        return method(self, *args, **kwargs)

    return new_method


def check_save_model(method):
    """A wrapper that wraps a parameter checker to the original function (save_model)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [vocab, path, filename], _ = parse_user_args(method, *args, **kwargs)

        if vocab is not None:
            type_check(vocab, (text.SentencePieceVocab,), "vocab")

        if path is not None:
            type_check(path, (str,), "path")

        if filename is not None:
            type_check(filename, (str,), "filename")

        return method(self, *args, **kwargs)

    return new_method


def check_sentence_piece_tokenizer(method):

    """A wrapper that wraps a parameter checker to the original function."""

    from .utils import SPieceTokenizerOutType
    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mode, out_type], _ = parse_user_args(method, *args, **kwargs)

        type_check(mode, (str, text.SentencePieceVocab), "mode is not an instance of str or text.SentencePieceVocab.")
        type_check(out_type, (SPieceTokenizerOutType,), "out_type is not an instance of SPieceTokenizerOutType")

        return method(self, *args, **kwargs)

    return new_method


def check_from_file_vectors(method):
    """A wrapper that wraps a parameter checker to from_file of class Vectors."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [file_path, max_vectors], _ = parse_user_args(method, *args, **kwargs)

        type_check(file_path, (str,), "file_path")
        check_filename(file_path)
        if max_vectors is not None:
            type_check(max_vectors, (int,), "max_vectors")
            check_non_negative_int32(max_vectors, "max_vectors")

        return method(self, *args, **kwargs)

    return new_method


def check_to_vectors(method):
    """A wrapper that wraps a parameter checker to ToVectors."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [vectors, unk_init, lower_case_backup], _ = parse_user_args(method, *args, **kwargs)

        type_check(vectors, (cde.Vectors,), "vectors")
        if unk_init is not None:
            type_check(unk_init, (list, tuple), "unk_init")
            for i, value in enumerate(unk_init):
                type_check(value, (int, float), "unk_init[{0}]".format(i))
        type_check(lower_case_backup, (bool,), "lower_case_backup")
        return method(self, *args, **kwargs)

    return new_method
