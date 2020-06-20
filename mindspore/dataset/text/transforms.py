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
import platform

import mindspore._c_dataengine as cde

from .utils import JiebaMode, NormalizeForm
from .validators import check_lookup, check_jieba_add_dict, \
    check_jieba_add_word, check_jieba_init, check_ngram, check_pair_truncate, \
    check_to_number
from ..core.datatypes import mstype_to_detype


class Lookup(cde.LookupOp):
    """
        Lookup operator that looks up a word to an id.
    Args:
        vocab(Vocab): a Vocab object.
        unknown(int, optional): default id to lookup a word that is out of vocab. If no argument is passed, 1 will be
            used to be the default id which is the convention for unknown_token <unk>. Otherwise, user is strongly
            encouraged to pass in the id for <unk> (default=None).
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
    Refer to https://en.wikipedia.org/wiki/N-gram#Examples for an overview of what n-gram is and how it works.

    Args:
        n([int, list]):  n in n-gram, n >= 1. n is a list of positive integers, for e.g. n=[4,3], The result
            would be a 4-gram followed by a 3-gram in the same tensor. If number of words is not enough to make up for
            a n-gram, an empty string would be returned. For e.g. 3 grams on ["mindspore","best"] would result in an
            empty string be produced.
        left_pad(tuple, optional): ("pad_token", pad_width). Padding performed on left side of the sequence. pad_width
            will be capped at n-1. left_pad=("_",2) would pad left side of the sequence with "__" (Default is None).
        right_pad(tuple, optional): ("pad_token", pad_width). Padding performed on right side of the sequence.
            pad_width will be capped at n-1. right_pad=("-":2) would pad right side of the sequence with "--"
            (Default is None).
        separator(str,optional): symbol used to join strings together. for e.g. if 2-gram the ["mindspore", "amazing"]
            with separator="-" the result would be ["mindspore-amazing"] (Default is None which means whitespace is
            used).
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


class WordpieceTokenizer(cde.WordpieceTokenizerOp):
    """
    Tokenize scalar token or 1-D tokens to 1-D subword tokens.

    Args
        vocab(Vocab): a Vocab object.
        suffix_indicator(string, optional): Used to show that the subword is the last part of a word(default '##').
        max_bytes_per_token(int, optional): Tokens exceeding this length will not be further split(default 100).
        unknown_token(string, optional): When we can not found the token: if 'unknown_token' is empty string,
            return the token directly, else return 'unknown_token'(default '[UNK]').
    """

    def __init__(self, vocab, suffix_indicator='##', max_bytes_per_token=100, unknown_token='[UNK]'):
        self.vocab = vocab
        self.suffix_indicator = suffix_indicator
        self.max_bytes_per_token = max_bytes_per_token
        self.unknown_token = unknown_token
        super().__init__(self.vocab, self.suffix_indicator, self.max_bytes_per_token, self.unknown_token)


if platform.system().lower() != 'windows':
    class WhitespaceTokenizer(cde.WhitespaceTokenizerOp):
        """
        Tokenize a scalar tensor of UTF-8 string on ICU defined whitespaces(such as: ' ', '\t', '\r', '\n').
        """


    class UnicodeScriptTokenizer(cde.UnicodeScriptTokenizerOp):
        """
        Tokenize a scalar tensor of UTF-8 string on Unicode script boundaries.

        Args:
            keep_whitespace(bool, optional): If or not emit whitespace tokens (default False)
        """

        def __init__(self, keep_whitespace=False):
            self.keep_whitespace = keep_whitespace
            super().__init__(self.keep_whitespace)


    class CaseFold(cde.CaseFoldOp):
        """
        Apply case fold operation on utf-8 string tensor.
        """


    DE_C_INTER_NORMALIZE_FORM = {
        NormalizeForm.NONE: cde.NormalizeForm.DE_NORMALIZE_NONE,
        NormalizeForm.NFC: cde.NormalizeForm.DE_NORMALIZE_NFC,
        NormalizeForm.NFKC: cde.NormalizeForm.DE_NORMALIZE_NFKC,
        NormalizeForm.NFD: cde.NormalizeForm.DE_NORMALIZE_NFD,
        NormalizeForm.NFKD: cde.NormalizeForm.DE_NORMALIZE_NFKD
    }


    class NormalizeUTF8(cde.NormalizeUTF8Op):
        """
        Apply normalize operation on utf-8 string tensor.

        Args:
            normalize_form(Enum, optional): Valid values are "NONE", "NFC", "NFKC", "NFD", "NFKD".
                If set "NONE", will do nothing for input string tensor.
                If set to any of "NFC", "NFKC", "NFD", "NFKD", will apply normalize operation(default "NFKC").
                See http://unicode.org/reports/tr15/ for details.
        """

        def __init__(self, normalize_form=NormalizeForm.NFKC):
            self.normalize_form = DE_C_INTER_NORMALIZE_FORM[normalize_form]
            super().__init__(self.normalize_form)


    class RegexReplace(cde.RegexReplaceOp):
        """
        Replace utf-8 string tensor with 'replace' according to regular expression 'pattern'.
        See http://userguide.icu-project.org/strings/regexp for support regex pattern.

        Args:
            pattern(string): the regex expression patterns.
            replace(string): the string to replace matched element.
            replace_all(bool, optional): If False, only replace first matched element;
                if True, replace all matched elements(default True).
        """

        def __init__(self, pattern, replace, replace_all=True):
            self.pattern = pattern
            self.replace = replace
            self.replace_all = replace_all
            super().__init__(self.pattern, self.replace, self.replace_all)


    class RegexTokenizer(cde.RegexTokenizerOp):
        """
        Tokenize a scalar tensor of UTF-8 string by regex expression pattern.
        See http://userguide.icu-project.org/strings/regexp for support regex pattern.

        Args:
            delim_pattern(string): The pattern of regex delimiters.
                The original string will be split by matched elements.
            keep_delim_pattern(string, optional): The string matched by 'delim_pattern' can be kept as a token
                if it can be matched by 'keep_delim_pattern'. And the default value is empty string(''),
                in this situation, delimiters will not kept as a output token.
        """

        def __init__(self, delim_pattern, keep_delim_pattern=''):
            self.delim_pattern = delim_pattern
            self.keep_delim_pattern = keep_delim_pattern
            super().__init__(self.delim_pattern, self.keep_delim_pattern)


    class BasicTokenizer(cde.BasicTokenizerOp):
        """
        Tokenize a scalar tensor of UTF-8 string by specific rules.

        Args:
            lower_case(bool, optional): If True, apply CaseFold, NormalizeUTF8(NFD mode), RegexReplace operation
                on input text to make the text to lower case and strip accents characters; If False, only apply
                NormalizeUTF8('normalization_form' mode) operation on input text(default False).
            keep_whitespace(bool, optional), If True, the whitespace will be kept in out tokens(default False).
            normalization_form(Enum, optional), Used to specify a specific normlaize mode,
                only effective when 'lower_case' is False. See NormalizeUTF8 for details(default 'NONE').
            preserve_unused_token(bool, optional), If True, do not split special tokens like
                '[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]'(default True).
        """

        def __init__(self, lower_case=False, keep_whitespace=False,
                     normalization_form=NormalizeForm.NONE, preserve_unused_token=True):
            self.lower_case = lower_case
            self.keep_whitespace = keep_whitespace
            self.normalization_form = DE_C_INTER_NORMALIZE_FORM[normalization_form]
            self.preserve_unused_token = preserve_unused_token
            super().__init__(self.lower_case, self.keep_whitespace,
                             self.normalization_form, self.preserve_unused_token)


    class BertTokenizer(cde.BertTokenizerOp):
        """
        Tokenizer used for Bert text process.

        Args:
            vocab(Vocab): a Vocab object.
            suffix_indicator(string, optional): Used to show that the subword is the last part of a word(default '##').
            max_bytes_per_token(int, optional): Tokens exceeding this length will not be further split(default 100).
            unknown_token(string, optional): When we can not found the token: if 'unknown_token' is empty string,
                return the token directly, else return 'unknown_token'(default '[UNK]').
            lower_case(bool, optional): If True, apply CaseFold, NormalizeUTF8(NFD mode), RegexReplace operation
                on input text to make the text to lower case and strip accents characters; If False, only apply
                NormalizeUTF8('normalization_form' mode) operation on input text(default False).
            keep_whitespace(bool, optional), If True, the whitespace will be kept in out tokens(default False).
            normalization_form(Enum, optional), Used to specify a specific normlaize mode,
                only effective when 'lower_case' is False. See NormalizeUTF8 for details(default 'NONE').
            preserve_unused_token(bool, optional), If True, do not split special tokens like
                '[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]'(default True).
        """

        def __init__(self, vocab, suffix_indicator='##', max_bytes_per_token=100,
                     unknown_token='[UNK]', lower_case=False, keep_whitespace=False,
                     normalization_form=NormalizeForm.NONE, preserve_unused_token=True):
            self.vocab = vocab
            self.suffix_indicator = suffix_indicator
            self.max_bytes_per_token = max_bytes_per_token
            self.unknown_token = unknown_token
            self.lower_case = lower_case
            self.keep_whitespace = keep_whitespace
            self.normalization_form = DE_C_INTER_NORMALIZE_FORM[normalization_form]
            self.preserve_unused_token = preserve_unused_token
            super().__init__(self.vocab, self.suffix_indicator, self.max_bytes_per_token, self.unknown_token,
                             self.lower_case, self.keep_whitespace, self.normalization_form, self.preserve_unused_token)


class TruncateSequencePair(cde.TruncateSequencePairOp):
    """
    Truncate a pair of rank-1 tensors such that the total length is less than max_length.

    This operation takes two input tensors and returns two output Tenors.

    Args:
        max_length(int): Maximum length required.

    Examples:
        >>> # Data before
        >>> # |  col1   |  col2   |
        >>> # +---------+---------|
        >>> # | [1,2,3] | [4,5]   |
        >>> # +---------+---------+
        >>> data = data.map(operations=TruncateSequencePair(4))
        >>> # Data after
        >>> # |  col1   |  col2   |
        >>> # +---------+---------+
        >>> # | [1,2]   | [4,5]   |
        >>> # +---------+---------+
    """

    @check_pair_truncate
    def __init__(self, max_length):
        super().__init__(max_length)


class ToNumber(cde.ToNumberOp):
    """
    Tensor operation to convert every element of a string tensor to a number.

    Strings are casted according to the rules specified in the following links:
    https://en.cppreference.com/w/cpp/string/basic_string/stof,
    https://en.cppreference.com/w/cpp/string/basic_string/stoul,
    except that any strings which represent negative numbers cannot be casted to an
    unsigned integer type.

    Args:
        data_type (mindspore.dtype): mindspore.dtype to be casted to. Must be
            a numeric type.

    Raises:
        RuntimeError: If strings are invalid to cast, or are out of range after being casted.
    """

    @check_to_number
    def __init__(self, data_type):
        data_type = mstype_to_detype(data_type)
        self.data_type = str(data_type)
        super().__init__(data_type)
