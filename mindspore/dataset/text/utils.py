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
The module text.utils provides some general methods for NLP text processing.
For example, you can use Vocab to build a dictionary,
use to_bytes and to_str to encode and decode strings into a specified format.
"""
from enum import IntEnum

import numpy as np
import mindspore._c_dataengine as cde

from .validators import check_from_file, check_from_list, check_from_dict, check_from_dataset, \
    check_from_dataset_sentencepiece, check_from_file_sentencepiece, check_save_model

__all__ = [
    "Vocab", "SentencePieceVocab", "to_str", "to_bytes"
]


class Vocab(cde.Vocab):
    """
    Vocab object that is used to lookup a word.

    It contains a map that maps each word(str) to an id (int).
    """

    @classmethod
    @check_from_dataset
    def from_dataset(cls, dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True):
        """
        Build a vocab from a dataset.

        This would collect all unique words in a dataset and return a vocab within
        the frequency range specified by user in freq_range. User would be warned if no words fall into the frequency.
        Words in vocab are ordered from highest frequency to lowest frequency. Words with the same frequency would be
        ordered lexicographically.

        Args:
            dataset(Dataset): dataset to build vocab from.
            columns(list[str], optional): column names to get words from. It can be a list of column names.
                (default=None, where all columns will be used. If any column isn't string type, will return error).
            freq_range(tuple, optional): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency=0 is the same as
                min_frequency=1. max_frequency > total_words is the same as max_frequency = total_words.
                min_frequency/max_frequency can be None, which corresponds to 0/total_words separately
                (default=None, all words are included).
            top_k(int, optional): top_k > 0. Number of words to be built into vocab. top_k most frequent words are
                taken. top_k is taken after freq_range. If not enough top_k, all words will be taken (default=None,
                all words are included).
            special_tokens(list, optional):  a list of strings, each one is a special token. for example
                special_tokens=["<pad>","<unk>"] (default=None, no special tokens will be added).
            special_first(bool, optional): whether special_tokens will be prepended/appended to vocab. If special_tokens
                is specified and special_first is set to True, special_tokens will be prepended (default=True).

        Returns:
            Vocab, vocab built from the dataset.
        """
        return dataset.build_vocab(columns, freq_range, top_k, special_tokens, special_first)

    @classmethod
    @check_from_list
    def from_list(cls, word_list, special_tokens=None, special_first=True):
        """
        Build a vocab object from a list of word.

        Args:
            word_list(list): a list of string where each element is a word of type string.
            special_tokens(list, optional):  a list of strings, each one is a special token. for example
                special_tokens=["<pad>","<unk>"] (default=None, no special tokens will be added).
            special_first(bool, optional): whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to True, special_tokens will be prepended (default=True).

        Returns:
            Vocab, vocab built from the `list`.
        """
        if special_tokens is None:
            special_tokens = []
        return super().from_list(word_list, special_tokens, special_first)

    @classmethod
    @check_from_file
    def from_file(cls, file_path, delimiter="", vocab_size=None, special_tokens=None, special_first=True):
        """
        Build a vocab object from a list of word.

        Args:
            file_path (str): path to the file which contains the vocab list.
            delimiter (str, optional): a delimiter to break up each line in file, the first element is taken to be
                the word (default="").
            vocab_size (int, optional): number of words to read from file_path (default=None, all words are taken).
            special_tokens (list, optional):  a list of strings, each one is a special token. for example
                special_tokens=["<pad>","<unk>"] (default=None, no special tokens will be added).
            special_first (bool, optional): whether special_tokens will be prepended/appended to vocab,
                If special_tokens is specified and special_first is set to True,
                special_tokens will be prepended (default=True).

        Returns:
            Vocab, vocab built from the file.
        """
        if vocab_size is None:
            vocab_size = -1
        if special_tokens is None:
            special_tokens = []
        return super().from_file(file_path, delimiter, vocab_size, special_tokens, special_first)

    @classmethod
    @check_from_dict
    def from_dict(cls, word_dict):
        """
        Build a vocab object from a dict.

        Args:
            word_dict (dict): dict contains word and id pairs, where word should be str and id be int. id is recommended
                to start from 0 and be continuous. ValueError will be raised if id is negative.

        Returns:
            Vocab, vocab built from the `dict`.
        """

        return super().from_dict(word_dict)


class SentencePieceVocab(cde.SentencePieceVocab):
    """
    SentencePiece obiect that is used to segmentate words
    """

    @classmethod
    @check_from_dataset_sentencepiece
    def from_dataset(cls, dataset, col_names, vocab_size, character_coverage, model_type, params):
        """
        Build a sentencepiece from a dataset

        Args:
            dataset(Dataset): Dataset to build sentencepiece.
            col_names(list): The list of the col name.
            vocab_size(int): Vocabulary size.
            character_coverage(float): Amount of characters covered by the model, good defaults are: 0.9995 for
                languages. with rich character set like Japanese or Chinese and 1.0 for other languages with small
                character set.
            model_type(SentencePieceModel): Choose from UNIGRAM (default), BPE, CHAR, or WORD. The input sentence
                must be pretokenized when using word type.
            params(dict): A dictionary with no incoming parameters.

        Returns:
            SentencePieceVocab, vocab built from the dataset.
        """

        return dataset.build_sentencepiece_vocab(col_names, vocab_size, character_coverage,
                                                 DE_C_INTER_SENTENCEPIECE_MODE[model_type], params)

    @classmethod
    @check_from_file_sentencepiece
    def from_file(cls, file_path, vocab_size, character_coverage, model_type, params):
        """
        Build a SentencePiece object from a list of word.

        Args:
            file_path(list): Path to the file which contains the sentencepiece list.
            vocab_size(int): Vocabulary size, the type of uint32_t.
            character_coverage(float): Amount of characters covered by the model, good defaults are: 0.9995 for
                languages. with rich character set like Japanse or Chinese and 1.0 for other languages with small
                character set.
            model_type(SentencePieceModel): Choose from unigram (default), bpe, char, or word. The input sentence
                must be pretokenized when using word type.
            params(dict): A dictionary with no incoming parameters(The parameters are derived from SentencePiece
                library).

                .. code-block::

                    input_sentence_size 0
                    max_sentencepiece_length 16

        Returns:
            SentencePieceVocab, vocab built from the file.
        """
        return super().from_file(file_path, vocab_size, character_coverage,
                                 DE_C_INTER_SENTENCEPIECE_MODE[model_type], params)

    @classmethod
    @check_save_model
    def save_model(cls, vocab, path, filename):
        """
        Save model to filepath

        Args:
            vocab(SentencePieceVocab): A sentencepiece object.
            path(str): Path to store model.
            filename(str): The name of the file.
        """
        super().save_model(vocab, path, filename)


def to_str(array, encoding='utf8'):
    """
    Convert NumPy array of `bytes` to array of `str` by decoding each element based on charset `encoding`.

    Args:
        array (numpy.ndarray): Array of type `bytes` representing strings.
        encoding (str): Indicating the charset for decoding.

    Returns:
        numpy.ndarray, NumPy array of `str`.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError('input should be a NumPy array.')

    return np.char.decode(array, encoding)


def to_bytes(array, encoding='utf8'):
    """
    Convert NumPy array of `str` to array of `bytes` by encoding each element based on charset `encoding`.

    Args:
        array (numpy.ndarray): Array of type `str` representing strings.
        encoding (str): Indicating the charset for encoding.

    Returns:
        numpy.ndarray, NumPy array of `bytes`.
    """

    if not isinstance(array, np.ndarray):
        raise ValueError('input should be a NumPy array.')

    return np.char.encode(array, encoding)


class JiebaMode(IntEnum):
    """An enumeration for JiebaTokenizer, effective enumeration types are MIX, MP, HMM."""
    MIX = 0
    MP = 1
    HMM = 2


class NormalizeForm(IntEnum):
    """An enumeration for NormalizeUTF8, effective enumeration types are NONE, NFC, NFKC, NFD, NFKD."""
    NONE = 0
    NFC = 1
    NFKC = 2
    NFD = 3
    NFKD = 4


class SentencePieceModel(IntEnum):
    """An enumeration for SentencePieceModel, effective enumeration types are UNIGRAM, BPE, CHAR, WORD."""
    UNIGRAM = 0
    BPE = 1
    CHAR = 2
    WORD = 3


DE_C_INTER_SENTENCEPIECE_MODE = {
    SentencePieceModel.UNIGRAM: cde.SentencePieceModel.DE_SENTENCE_PIECE_UNIGRAM,
    SentencePieceModel.BPE: cde.SentencePieceModel.DE_SENTENCE_PIECE_BPE,
    SentencePieceModel.CHAR: cde.SentencePieceModel.DE_SENTENCE_PIECE_CHAR,
    SentencePieceModel.WORD: cde.SentencePieceModel.DE_SENTENCE_PIECE_WORD
}


class SPieceTokenizerOutType(IntEnum):
    """An enumeration for SPieceTokenizerOutType, effective enumeration types are STRING, INT."""
    STRING = 0
    INT = 1


class SPieceTokenizerLoadType(IntEnum):
    """An enumeration for SPieceTokenizerLoadType, effective enumeration types are FILE, MODEL."""
    FILE = 0
    MODEL = 1
