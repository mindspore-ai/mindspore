# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from .validators import check_vocab, check_from_file, check_from_list, check_from_dict, check_from_dataset, \
    check_from_dataset_sentencepiece, check_from_file_sentencepiece, check_save_model, \
    check_from_file_vectors, check_tokens_to_ids, check_ids_to_tokens


class CharNGram(cde.CharNGram):
    """
    CharNGram object that is used to map tokens into pre-trained vectors.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Build a `CharNGram` vector from a file.

        Args:
            file_path (str): Path of the file that contains the `CharNGram` vectors.
            max_vectors (int, optional): This can be used to limit the number of pre-trained vectors loaded.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                passing `max_vectors` can limit the size of the loaded set. Default: None, no limit.

        Returns:
            CharNGram, CharNGram vector build from a file.

        Raises:
            RuntimeError: If `file_path` contains invalid data.
            ValueError: If `max_vectors` is invalid.
            TypeError: If `max_vectors` is not type of integer.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> char_n_gram = text.CharNGram.from_file("/path/to/char_n_gram/file", max_vectors=None)
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class FastText(cde.FastText):
    """
    FastText object that is used to map tokens into vectors.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Build a FastText vector from a file.

        Args:
            file_path (str): Path of the file that contains the vectors. The shuffix of pre-trained vector sets
                must be `*.vec` .
            max_vectors (int, optional): This can be used to limit the number of pre-trained vectors loaded.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                passing `max_vectors` can limit the size of the loaded set. Default: None, no limit.

        Returns:
            FastText, FastText vector build from a file.

        Raises:
            RuntimeError: If `file_path` contains invalid data.
            ValueError: If `max_vectors` is invalid.
            TypeError: If `max_vectors` is not type of integer.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> fast_text = text.FastText.from_file("/path/to/fast_text/file", max_vectors=None)
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class GloVe(cde.GloVe):
    """
    GloVe object that is used to map tokens into vectors.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Build a GloVe vector from a file.

        Args:
            file_path (str): Path of the file that contains the vectors. The format of pre-trained vector sets
                must be `glove.6B.*.txt` .
            max_vectors (int, optional): This can be used to limit the number of pre-trained vectors loaded.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                passing `max_vectors` can limit the size of the loaded set. Default: None, no limit.

        Returns:
            GloVe, GloVe vector build from a file.

        Raises:
            RuntimeError: If `file_path` contains invalid data.
            ValueError: If `max_vectors` is invalid.
            TypeError: If `max_vectors` is not type of integer.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> glove = text.GloVe.from_file("/path/to/glove/file", max_vectors=None)
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class JiebaMode(IntEnum):
    """
    An enumeration for :class:`mindspore.dataset.text.JiebaTokenizer` .

    Possible enumeration values are: JiebaMode.MIX, JiebaMode.MP, JiebaMode.HMM.

    - JiebaMode.MIX: tokenize with a mix of MPSegment and HMMSegment algorithm.
    - JiebaMode.MP: tokenize with MPSegment algorithm.
    - JiebaMode.HMM: tokenize with Hidden Markov Model Segment algorithm.
    """

    MIX = 0
    MP = 1
    HMM = 2


class NormalizeForm(IntEnum):
    """
    Enumeration class for `Unicode normalization forms <http://unicode.org/reports/tr15/>`_ .

    Possible enumeration values are: NormalizeForm.NONE, NormalizeForm.NFC, NormalizeForm.NFKC, NormalizeForm.NFD
    and NormalizeForm.NFKD.

    - NormalizeForm.NONE: no normalization.
    - NormalizeForm.NFC: Canonical Decomposition, followed by Canonical Composition.
    - NormalizeForm.NFKC: Compatibility Decomposition, followed by Canonical Composition.
    - NormalizeForm.NFD: Canonical Decomposition.
    - NormalizeForm.NFKD: Compatibility Decomposition.
    """

    NONE = 0
    NFC = 1
    NFKC = 2
    NFD = 3
    NFKD = 4


class SentencePieceModel(IntEnum):
    """
    An enumeration for SentencePieceModel.

    Possible enumeration values are: SentencePieceModel.UNIGRAM, SentencePieceModel.BPE, SentencePieceModel.CHAR,
    SentencePieceModel.WORD.

    - SentencePieceModel.UNIGRAM: Unigram Language Model means the next word in the sentence is assumed to be
      independent of the previous words generated by the model.
    - SentencePieceModel.BPE: refers to byte pair encoding algorithm, which replaces the most frequent pair of bytes in
      a sentence with a single, unused byte.
    - SentencePieceModel.CHAR: refers to char based sentencePiece Model type.
    - SentencePieceModel.WORD: refers to word based sentencePiece Model type.
    """

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


class SentencePieceVocab:
    """
    SentencePiece object that is used to do words segmentation.
    """

    def __init__(self):
        self.c_sentence_piece_vocab = None

    @classmethod
    @check_from_dataset_sentencepiece
    def from_dataset(cls, dataset, col_names, vocab_size, character_coverage, model_type, params):
        """
        Build a SentencePiece from a dataset.

        Args:
            dataset (Dataset): Dataset to build SentencePiece.
            col_names (list): The list of the col name.
            vocab_size (int): Vocabulary size.
            character_coverage (float): Amount of characters covered by the model, good defaults are: 0.9995 for
                languages with rich character set like Japanese or Chinese and 1.0 for other languages with small
                character set.
            model_type (SentencePieceModel): It can be any of [SentencePieceModel.UNIGRAM, SentencePieceModel.BPE,
                SentencePieceModel.CHAR, SentencePieceModel.WORD], default is SentencePieceModel.UNIGRAM. The input
                sentence must be pre-tokenized when using SentencePieceModel.WORD type.

                - SentencePieceModel.UNIGRAM, Unigram Language Model means the next word in the sentence is assumed to
                  be independent of the previous words generated by the model.
                - SentencePieceModel.BPE, refers to byte pair encoding algorithm, which replaces the most frequent pair
                  of bytes in a sentence with a single, unused byte.
                - SentencePieceModel.CHAR, refers to char based sentencePiece Model type.
                - SentencePieceModel.WORD, refers to word based sentencePiece Model type.

            params (dict): A dictionary with no incoming parameters.

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> from mindspore.dataset.text import SentencePieceVocab, SentencePieceModel
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> vocab = SentencePieceVocab.from_dataset(dataset, ["text"], 5000, 0.9995,
            ...                                         SentencePieceModel.UNIGRAM, {})
        """

        sentence_piece_vocab = cls()
        # pylint: disable=protected-access
        sentence_piece_vocab.c_sentence_piece_vocab = dataset._build_sentencepiece_vocab(col_names, vocab_size,
                                                                                         character_coverage,
                                                                                         model_type, params)
        return sentence_piece_vocab

    @classmethod
    @check_from_file_sentencepiece
    def from_file(cls, file_path, vocab_size, character_coverage, model_type, params):
        """
        Build a SentencePiece object from a file.

        Args:
            file_path (list): Path to the file which contains the SentencePiece list.
            vocab_size (int): Vocabulary size.
            character_coverage (float): Amount of characters covered by the model, good defaults are: 0.9995 for
                languages with rich character set like Japanese or Chinese and 1.0 for other languages with small
                character set.
            model_type (SentencePieceModel): It can be any of [SentencePieceModel.UNIGRAM, SentencePieceModel.BPE,
                SentencePieceModel.CHAR, SentencePieceModel.WORD], default is SentencePieceModel.UNIGRAM. The input
                sentence must be pre-tokenized when using SentencePieceModel.WORD type.

                - SentencePieceModel.UNIGRAM, Unigram Language Model means the next word in the sentence is assumed to
                  be independent of the previous words generated by the model.
                - SentencePieceModel.BPE, refers to byte pair encoding algorithm, which replaces the most frequent pair
                  of bytes in a sentence with a single, unused byte.
                - SentencePieceModel.CHAR, refers to char based sentencePiece Model type.
                - SentencePieceModel.WORD, refers to word based sentencePiece Model type.

            params (dict): A dictionary with no incoming parameters(The parameters are derived from SentencePiece
                library).

        Returns:
            SentencePieceVocab, vocab built from the file.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceVocab, SentencePieceModel
            >>> vocab = SentencePieceVocab.from_file(["/path/to/sentence/piece/vocab/file"], 5000, 0.9995,
            ...                                      SentencePieceModel.UNIGRAM, {})
        """

        sentence_piece_vocab = cls()
        sentence_piece_vocab.c_sentence_piece_vocab = cde.SentencePieceVocab.from_file(
            file_path, vocab_size, character_coverage, DE_C_INTER_SENTENCEPIECE_MODE.get(model_type), params)
        return sentence_piece_vocab

    @classmethod
    @check_save_model
    def save_model(cls, vocab, path, filename):
        """
        Save model into given filepath.

        Args:
            vocab (SentencePieceVocab): A SentencePiece object.
            path (str): Path to store model.
            filename (str): The name of the file.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceVocab, SentencePieceModel
            >>> vocab = SentencePieceVocab.from_file(["/path/to/sentence/piece/vocab/file"], 5000, 0.9995,
            ...                                      SentencePieceModel.UNIGRAM, {})
            >>> SentencePieceVocab.save_model(vocab, "./", "m.model")
        """

        cde.SentencePieceVocab.save_model(vocab.c_sentence_piece_vocab, path, filename)


class SPieceTokenizerLoadType(IntEnum):
    """
    An enumeration for loading type of :class:`mindspore.dataset.text.SentencePieceTokenizer` .

    Possible enumeration values are: SPieceTokenizerLoadType.FILE, SPieceTokenizerLoadType.MODEL.

    - SPieceTokenizerLoadType.FILE: Load SentencePiece tokenizer from a Vocab file.
    - SPieceTokenizerLoadType.MODEL: Load SentencePiece tokenizer from a SentencePieceVocab object.
    """

    FILE = 0
    MODEL = 1


class SPieceTokenizerOutType(IntEnum):
    """
    An enumeration for :class:`mindspore.dataset.text.SentencePieceTokenizer` .

    Possible enumeration values are: SPieceTokenizerOutType.STRING, SPieceTokenizerOutType.INT.

    - SPieceTokenizerOutType.STRING: means output type of SentencePiece Tokenizer is string.
    - SPieceTokenizerOutType.INT: means output type of SentencePiece Tokenizer is int.
    """

    STRING = 0
    INT = 1


class Vectors(cde.Vectors):
    """
    Vectors object that is used to map tokens into vectors.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Build a vector from a file.

        Args:
            file_path (str): Path of the file that contains the vectors.
            max_vectors (int, optional): This can be used to limit the number of pre-trained vectors loaded.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                passing `max_vectors` can limit the size of the loaded set. Default: None, no limit.

        Returns:
            Vectors, Vectors build from a file.

        Raises:
            RuntimeError: If `file_path` contains invalid data.
            ValueError: If `max_vectors` is invalid.
            TypeError: If `max_vectors` is not type of integer.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vector = text.Vectors.from_file("/path/to/vectors/file", max_vectors=None)
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class Vocab:
    """
    Vocab object that is used to save pairs of words and ids.

    It contains a map that maps each word(str) to an id(int) or reverse.
    """

    def __init__(self):
        self.c_vocab = None

    @classmethod
    @check_from_dataset
    def from_dataset(cls, dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True):
        """
        Build a Vocab from a dataset.

        This would collect all unique words in a dataset and return a vocab within
        the frequency range specified by user in freq_range. User would be warned if no words fall into the frequency.
        Words in vocab are ordered from the highest frequency to the lowest frequency. Words with the same frequency
        would be ordered lexicographically.

        Args:
            dataset (Dataset): dataset to build vocab from.
            columns (list[str], optional): column names to get words from. It can be a list of column names.
                Default: None.
            freq_range (tuple, optional): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency=0 is the same as
                min_frequency=1. max_frequency > total_words is the same as max_frequency = total_words.
                min_frequency/max_frequency can be None, which corresponds to 0/total_words separately.
                Default: None, all words are included.
            top_k (int, optional): top_k is greater than 0. Number of words to be built into vocab. top_k means most
                frequent words are taken. top_k is taken after freq_range. If not enough top_k, all words will be taken.
                Default: None, all words are included.
            special_tokens (list, optional):  A list of strings, each one is a special token. For example
                special_tokens=["<pad>","<unk>"]. Default: None, no special tokens will be added.
            special_first (bool, optional): Whether special_tokens will be prepended/appended to vocab. If
                special_tokens is specified and special_first is set to True, special_tokens will be prepended.
                Default: True.

        Returns:
            Vocab, Vocab object built from the dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.text as text
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> vocab = text.Vocab.from_dataset(dataset, "text", freq_range=None, top_k=None,
            ...                                 special_tokens=["<pad>", "<unk>"],
            ...                                 special_first=True)
            >>> dataset = dataset.map(operations=text.Lookup(vocab, "<unk>"), input_columns=["text"])
        """

        vocab = cls()
        # pylint: disable=protected-access
        vocab.c_vocab = dataset._build_vocab(columns, freq_range, top_k, special_tokens, special_first)
        return vocab

    @classmethod
    @check_from_list
    def from_list(cls, word_list, special_tokens=None, special_first=True):
        """
        Build a vocab object from a list of word.

        Args:
            word_list (list): A list of string where each element is a word of type string.
            special_tokens (list, optional):  A list of strings, each one is a special token. For example
                special_tokens=["<pad>","<unk>"]. Default: None, no special tokens will be added.
            special_first (bool, optional): Whether special_tokens is prepended or appended to vocab. If special_tokens
                is specified and special_first is set to True, special_tokens will be prepended. Default: True.

        Returns:
            Vocab, Vocab object built from the list.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
        """

        if special_tokens is None:
            special_tokens = []
        vocab = Vocab()
        vocab.c_vocab = cde.Vocab.from_list(word_list, special_tokens, special_first)
        return vocab

    @classmethod
    @check_from_file
    def from_file(cls, file_path, delimiter="", vocab_size=None, special_tokens=None, special_first=True):
        """
        Build a vocab object from a file.

        Args:
            file_path (str): Path to the file which contains the vocab list.
            delimiter (str, optional): A delimiter to break up each line in file, the first element is taken to be
                the word. Default: '', the whole line will be treated as a word.
            vocab_size (int, optional): Number of words to read from file_path. Default: None, all words are taken.
            special_tokens (list, optional):  A list of strings, each one is a special token. For example
                special_tokens=["<pad>","<unk>"]. Default: None, no special tokens will be added.
            special_first (bool, optional): Whether special_tokens will be prepended/appended to vocab,
                If special_tokens is specified and special_first is set to True,
                special_tokens will be prepended. Default: True.

        Returns:
            Vocab, Vocab object built from the file.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> # Assume vocab file contains the following content:
            >>> # --- begin of file ---
            >>> # apple,apple2
            >>> # banana, 333
            >>> # cat,00
            >>> # --- end of file ---
            >>>
            >>> # Read file through this API and specify "," as delimiter.
            >>> # The delimiter will break up each line in file, then the first element is taken to be the word.
            >>> vocab = text.Vocab.from_file("/path/to/simple/vocab/file", ",", None, ["<pad>", "<unk>"], True)
            >>>
            >>> # Finally, there are 5 words in the vocab: "<pad>", "<unk>", "apple", "banana", "cat".
            >>> vocabulary = vocab.vocab()
        """

        if vocab_size is None:
            vocab_size = -1
        if special_tokens is None:
            special_tokens = []
        vocab = cls()
        vocab.c_vocab = cde.Vocab.from_file(file_path, delimiter, vocab_size, special_tokens, special_first)
        return vocab

    @classmethod
    @check_from_dict
    def from_dict(cls, word_dict):
        """
        Build a vocab object from a dict.

        Args:
            word_dict (dict): Dict contains word and id pairs, where word should be str and id be int. id is recommended
                to start from 0 and be continuous. ValueError will be raised if id is negative.

        Returns:
            Vocab, Vocab object built from the dict.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
        """

        vocab = cls()
        vocab.c_vocab = cde.Vocab.from_dict(word_dict)
        return vocab

    def vocab(self):
        """
        Get the vocabory table in dict type.

        Returns:
            A vocabulary consisting of word and id pairs.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["word_1", "word_2", "word_3", "word_4"])
            >>> vocabory_dict = vocab.vocab()
        """
        check_vocab(self.c_vocab)
        return self.c_vocab.vocab()

    @check_tokens_to_ids
    def tokens_to_ids(self, tokens):
        """
        Converts a token string or a sequence of tokens in a single integer id or a sequence of ids.
        If token does not exist, return id with value -1.

        Args:
            tokens (Union[str, list[str]]): One or several token(s) to convert to token id(s).

        Returns:
            The token id or list of token ids.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> ids = vocab.tokens_to_ids(["w1", "w3"])
        """
        check_vocab(self.c_vocab)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if isinstance(tokens, str):
            tokens = [tokens]
        return self.c_vocab.tokens_to_ids(tokens)

    @check_ids_to_tokens
    def ids_to_tokens(self, ids):
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens.
        If id does not exist, return empty string.

        Args:
            ids (Union[int, list[int]]): The token id (or token ids) to convert to tokens.

        Returns:
            The decoded token(s).

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> token = vocab.ids_to_tokens(0)
        """
        check_vocab(self.c_vocab)
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.c_vocab.ids_to_tokens(ids)


def to_bytes(array, encoding='utf8'):
    """
    Convert NumPy array of `str` to array of `bytes` by encoding each element based on charset `encoding` .

    Args:
        array (numpy.ndarray): Array of `str` type representing strings.
        encoding (str): Indicating the charset for encoding. Default: 'utf8'.

    Returns:
        numpy.ndarray, NumPy array of `bytes` .

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>>
        >>> data = np.array([["1", "2", "3"]], dtype=np.str_)
        >>> dataset = ds.NumpySlicesDataset(data, column_names=["text"])
        >>> for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     bytes_data = text.to_bytes(item["text"])
    """

    if not isinstance(array, np.ndarray):
        raise ValueError('input should be a NumPy array.')

    return np.char.encode(array, encoding)


def to_str(array, encoding='utf8'):
    """
    Convert NumPy array of `bytes` to array of `str` by decoding each element based on charset `encoding` .

    Args:
        array (numpy.ndarray): Array of `bytes` type representing strings.
        encoding (str): Indicating the charset for decoding. Default: 'utf8'.

    Returns:
        numpy.ndarray, NumPy array of `str` .

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>>
        >>> data = np.array([["1", "2", "3"]], dtype=np.bytes_)
        >>> dataset = ds.NumpySlicesDataset(data, column_names=["text"])
        >>> for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     str_data = text.to_str(item["text"])
    """

    if not isinstance(array, np.ndarray):
        raise TypeError('input should be a NumPy array.')

    return np.char.decode(array, encoding)
