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
    CharNGram pre-trained word embeddings.

    A word or sentence is represented using a character n-gram count vector, followed by a single
    nonlinear transformation to yield a low-dimensional embedding.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Load the CharNGram pre-training vector set file.

        Args:
            file_path (str): Path to the CharNGram pre-training vector set file.
            max_vectors (int, optional): The upper limit on the number of pre-trained vectors to load.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                this value can limit the size of the loaded set. Default: ``None``, no upper limit.

        Returns:
            CharNGram, CharNGram pre-training vectors.

        Raises:
            TypeError: If `file_path` is not of type str.
            RuntimeError: If `file_path` does not exist or is not accessible.
            TypeError: If `max_vectors` is not of type int.
            ValueError: If `max_vectors` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>>
            >>> char_n_gram = text.CharNGram.from_file("/path/to/char_n_gram/file", max_vectors=None)
            >>> to_vectors = text.ToVectors(char_n_gram)
            >>> # Look up a token into vectors according CharNGram model.
            >>> word_vector = to_vectors(["word1", "word2"])
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class FastText(cde.FastText):
    """
    FastText pre-trained word embeddings.

    FastText allows one to create an unsupervised learning or supervised learning algorithm vector
    representations for words.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Load the FastText pre-training vector set file.

        Args:
            file_path (str): Path to the FastText pre-trained vector set file. File suffix should be `*.vec`.
            max_vectors (int, optional): The upper limit on the number of pre-trained vectors to load.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                this value can limit the size of the loaded set. Default: ``None``, no upper limit.

        Returns:
            FastText, FastText pre-training vectors.

        Raises:
            TypeError: If `file_path` is not of type str.
            RuntimeError: If `file_path` does not exist or is not accessible.
            TypeError: If `max_vectors` is not of type int.
            ValueError: If `max_vectors` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> fast_text = text.FastText.from_file("/path/to/fast_text/file", max_vectors=None)
            >>> to_vectors = text.ToVectors(fast_text)
            >>> # Look up a token into vectors according FastText model.
            >>> word_vector = to_vectors(["word1", "word2"])
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class GloVe(cde.GloVe):
    """
    Global Vectors (GloVe) pre-trained word embeddings.

    GloVe is an unsupervised learning algorithm for obtaining vector representations for word.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Load the GloVe pre-training vector set file.

        Args:
            file_path (str): Path to the GloVe pre-training vector set file. File name is similar to `glove.*.txt`.
            max_vectors (int, optional): The upper limit on the number of pre-trained vectors to load.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                this value can limit the size of the loaded set. Default: ``None``, no upper limit.

        Returns:
            GloVe, GloVe pre-training vectors.

        Raises:
            TypeError: If `file_path` is not of type str.
            RuntimeError: If `file_path` does not exist or is not accessible.
            TypeError: If `max_vectors` is not of type int.
            ValueError: If `max_vectors` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> glove = text.GloVe.from_file("/path/to/glove/file", max_vectors=None)
            >>> to_vectors = text.ToVectors(glove)
            >>> # Look up a token into vectors according GloVe model.
            >>> word_vector = to_vectors(["word1", "word2"])
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class JiebaMode(IntEnum):
    """
    An enumeration for :class:`mindspore.dataset.text.JiebaTokenizer` .

    Possible enumeration values are: ``JiebaMode.MIX``, ``JiebaMode.MP``, ``JiebaMode.HMM``.

    - JiebaMode.MIX: tokenize with a mix of MPSegment and HMMSegment algorithm.
    - JiebaMode.MP: tokenize with MPSegment algorithm.
    - JiebaMode.HMM: tokenize with Hidden Markov Model Segment algorithm.
    """

    MIX = 0
    MP = 1
    HMM = 2


class NormalizeForm(IntEnum):
    """
    `Unicode normalization forms <http://unicode.org/reports/tr15/>`_ .

    Available values are as follows:

    - NormalizeForm.NONE: No normalization.
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
    Subword algorithms for SentencePiece.

    Available values are as follows:

    - SentencePieceModel.UNIGRAM: `Unigram Language Model <https://arxiv.org/abs/1804.10959>`_ subword algorithm.
    - SentencePieceModel.BPE: `Byte-Pair-Encoding <https://arxiv.org/abs/1508.07909>`_ subword algorithm.
    - SentencePieceModel.CHAR: Character-based subword algorithm.
    - SentencePieceModel.WORD: Word-based subword algorithm.
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
            character_coverage (float): Amount of characters covered by the model. Recommend ``0.9995`` for
                languages with rich character set like Japanese or Chinese and ``1.0`` for other languages with small
                character set.
            model_type (SentencePieceModel): The desired subword algorithm. See :class:`~.text.SentencePieceModel`
                for details on optional values.
            params (dict): A dictionary with no incoming parameters.

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.text as text
            >>>
            >>> from mindspore.dataset.text import SentencePieceVocab, SentencePieceModel
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> vocab = SentencePieceVocab.from_dataset(dataset, ["text"], 5000, 0.9995,
            ...                                         SentencePieceModel.UNIGRAM, {})
            >>> # Build tokenizer based on vocab
            >>> tokenizer = text.SentencePieceTokenizer(vocab, out_type=text.SPieceTokenizerOutType.STRING)
            >>> txt = "Today is Tuesday."
            >>> token = tokenizer(txt)
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
            character_coverage (float): Amount of characters covered by the model. Recommend ``0.9995`` for
                languages with rich character set like Japanese or Chinese and ``1.0`` for other languages with small
                character set.
            model_type (SentencePieceModel): The desired subword algorithm. See :class:`~.text.SentencePieceModel`
                for details on optional values.
            params (dict): A dictionary with no incoming parameters(The parameters are derived from SentencePiece
                library).

        Returns:
            SentencePieceVocab, vocab built from the file.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceVocab, SentencePieceModel
            >>> vocab = SentencePieceVocab.from_file(["/path/to/sentence/piece/vocab/file"], 5000, 0.9995,
            ...                                      SentencePieceModel.UNIGRAM, {})
            >>> # Build tokenizer based on vocab model
            >>> tokenizer = text.SentencePieceTokenizer(vocab, out_type=text.SPieceTokenizerOutType.STRING)
            >>> txt = "Today is Friday."
            >>> token = tokenizer(txt)
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
    Model input type for the SentencePiece tokenizer.

    Available values are as follows:

    - SPieceTokenizerLoadType.FILE: Load model from specified file path.
    - SPieceTokenizerLoadType.MODEL: Load model from specified vocab object.
    """

    FILE = 0
    MODEL = 1


class SPieceTokenizerOutType(IntEnum):
    """
    An enumeration for :class:`mindspore.dataset.text.SentencePieceTokenizer` .

    Possible enumeration values are: ``SPieceTokenizerOutType.STRING``, ``SPieceTokenizerOutType.INT``.

    - SPieceTokenizerOutType.STRING: means output type of SentencePiece Tokenizer is string.
    - SPieceTokenizerOutType.INT: means output type of SentencePiece Tokenizer is int.
    """

    STRING = 0
    INT = 1


class Vectors(cde.Vectors):
    """
    Pre-trained word embeddings.
    """

    @classmethod
    @check_from_file_vectors
    def from_file(cls, file_path, max_vectors=None):
        """
        Load a pre-training vector set file.

        Args:
            file_path (str): Path to the pre-training vector set file.
            max_vectors (int, optional): The upper limit on the number of pre-trained vectors to load.
                Most pre-trained vector sets are sorted in the descending order of word frequency. Thus, in
                situations where the entire set doesn't fit in memory, or is not needed for another reason,
                this value can limit the size of the loaded set. Default: ``None``, no upper limit.

        Returns:
            Vectors, pre-training vectors.

        Raises:
            TypeError: If `file_path` is not of type str.
            RuntimeError: If `file_path` does not exist or is not accessible.
            TypeError: If `max_vectors` is not of type int.
            ValueError: If `max_vectors` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vector = text.Vectors.from_file("/path/to/vectors/file", max_vectors=None)
            >>> to_vectors = text.ToVectors(vector)
            >>> # Look up a token into vectors according Vector model.
            >>> word_vector = to_vectors(["word1", "word2"])
        """

        max_vectors = max_vectors if max_vectors is not None else 0
        return super().from_file(file_path, max_vectors)


class Vocab:
    """
    Create Vocab for training NLP models.

    Vocab is a collection of all possible Tokens in the data, preserving the mapping between each Token and its ID.
    """

    def __init__(self):
        self.c_vocab = None

    @classmethod
    @check_from_dataset
    def from_dataset(cls, dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True):
        """
        Build a Vocab from a given dataset.

        The samples in the dataset are used as a corpus to create Vocab, in which the Token is arranged in ascending
        order of Token frequency, and Tokens with the same frequency are arranged in alphabetical order.

        Args:
            dataset (Dataset): The dataset to build the Vocab from.
            columns (list[str], optional): The name of the data columns used to create the Vocab.
                Default: ``None`` , use all columns.
            freq_range (tuple[int, int], optional): The Token frequency range used to create the Vocab. Must contain
                two elements representing the minimum and maximum frequencies, within which the Token will be retained.
                When the minimum or maximum frequency is None, it means there is no minimum or maximum frequency limit.
                Default: ``None`` , no Token frequency range restriction.
            top_k (int, optional): Only the first specified number of Tokens with the highest Token frequency are
                selected to build the Vocab. This operation will be performed after Token frequency filtering. If
                the value is greater than the total number of Tokens, all Tokens will be retained. Default: ``None`` ,
                there is no limit to the number of Tokens.
            special_tokens (list[str], optional):  A list of special Token to append to the Vocab. Default: ``None`` ,
                no special Token is appended.
            special_first (bool, optional): Whether to add the special Token to the top of the Vocab, otherwise to
                the bottom of the Vocab. Default: ``True``.

        Returns:
            Vocab, Vocab built from the dataset.

        Raises:
            TypeError: If `columns` is not of type list[str].
            TypeError: If `freq_range` is not of type tuple[int, int]l.
            ValueError: If element of `freq_range` is negative.
            TypeError: If `top_k` is not of type int.
            ValueError: If `top_k` is not positive.
            TypeError: If `special_tokens` is not of type list[str].
            ValueError: If there are duplicate elements in `special_tokens`.
            TypeError: If `special_first` is not of type bool.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.text as text
            >>>
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> vocab = text.Vocab.from_dataset(dataset, "text", freq_range=None, top_k=None,
            ...                                 special_tokens=["<pad>", "<unk>"],
            ...                                 special_first=True)
            >>> # Use the vocab to look up string to id
            >>> lookup = text.Lookup(vocab, "<unk>")
            >>> id = lookup("text1")
        """

        vocab = cls()
        # pylint: disable=protected-access
        vocab.c_vocab = dataset._build_vocab(columns, freq_range, top_k, special_tokens, special_first)
        return vocab

    @classmethod
    @check_from_list
    def from_list(cls, word_list, special_tokens=None, special_first=True):
        """
        Build a Vocab from a given Token list.

        Args:
            word_list (list[str]): The Token list to build the Vocab from.
            special_tokens (list[str], optional):  A list of special Token to append to the Vocab. Default: ``None`` ,
                no special Token is appended.
            special_first (bool, optional): Whether to add the special Token to the top of the Vocab, otherwise to
                the bottom of the Vocab. Default: ``True``.

        Returns:
            Vocab, Vocab built from the list.

        Raises:
            TypeError: If `word_list` is not of type list[str].
            ValueError: If there are duplicate elements in `word_list`.
            TypeError: If `special_tokens` is not of type list[str].
            ValueError: If there are duplicate elements in `special_tokens`.
            TypeError: If `special_first` is not of type bool.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> # look up strings to ids
            >>> ids = vocab.tokens_to_ids(["w1", "w3"])
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
        Build a Vocab from a file.

        Args:
            file_path (str): The path of the file to build the Vocab from.
            delimiter (str, optional): The separator for the Token in the file line. The string before the separator
                will be treated as a Token. Default: ``''``, the whole line will be treated as a Token.
            vocab_size (int, optional): The upper limit on the number of Tokens that Vocab can contain.
                Default: ``None`` , no upper limit on the number of Token.
            special_tokens (list[str], optional):  A list of special Token to append to the Vocab. Default: ``None`` ,
                no special Token is appended.
            special_first (bool, optional): Whether to add the special Token to the top of the Vocab, otherwise to
                the bottom of the Vocab. Default: ``True``.

        Returns:
            Vocab, Vocab built from the file.

        Raises:
            TypeError: If `file_path` is not of type str.
            TypeError: If `delimiter` is not of type str.
            ValueError: If `vocab_size` is not positive.
            TypeError: If `special_tokens` is not of type list[str].
            ValueError: If there are duplicate elements in `special_tokens`.
            TypeError: If `special_first` is not of type bool.

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
            >>>
            >>> # look up strings to ids
            >>> ids = vocab.tokens_to_ids(["apple", "banana"])
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
        Build a Vocab from a given dictionary.

        Args:
            word_dict (dict[str, int]): A dictionary storing the mappings between each Token and its ID.

        Returns:
            Vocab, Vocab built from the dictionary.

        Raises:
            TypeError: If `word_dict` is not of type dict[str, int].
            ValueError: If key value of `word_dict` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
            >>>
            >>> # look up ids to string
            >>> tokens = vocab.ids_to_tokens([3, 4, 5])
            >>> print(tokens)
            ['home', 'the', 'world']
        """

        vocab = cls()
        vocab.c_vocab = cde.Vocab.from_dict(word_dict)
        return vocab

    def vocab(self):
        """
        Get the dictionary of the mappings between Tokens and its IDs.

        Returns:
            dict[str, int], the dictionary of mappings between Tokens and IDs.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["word_1", "word_2", "word_3", "word_4"])
            >>> vocabory_dict = vocab.vocab()
            >>> print(sorted(vocabory_dict.items()))
            [('word_1', 0), ('word_2', 1), ('word_3', 2), ('word_4', 3)]
        """
        check_vocab(self.c_vocab)
        return self.c_vocab.vocab()

    @check_tokens_to_ids
    def tokens_to_ids(self, tokens):
        """
        Look up the ID corresponding to the specified Token.

        Args:
            tokens (Union[str, list[str], numpy.ndarray]): The Token or list of Tokens to be looked up.
                If the Token does not exist, -1 is returned.

        Returns:
            Union[int, list[int]], the ID(s) corresponding to the Token(s).

        Raises:
            TypeError: If `tokens` is not of type Union[str, list[str], numpy.ndarray].

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> ids = vocab.tokens_to_ids(["w1", "w3"])
            >>> print(ids)
            [1, 3]
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
        Look up the Token corresponding to the specified ID.

        Args:
            ids (Union[int, list[int], numpy.ndarray]): The ID or list of IDs to be looked up.
                If the ID does not exist, an empty string is returned.

        Returns:
            Union[str, list[str]], the Token(s) corresponding to the ID(s).

        Raises:
            TypeError: If `ids` is not of type Union[int, list[int], numpy.ndarray].
            ValueError: If element of `ids` is negative.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> token = vocab.ids_to_tokens(1)
            >>> print(token)
            w1
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
        encoding (str): Indicating the charset for encoding. Default: ``'utf8'``.

    Returns:
        numpy.ndarray, NumPy array of `bytes` .

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.text as text
        >>>
        >>> data = np.array([["1", "2", "3"]], dtype=np.str_)
        >>> dataset = ds.NumpySlicesDataset(data, column_names=["text"])
        >>> result = []
        >>> for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     result.append(text.to_bytes(item["text"]))
        >>> print(result)
        [array([b'1', b'2', b'3'], dtype='|S1')]
    """

    if not isinstance(array, np.ndarray):
        raise ValueError('input should be a NumPy array.')

    return np.char.encode(array, encoding)


def to_str(array, encoding='utf8'):
    """
    Convert NumPy array of `bytes` to array of `str` by decoding each element based on charset `encoding` .

    Args:
        array (numpy.ndarray): Array of `bytes` type representing strings.
        encoding (str): Indicating the charset for decoding. Default: ``'utf8'``.

    Returns:
        numpy.ndarray, NumPy array of `str` .

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.text as text
        >>>
        >>> data = np.array([["1", "2", "3"]], dtype=np.bytes_)
        >>> dataset = ds.NumpySlicesDataset(data, column_names=["text"])
        >>> result = []
        >>> for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     result.append(text.to_str(item["text"]))
        >>> print(result)
        [array(['1', '2', '3'], dtype='<U1')]
    """

    if not isinstance(array, np.ndarray):
        raise TypeError('input should be a NumPy array.')

    return np.char.decode(array, encoding)
