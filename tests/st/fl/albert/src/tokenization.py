# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open


logger = logging.getLogger(__name__)


def load_vocab(vocab_file, vocab_map_ids_path=None):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    if vocab_map_ids_path is not None:
        vocab_new_ids = list()
        with open(vocab_map_ids_path, "r", encoding="utf-8") as vocab_new_ids_reader:
            while True:
                index = vocab_new_ids_reader.readline()
                if not index:
                    break
                index = index.strip()
                vocab_new_ids.append(int(index))
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = vocab_new_ids[index]
                index += 1
        return vocab
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer:
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True, basic_only=False,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
        self.basic_only = basic_only

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                if self.basic_only:
                    split_tokens.append(token)
                else:
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab['[UNK]']))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, 'vocab.txt')
        else:
            raise FileNotFoundError
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """

        if 'txt' in pretrained_model_name_or_path:
            resolved_vocab_file = pretrained_model_name_or_path
        else:
            resolved_vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.txt')

        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)

        return tokenizer


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @staticmethod
    def _run_strip_accents(text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)
        ):
            return True

        return False

    @staticmethod
    def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class CustomizedBasicTokenizer(BasicTokenizer):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"), keywords=None):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        super().__init__(do_lower_case, never_split)
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.keywords = keywords

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)

        if self.keywords is not None:
            new_orig_tokens = []
            lengths = [len(_) for _ in self.keywords]
            max_length = max(lengths)
            orig_tokens_len = len(orig_tokens)
            i = 0
            while i < orig_tokens_len:
                has_add = False
                for length in range(max_length, 0, -1):
                    if i + length > orig_tokens_len:
                        continue
                    add_token = ''.join(orig_tokens[i:i+length])
                    if add_token in self.keywords:
                        new_orig_tokens.append(add_token)
                        i += length
                        has_add = True
                        break
                if not has_add:
                    new_orig_tokens.append(orig_tokens[i])
                    i += 1
        else:
            new_orig_tokens = orig_tokens

        split_tokens = []
        for token in new_orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class CustomizedTokenizer(BertTokenizer):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"), keywords=None):
        """Constructs a CustomizedTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        super().__init__(vocab_file, do_lower_case, max_len, never_split)

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = CustomizedBasicTokenizer(do_lower_case=do_lower_case, never_split=never_split,
                                                        keywords=keywords)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        basic_tokens = self.basic_tokenizer.tokenize(text)
        for token in basic_tokens:
            wordpiece_tokens = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in wordpiece_tokens:
                split_tokens.append(sub_token)
        return split_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        resolved_vocab_file = os.path.join(pretrained_model_name_or_path, 'customized_vocab.txt')

        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)

        return tokenizer


class CustomizedTextBasicTokenizer(BasicTokenizer):
    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp) or len(char.encode('utf-8')) > 1:
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class CustomizedTextTokenizer(BertTokenizer):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),
                 vocab_map_ids_path=None):
        """Constructs a CustomizedTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        super().__init__(vocab_file, do_lower_case, max_len, never_split)

        self.vocab = load_vocab(vocab_file, vocab_map_ids_path)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = CustomizedTextBasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
