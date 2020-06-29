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
# ============================================================================
"""Tokenization utilities."""

import sys
import collections
import unicodedata

def convert_to_printable(text):
    """
    Converts `text` to a printable coding format.
    """
    if sys.version_info[0] == 3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        raise ValueError("Only support type `str` or `bytes`, while text type is `%s`" % (type(text)))
    if sys.version_info[0] == 2:
        if isinstance(text, str):
            return text
        if isinstance(text, unicode):
            return text.encode("utf-8")
        raise ValueError("Only support type `str` or `unicode`, while text type is `%s`" % (type(text)))
    raise ValueError("Only supported when running on Python2 or Python3.")


def convert_to_unicode(text):
    """
    Converts `text` to Unicode format.
    """
    if sys.version_info[0] == 3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        raise ValueError("Only support type `str` or `bytes`, while text type is `%s`" % (type(text)))
    if sys.version_info[0] == 2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        if isinstance(text, unicode):
            return text
        raise ValueError("Only support type `str` or `unicode`, while text type is `%s`" % (type(text)))
    raise ValueError("Only supported when running on Python2 or Python3.")


def load_vocab_file(vocab_file):
    """
    Loads a vocabulary file and turns into a {token:id} dictionary.
    """
    vocab_dict = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as vocab:
        while True:
            token = convert_to_unicode(vocab.readline())
            if not token:
                break
            token = token.strip()
            vocab_dict[token] = index
            index += 1
    return vocab_dict


def convert_by_vocab_dict(vocab_dict, items):
    """
    Converts a sequence of [tokens|ids] according to the vocab dict.
    """
    output = []
    for item in items:
        if item in vocab_dict:
            output.append(vocab_dict[item])
        else:
            output.append(vocab_dict["<unk>"])
    return output


class WhiteSpaceTokenizer():
    """
    Whitespace tokenizer.
    """
    def __init__(self, vocab_file):
        self.vocab_dict = load_vocab_file(vocab_file)
        self.inv_vocab_dict = {index: token for token, index in self.vocab_dict.items()}

    def _is_whitespace_char(self, char):
        """
        Checks if it is a whitespace character(regard "\t", "\n", "\r" as whitespace here).
        """
        if char in (" ", "\t", "\n", "\r"):
            return True
        uni = unicodedata.category(char)
        if uni == "Zs":
            return True
        return False

    def _is_control_char(self, char):
        """
        Checks if it is a control character.
        """
        if char in ("\t", "\n", "\r"):
            return False
        uni = unicodedata.category(char)
        if uni in ("Cc", "Cf"):
            return True
        return False

    def _clean_text(self, text):
        """
        Remove invalid characters and cleanup whitespace.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control_char(char):
                continue
            if self._is_whitespace_char(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        """
        Clean whitespace and split text into tokens.
        """
        text = text.strip()
        if not text:
            tokens = []
        else:
            tokens = text.split()
        return tokens

    def tokenize(self, text):
        """
        Tokenizes text.
        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        tokens = self._whitespace_tokenize(text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab_dict(self.vocab_dict, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab_dict(self.inv_vocab_dict, ids)
