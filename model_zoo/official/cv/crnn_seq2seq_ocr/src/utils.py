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
"""Util class or function."""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import codecs
import logging


def initialize_vocabulary(vocabulary_path):
    """
    initialize vocabulary from file.
    assume the vocabulary is stored one-item-per-line
    """
    characters_class = 9999

    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs.open(vocabulary_path, 'r', encoding='utf-8') as voc_file:
            rev_vocab = [line.strip() for line in voc_file]

        vocab = {x: y for (y, x) in enumerate(rev_vocab)}

        reserved_char_size = characters_class - len(rev_vocab)
        if reserved_char_size < 0:
            raise ValueError("Number of characters in vocabulary is equal or larger than config.characters_class")

        for _ in range(reserved_char_size):
            rev_vocab.append('')

        # put space at the last position
        vocab[' '] = len(rev_vocab)
        rev_vocab.append(' ')
        logging.info("Initializing vocabulary ends: %s", vocabulary_path)
        return vocab, rev_vocab

    raise ValueError("Initializing vocabulary ends: %s" % vocabulary_path)
