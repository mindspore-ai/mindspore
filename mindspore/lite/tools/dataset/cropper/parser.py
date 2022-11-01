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
""" extract ops from user code """

from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os

ASSOCIATION_FILENAME = 'associations.txt'


@lru_cache(maxsize=1)
def _load_ops_names():
    """
    Get the name of all ops available in MindData lite.

    :return: a list of all available ops in MindData lite
    """
    with open(os.path.expanduser(ASSOCIATION_FILENAME), 'r') as f:
        _dict = json.load(f)
    return _dict.keys()


class Parser(ABC):
    """
    Abstract Base Class for parsers for looking up ops in user code.
    """

    def __init__(self):
        self._all_ops = _load_ops_names()

    @abstractmethod
    def parse(self, user_filename):
        """
        finds ops detected in the user code

        :param user_filename: string, name of file containing user code
        :return: list of ops found in the user code
        """


class SimpleParser(Parser):
    """
    A simple parser that works by string matching:
    Code uses an op if it is found anywhere in the text.
    """

    def parse(self, user_filename):
        """
        Find and return ops in the user file.

        :param user_filename: filename of user code
        :return: a list of ops present in the file
        """
        user_filename = os.path.realpath(user_filename)
        if not os.path.isfile(user_filename):
            raise FileNotFoundError("file does not exist: {}".format(user_filename))
        with open(user_filename) as f:
            data = f.read().strip()
        user_ops = self._simple_string_match(data)
        return user_ops

    def _simple_string_match(self, user_text):
        """
        Find and return ops in the user code (provided as a string).

        :param user_text: string containing user code
        :return: a list of ops found in the user_text
        """
        processed_user_text = user_text.strip().lower()
        user_ops = [op for op in self._all_ops if op in processed_user_text]
        return user_ops
