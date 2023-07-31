# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Parser register."""
from typing import Optional

from .parser import Parser


class ParserRegister:
    """Parser register."""

    def __init__(self):
        self._parsers: dict = {}

    @classmethod
    def instance(cls) -> 'ParserRegister':
        """
        Get singleton of ParserRegister.

        Returns:
            An instance of ParserRegister.
        """
        if not hasattr(ParserRegister, "_instance"):
            ParserRegister._instance = ParserRegister()
        return ParserRegister._instance

    @staticmethod
    def reg_parser(parser: Parser):
        """
        Register a 'parser' to current ParserRegister.

        Args:
            parser (Parser): An instance of Parser to be registered.
        """
        if isinstance(parser, Parser):
            ParserRegister.instance().get_parsers()[parser.target()] = parser

    def get_parser(self, ast_type: type) -> Optional[Parser]:
        """
        Get parser from current ParserRegister by type of ast.

        Args:
            ast_type (type): An type of ast which want to be parsed.

        Returns:
            An instance of Parser if there existing suitable parser in current ParserRegister else None.
        """
        return self._parsers.get(ast_type)

    def get_parsers(self) -> [Parser]:
        """
        Get all parsers registered in current ParserRegister.

        Returns:
            An list of instances of Parser.
        """
        return self._parsers


class ParserRegistry:
    """Parser registry."""

    def __init__(self, parser: Parser):
        ParserRegister.instance().reg_parser(parser)


def reg_parser(parser: Parser):
    """
    A global method for registering parser into ParserRegister singleton.

    Args:
        parser (Parser): An instance of Parser to be registered.
    """
    return ParserRegistry(parser)
