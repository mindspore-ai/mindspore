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
"""
Parsers for resolve ast to SymbolTree
"""
from .parser import Parser
from .parser_register import ParserRegister, ParserRegistry, reg_parser
from .module_parser import ModuleParser
from .arguments_parser import ArgumentsParser
from .assign_parser import AssignParser
from .for_parser import ForParser
from .function_def_parser import FunctionDefParser
from .if_parser import IfParser
from .return_parser import ReturnParser
from .class_def_parser import ClassDefParser
from .while_parser import WhileParser
from .expr_parser import ExprParser # Rely on AssignParser
