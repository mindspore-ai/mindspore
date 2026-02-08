# Copyright 2025 Huawei Technologies Co., Ltd
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
This module defines the `PyboostInternalFunctionsCppGenerator` class, which is used to generate the source file
(`functions.cc`) that contains function definitions for internal op in Pyboost.

The class uses templates and operation prototypes to create function definitions based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

from common import template
import common.gen_constants as K
from common.gen_utils import save_file
from common.base_generator import BaseGenerator

from .op_template_parser import OpTemplateParser


class PyboostInternalFunctionsCppGenerator(BaseGenerator):
    """
    A class to generate the `functions.cc` source file, which contains internal op function definitions.
    """

    def __init__(self):
        """Initializes the PyboostInternalFunctionsCppGenerator with the necessary templates."""
        self.pyboost_internal_functions_source_template = template.PYBOOST_INTERNAL_FUNCTION_SOURCE_TEMPLATE
        self.pyboost_internal_functions_template = template.PYBOOST_INTERNAL_FUNCTION_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates the Pyboost internal function source file (`functions.cc`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse and convert into function definitions.

        Returns:
            None: The method writes the generated source file to the specified directory.
        """
        func_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            operator_name = op_proto.op_name
            op_name = op_proto.op_class.name
            op_parser = OpTemplateParser(op_proto)
            call_args_after_convert, _, _ = op_parser.op_args_converter()
            call_args_with_type = op_parser.parse_call_args_with_types(is_convert=True)
            func_list.append(template.NEW_LINE + self.pyboost_internal_functions_template.replace(
                operator_name=operator_name,
                op_name=op_name,
                call_args_with_type=call_args_with_type,
                call_args_after_convert=call_args_after_convert))

        if not func_list:
            return
        pyboost_internal_op_functions_str = self.pyboost_internal_functions_source_template.replace(func_list=func_list)
        save_path = os.path.join(work_path, K.MS_PYBOOST_INTERNAL_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.cc"
        save_file(save_path, file_name, pyboost_internal_op_functions_str)
