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
This module defines the `PyboostInternalFunctionsHeaderGenerator` class, which is used to generate the header file
(`functions.h`) that contains function declarations for internal op in Pyboost.

The class uses templates and operation prototypes to create function declarations based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

from common import template
import common.gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.base_generator import BaseGenerator

from .op_template_parser import OpTemplateParser


class PyboostInternalFunctionsHeaderGenerator(BaseGenerator):
    """
    A class to generate the `functions.h` header file, which contains internal op function declarations.
    """

    def __init__(self):
        """Initializes the PyboostInternalFunctionsHeaderGenerator with the necessary templates."""
        self.pyboost_internal_function_header_template = template.PYBOOST_INTERNAL_FUNCTION_HEADER_TEMPLATE

        self.pyboost_internal_func_template = Template(
            'void internal_${operator_name}(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type});'
        )

    def generate(self, work_path, op_protos):
        """
        Generates the Pyboost internal function header file (`functions.h`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse and convert into function declarations.

        Returns:
            None: The method writes the generated header file to the specified directory.
        """
        func_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            operator_name = op_proto.op_name
            op_parser = OpTemplateParser(op_proto)
            call_args_with_types = op_parser.parse_call_args_with_types(is_convert=True)
            func_list.append(self.pyboost_internal_func_template.replace(operator_name=operator_name,
                                                                         call_args_with_type=call_args_with_types))

        if not func_list:
            return
        pyboost_internal_func_h_str = \
            self.pyboost_internal_function_header_template.replace(internal_func_list=func_list)
        save_path = os.path.join(work_path, K.MS_PYBOOST_INTERNAL_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.h"
        save_file(save_path, file_name, pyboost_internal_func_h_str)
