# Copyright 2024 Huawei Technologies Co., Ltd
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
This module defines the `PyboostFunctionsHeaderGenerator` class, which is responsible for generating
the header file (`pyboost_functions.h`) for Pyboost function declarations.

The class uses templates and operation prototypes to create function declarations based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

from common import template
import common.gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.base_generator import BaseGenerator
from common.op_proto import OpProto

from .op_template_parser import OpTemplateParser
from .pyboost_utils import is_optional_param, get_input_args_type_str


class PyboostFunctionsHeaderGenerator(BaseGenerator):
    """
    A class to generate the `pyboost_functions.h` header file, which contains Pyboost function declarations.
    """

    def __init__(self):
        """Initializes the PyboostFunctionsHeaderGenerator with the necessary templates."""
        self.PYBOOST_FUNCTION_HEADER_TEMPLATE = template.PYBOOST_FUNCTION_HEADER_TEMPLATE

        self.PYBOOST_CORE_HEADER_TEMPLATE = template.PYBOOST_CORE_HEADER_TEMPLATE

        self.pyboost_func_template = Template(
            'PyObject* ${func_name}_Base(const PrimitivePtr &prim, PyObject* args);'
        )
        self.pyboost_op_func_template = Template(
            'PYNATIVE_EXPORT PyObject* ${func_name}_OP(const PrimitivePtr &prim, '
            'const std::vector<ops::OP_DTYPE>& source_type, ${input_args});'
        )
        self.input_args_template = Template(" const ${arg_type}& ${arg_name},")

    def generate(self, work_path, op_protos):
        """
        Generates the Pyboost function header file (`pyboost_functions.h`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse and convert into Pyboost function declarations.

        Returns:
            None: The method writes the generated header file to the specified directory.
        """
        prim_func_list = []
        op_func_list_str = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            op_parser = OpTemplateParser(op_proto)
            op_pyboost_func_name = op_parser.get_pyboost_func_name()
            op_input_args_str = self._get_input_args_str(op_proto)
            prim_func_list.append(self.pyboost_func_template.replace(func_name=op_pyboost_func_name))
            op_func_list_str.append(self.pyboost_op_func_template.replace(func_name=op_pyboost_func_name,
                                                                          input_args=op_input_args_str))
        pyboost_func_h_str = self.PYBOOST_FUNCTION_HEADER_TEMPLATE.replace(prim_func_list=prim_func_list)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_api.h"
        save_file(save_path, file_name, pyboost_func_h_str)

        # impl header
        pyboost_core_header_str = self.PYBOOST_CORE_HEADER_TEMPLATE.replace(op_func_list=op_func_list_str)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_HEADER_FUNC_GEN_PATH)
        file_name = "pyboost_core.h"
        save_file(save_path, file_name, pyboost_core_header_str)


    def _get_input_args_str(self, op_proto: OpProto) -> str:
        """
        Generates the input arguments list for the pyboost operator.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            str: The generated input arguments list as a string.
        """
        parser_func_str = ''
        for _, op_arg in enumerate(op_proto.op_args):
            is_optional = is_optional_param(op_arg)
            if op_arg.is_type_id:
                arg_type_str = get_input_args_type_str('type', is_optional, op_proto.op_view)
            else:
                arg_type_str = get_input_args_type_str(op_arg.arg_dtype, is_optional, op_proto.op_view)
            parser_func_str += self.input_args_template.replace(arg_name=op_arg.arg_name, arg_type=arg_type_str)
        return parser_func_str[:-1]
