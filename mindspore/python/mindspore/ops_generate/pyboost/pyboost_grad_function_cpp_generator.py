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
This module defines the PyboostGradFunctionsGenerator class, which is responsible for generating PyBoost gradient
functions and saving the corresponding C++ source files. The generator parses operator prototypes and constructs
function definitions, includes necessary headers, and generates the registration code for the PyBoost functions.
"""


import os

from common import template
from common.template import Template
from common.gen_utils import save_file
import common.gen_constants as K
from common.op_proto import OpProto
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils

from .op_template_parser import OpTemplateParser


class PyboostGradFunctionsGenerator(BaseGenerator):
    """
    PyboostGradFunctionsGenerator generates PyBoost gradient functions based on operator prototypes (instances of
    OpProto).

    This class processes operator prototypes (`op_protos`) to generate PyBoost functions. It defines the function
    body, handles value conversion, creates contiguous tensor values, and generates the necessary header includes and
    registration code. The generated content is then saved to a specified location in the file system.
    """

    def __init__(self):
        super().__init__()
        self.GEN_OPS_DEF_HEADER_TEMPLATE = template.GEN_OPS_DEF_HEADER_TEMPLATE
        self.contiguous_template = Template(
            "convert_$arg_name = runtime::ValueConverter::ContiguousTensorValue($device_target, convert_$arg_name);\n")
        self.PYBOOST_GRAD_FUNCTION_TEMPLATE = template.PYBOOST_GRAD_FUNCTION_TEMPLATE
        self.PYBOOST_VIEW_GRAD_FUNCTION_TEMPLATE = template.PYBOOST_VIEW_GRAD_FUNCTION_TEMPLATE
        self.composite_include_header_template = template.COMPOSITE_INCLUDE_HEADER_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates the PyBoost gradient functions and writes them to the appropriate files.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information such as
        operator names, arguments, and conversion types. It uses this data to construct function bodies, includes,
        and registration code. The generated content is saved to a specified path as a C++ source file.

        Args:
            work_path (str): The file path where the generated files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.

        Returns:
            None
        """
        pyboost_func_str = ''
        pyboost_func_reg_def = ''
        composite_headers_str = ''
        for op_proto in op_protos:
            if (op_proto.op_dispatch is None) or (not op_proto.op_dispatch.enable):
                continue
            op_parser = OpTemplateParser(op_proto)
            op_pyboost_func_name = op_parser.get_pyboost_func_name()
            op_name_str = op_proto.op_class.name
            op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
            convert_value_type_str = self._convert_value_type(op_proto)
            device_target = "op_runner_info->device_target"
            convert_value_type_str += self._contiguous_tensor_value(op_proto, device_target)

            call_args_str = []
            for op_arg in op_proto.op_args:
                call_arg = 'convert_' + op_arg.arg_name
                call_args_str.append(call_arg)
            pyboost_grad_function_template = self._get_pyboost_grad_function_template(op_proto)
            pyboost_func_str += pyboost_grad_function_template.replace(
                func_name=op_pyboost_func_name,
                op_name=op_name_str,
                op_args=op_args_str,
                convert_body=convert_value_type_str,
                call_args=call_args_str,
                operator_name=op_proto.op_name)
            pyboost_func_str = pyboost_func_str + template.NEW_LINE
            pyboost_func_reg_def += template.REGISTER_PYBOOST_GRAD_DEFINE_TEMPLATE.replace(
                pyboost_op_name=op_proto.op_class.name,
                pyboost_cfunc_name=op_pyboost_func_name)

            if op_proto.composite:
                composite_headers_str +=self.composite_include_header_template.replace(
                    operator_name=op_proto.op_name
                )

        register_func_str = template.REGISTER_PYBOOST_GRAD_TEMPLATE.replace(register_func=pyboost_func_reg_def)
        pyboost_func_file = \
            template.PYBOOST_GRAD_HEADER_TEMPLATE.replace(composite_headers=composite_headers_str,
                                                          function_body=pyboost_func_str,
                                                          register_function_body=register_func_str)
        save_path = os.path.join(work_path, K.PYBOOST_GRAD_FUNC_GEN_PATH)
        file_name = "pyboost_grad_functions.cc"
        save_file(save_path, file_name, pyboost_func_file)

    def _get_pyboost_grad_function_template(self, op_proto: OpProto):
        if op_proto.op_view:
            return self.PYBOOST_VIEW_GRAD_FUNCTION_TEMPLATE
        return self.PYBOOST_GRAD_FUNCTION_TEMPLATE

    def _convert_value_type(self, op_proto: OpProto) -> str:
        """
        Generates the code for converting the operator's input values to the required types.

        This method iterates over the operator's arguments, checks if they are optional, and generates the appropriate
        conversion code based on the argument's data type.

        Args:
            op_proto (OpProto): The operator prototype containing information about the operator's arguments.

        Returns:
            str: A string containing the code for converting the input values to the required types.
        """
        convert_template = Template(
            "auto convert_$arg_name = ValueConverter::${convert_func}(op_runner_info->inputs[$arg_index]);\n")
        parser_func_str = ''
        for index, arg in enumerate(op_proto.op_args):
            is_optional = pyboost_utils.is_optional_param(arg)
            convert_type_str = pyboost_utils.get_value_convert_type_str(arg.arg_dtype, is_optional, op_proto.op_view)
            parser_func_str += convert_template.replace(arg_name=arg.arg_name, convert_func=convert_type_str,
                                                        arg_index=pyboost_utils.get_index(index))
        return parser_func_str

    def _contiguous_tensor_value(self, op_proto: OpProto, device_target: str) -> str:
        """
        Generates the code for converting tensors to contiguous format if required.

        This method checks the data type of the operator's arguments and generates code for converting tensors
        to contiguous format, which is necessary for certain types of tensors.

        Args:
            op_proto (OpProto): The operator prototype containing information about the operator's arguments.
            device_target (str): The device target string used in the conversion code.

        Returns:
            str: A string containing the code for converting tensors to contiguous format.
            If the operator is a view operation, an empty string is returned.
        """
        if op_proto.op_view:
            return ''
        contiguous_func_str = ''
        need_contiguous_dtype = {'tensor', 'tuple[tensor]'}
        for arg in op_proto.op_args:
            if arg.arg_dtype not in need_contiguous_dtype:
                continue
            contiguous_func_str += self.contiguous_template.replace(arg_name=arg.arg_name, device_target=device_target)
        return contiguous_func_str
