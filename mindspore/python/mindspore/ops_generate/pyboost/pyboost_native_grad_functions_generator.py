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
This module defines the PyboostGradFunctionsCppGenerator and PyboostGradFunctionsHeaderGenerator classes,
which are responsible for generating C++ gradient function implementations and headers for PyBoost operations.

The PyboostGradFunctionsCppGenerator generates the actual function definitions for the gradient functions,
while the PyboostGradFunctionsHeaderGenerator creates the corresponding function declarations in header files.
"""

import os

from pyboost import pyboost_utils
from pyboost.pyboost_utils import is_optional_param
from common import template
from common.template import Template
import common.gen_constants as K
from common.gen_utils import save_file
from common.op_proto import OpProto
from common.base_generator import BaseGenerator


class PyboostGradFunctionsCppGenerator(BaseGenerator):
    """
    PyboostGradFunctionsCppGenerator generates C++ implementations for PyBoost gradient functions.

    This class processes operator prototypes (`op_protos`) to create C++ function definitions that
    wrap PyBoost gradient functionality. It constructs function bodies, handles value conversion,
    and generates necessary include statements for each operator.

    Attributes:
       PYBOOST_NATIVE_GRAD_FUNCTION_TEMPLATE (Template): Template for generating native gradient function definitions.
       PYBOOST_NATIVE_GRAD_FUNCTIONS_TEMPLATE (Template): Template for generating the overall gradient functions file.
       native_function_multi_output_template (Template): Template for handling multiple output functions.
       native_function_single_output_template (str): Template for handling single output functions.
       convert_template (Template): Template for converting argument values to native types.
    """

    def __init__(self):
        self.PYBOOST_NATIVE_GRAD_FUNCTION_TEMPLATE = template.PYBOOST_NATIVE_GRAD_FUNCTION_TEMPLATE
        self.PYBOOST_NATIVE_VIEW_GRAD_FUNCTION_TEMPLATE = template.PYBOOST_NATIVE_VIEW_GRAD_FUNCTION_TEMPLATE
        self.PYBOOST_NATIVE_GRAD_FUNCTIONS_TEMPLATE = template.PYBOOST_NATIVE_GRAD_FUNCTIONS_TEMPLATE
        self.native_function_multi_output_template = template.MULTI_OUTPUT_TEMPLATE
        self.PYBOOST_NATIVE_COMM_GRAD_FUNCTION_TEMPLATE = template.PYBOOST_NATIVE_COMM_GRAD_FUNCTION_TEMPLATE
        self.composite_include_header_template = template.COMPOSITE_INCLUDE_HEADER_TEMPLATE
        self.native_view_function_output_template =\
            "const auto &output_value = runtime::ValueConverter::ToValue(outputs);\n"
        self.native_function_single_output_template = "const auto &output_value = op->outputs()[0];\n"
        self.convert_template = Template(
            "auto convert_$arg_name = runtime::ValueConverter::${convert_func}(ConvertNode2Value($arg_name));\n")

    def generate(self, work_path, op_protos):
        """
        Generates C++ gradient function implementations and writes them to a source file.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information such as
        operator names, arguments, and conversion types. It constructs C++ function bodies and generates the required
        include statements. The generated content is saved to a specified path.

        Args:
            work_path (str): The file path where the generated C++ file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.

        Returns:
            None
        """
        pyboost_func_str = ''
        ops_inc_head_set = set()
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if not op_proto.op_dispatch.enable:
                continue

            op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
            convert_value_type_str = self._convert_native_value_type(op_proto)
            convert_value_type_str += self._contiguous_tensor_value(op_proto, "device_target_")
            call_args_str = self._get_call_args(op_proto)
            call_args_with_type = self._get_call_args_with_type(op_proto)

            first_var_name = op_proto.op_args[0].arg_name
            output_expr = self._get_output_expr(op_proto)

            pyboost_func_str += \
                self._get_native_grad_function_template(op_proto).replace(func_name=op_proto.op_class.name,
                                                                          op_name=op_proto.op_class.name,
                                                                          op_args=op_args_str,
                                                                          convert_body=convert_value_type_str,
                                                                          call_args=call_args_str,
                                                                          call_args_with_type=call_args_with_type,
                                                                          first_var_name=first_var_name,
                                                                          output_expr=output_expr,
                                                                          operator_name=op_proto.op_name)
            pyboost_func_str = pyboost_func_str + template.NEW_LINE
            ops_inc_head_set.add(
                template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_proto.op_class.name[0].lower()))
            if op_proto.composite:
                ops_inc_head_set.add(self.composite_include_header_template.replace(
                    operator_name=op_proto.op_name
                ))
        native_grad_func_file = \
            self.PYBOOST_NATIVE_GRAD_FUNCTIONS_TEMPLATE.replace(function_body=pyboost_func_str,
                                                                ops_inc=list(sorted(ops_inc_head_set)))
        save_file(os.path.join(work_path, K.PYBOOST_NATIVE_GRAD_FUNC_GEN_PATH),
                  "pyboost_native_grad_functions.cc", native_grad_func_file)

    def _get_native_grad_function_template(self, op_proto):
        if op_proto.op_view:
            return self.PYBOOST_NATIVE_VIEW_GRAD_FUNCTION_TEMPLATE
        if op_proto.op_dispatch.enable and op_proto.op_dispatch.is_comm_op:
            return self.PYBOOST_NATIVE_COMM_GRAD_FUNCTION_TEMPLATE
        return self.PYBOOST_NATIVE_GRAD_FUNCTION_TEMPLATE

    def _convert_native_value_type(self, op_proto: OpProto) -> str:
        """
        Generates native value conversion functions for operator arguments.

        This method processes each argument of the operator prototype and constructs conversion statements
        based on the argument's data type.

        Args:
            op_proto (OpProto): The operator prototype from which the argument data will be extracted.

        Returns:
            str: A string containing the conversion statements for the operator's arguments.
        """
        parser_func_str = ''
        for op_arg in op_proto.op_args:
            is_optional = is_optional_param(op_arg)
            convert_type_str = pyboost_utils.get_value_convert_type_str(op_arg.arg_dtype, is_optional, op_proto.op_view)
            parser_func_str += self.convert_template.replace(arg_name=op_arg.arg_name, convert_func=convert_type_str)
        return parser_func_str

    def _contiguous_tensor_value(self, op_proto: OpProto, device_target: str) -> str:
        """
        Generates contiguous tensor value conversion functions if applicable.

        This method constructs conversion statements for tensors that need to be contiguous. If the operator is a view
        operation, no conversion is performed.

        Args:
            op_proto (OpProto): The operator prototype that contains the argument data.
            device_target (str): The device target to be used in the conversion statements.

        Returns:
            str: A string containing the contiguous tensor conversion statements.
        """
        if op_proto.op_view:
            return ''
        contiguous_template = Template(
            "convert_$arg_name = runtime::ValueConverter::ContiguousTensorValue($device_target, convert_$arg_name);\n")
        contiguous_func_str = ''
        need_contiguous_dtype = {'tensor', 'tuple[tensor]'}
        for op_arg in op_proto.op_args:
            if op_arg.arg_dtype not in need_contiguous_dtype:
                continue
            contiguous_func_str += contiguous_template.replace(arg_name=op_arg.arg_name, device_target=device_target)
        return contiguous_func_str

    def _get_output_expr(self, op_proto: OpProto):
        """
        Determines the output expression based on the operator prototype.

        This method checks if the operator produces multiple outputs and returns the corresponding output expression.

        Args:
            op_proto (OpProto): The operator prototype to evaluate.

        Returns:
            str: The output expression used in the function implementation.
        """
        if op_proto.op_view:
            return self.native_view_function_output_template
        output_expr = self.native_function_single_output_template
        if pyboost_utils.is_op_multi_output(op_proto.op_returns):
            output_expr = self.native_function_multi_output_template
        return output_expr

    def _get_call_args(self, op_proto: OpProto):
        """
        Generates the list of call arguments for the operator function.

        This method constructs a list of argument names prefixed with 'convert_' for use in the function call.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            list: A list of formatted argument names to be used in the function call.
        """
        call_args_str = []
        for op_arg in op_proto.op_args:
            call_arg = 'convert_' + op_arg.arg_name
            call_args_str.append(call_arg)
        return call_args_str

    def _get_call_args_with_type(self, op_proto: OpProto):
        """
        Generates the list of call arguments with type information.

        This method constructs a list of argument declarations with the appropriate type for the function definition.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            list: A list of argument declarations with types for the function definition.
        """
        call_args_with_type = []
        for op_arg in op_proto.op_args:
            call_args_with_type.append('const NodePtr &' + op_arg.arg_name)
        return call_args_with_type


class PyboostGradFunctionsHeaderGenerator(BaseGenerator):
    """
    PyboostGradFunctionsHeaderGenerator generates C++ header declarations for PyBoost gradient functions.

    This class processes operator prototypes to create function declarations for the grad functions in a header file.
    """

    def __init__(self):
        self.native_function_header_template = Template("static NodePtr $func_name(${call_args_with_type});\n")

    def generate(self, work_path, op_protos):
        """
        Generates C++ header declarations for gradient functions and writes them to a header file.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information
        and constructing the function declarations. The generated content is saved to a specified path.

        Args:
            work_path (str): The file path where the generated header file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.

        Returns:
            None
        """
        native_function_headers_str = ''
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if not op_proto.op_dispatch.enable:
                continue

            call_args_with_type = self._get_call_args_with_type(op_proto)

            func_header = self.native_function_header_template.replace(func_name=op_proto.op_class.name,
                                                                       call_args_with_type=call_args_with_type)
            native_function_headers_str += func_header
        native_grad_func_header_file = template.PYBOOST_NATIVE_GRAD_FUNCTIONS_HEADER_TEMPLATE.replace(
            native_grad_func_def=native_function_headers_str)

        save_file(os.path.join(work_path, K.PYBOOST_NATIVE_GRAD_FUNC_GEN_PATH),
                  "pyboost_native_grad_functions.h", native_grad_func_header_file)

    def _get_call_args_with_type(self, op_proto: OpProto):
        """
        Generates the list of call arguments with type information for the header declarations.

        This method constructs a list of argument declarations with the appropriate type for the function declaration.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            list: A list of argument declarations with types for the function declaration.
        """
        call_args_with_type = []
        for op_arg in op_proto.op_args:
            call_args_with_type.append('const NodePtr &' + op_arg.arg_name)
        return call_args_with_type
