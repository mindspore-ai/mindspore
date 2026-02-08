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
This module defines the PyboostFunctionsGenerator class for generating C++ functions for PyBoost operations.

The generator processes operator prototypes and constructs the necessary function definitions, including
conversions for optional parameters and tensor arguments. It generates the registration code and includes
the necessary header files for the generated functions.
"""

import os

from common import template
import common.gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.op_proto import OpProto
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils
from pyboost.pyboost_utils import get_convert_type_str, is_optional_param, get_input_args_type_str, \
    is_tensor_list
from .op_template_parser import OpTemplateParser


class PyboostFunctionsImplGenerator(BaseGenerator):
    """
    Generates PyBoost functions based on operator prototypes.

    This class processes operator prototypes (`op_protos`) to create the necessary C++ function definitions for
    PyBoost operations. It constructs function bodies, handles optional value conversions, and generates
    registration code and header inclusions.
    """

    def __init__(self):
        """Initializes the PyboostFunctionsGenerator with the necessary templates."""
        self.pyboost_func_include_header_template = Template(
            f'#include "{K.MS_PYBOOST_BASE_HEADER_PATH}/auto_generate/${{operator_name}}.h"\n'
        )
        self.composite_include_header_template = template.COMPOSITE_INCLUDE_HEADER_TEMPLATE

        self.convert_optional_to_value_template = Template(
            "auto ${output} = PyNativeAlgo::PyBoost::OptionalToValue(${input});\n"
        )
        self.convert_to_tensor_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToTensor(${input}, ${need_contiguous}, '
            'op_run_info->requires_grad, ${is_inplace});\n'
        )
        self.convert_to_tensor_view_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToTensor(${input}, ${need_contiguous}, '
            'requires_grad, ${is_inplace});\n'
        )
        self.convert_to_tensor_inplace_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToTensor(${input}, ${need_contiguous}, '
            'op_run_info->requires_grad, ${is_inplace});\n'
        )
        self.convert_to_tensor_list_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToValueTuple(${input}, ${need_contiguous}, '
            'op_run_info->requires_grad, ${is_inplace});\n'
        )
        self.convert_to_tensor_list_view_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToValueTuple(${input}, ${need_contiguous}, '
            'requires_grad, ${is_inplace});\n'
        )
        self.convert_to_tensor_list_inplace_template = Template(
            'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToValueTuple(${input}, ${need_contiguous}, '
            'op_run_info->requires_grad, ${is_inplace});\n'
        )
        self.implicit_cast_template = Template(
            '// Do mixed precision and implicit cast\n' \
            'static const std::vector<std::vector<size_t>> same_type_table{${same_type}};\n' \
            'auto [${cast_args}] =\n' \
            '   PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<${type_num}>(op_run_info, "${class_name}", \
 same_type_table, ${call_args});\n'
        )
        self.convert_template = Template("auto $arg_name = converter.${convert_func}(args, $arg_index);\n")
        self.input_args_template = Template(" const ${arg_type}& ${arg_name},")
        self.PYBOOST_CORE_CC_TEMPLATE = template.PYBOOST_CORE_CC_TEMPLATE
        self.TENSOR_FUNC_CLASS_REG = template.TENSOR_FUNC_CLASS_REG
        self.OP_DEF_INC_HEAD_TEMPLATE = template.OP_DEF_INC_HEAD_TEMPLATE

        self.PYBOOST_CORE_BODY_TEMPLATE = template.PYBOOST_CORE_BODY_TEMPLATE
        self.PYBOOST_CORE_BODY_VIEW_TEMPLATE = template.PYBOOST_CORE_BODY_VIEW_TEMPLATE
        self.PYBOOST_CORE_BODY_COMM_TEMPLATE = template.PYBOOST_CORE_BODY_COMM_TEMPLATE
        self.PYBOOST_CORE_BODY_SYNC_TEMPLATE = template.PYBOOST_CORE_BODY_SYNC_TEMPLATE
        self.PYBOOST_CORE_BODY_VIEW_SYNC_TEMPLATE = template.PYBOOST_CORE_BODY_VIEW_SYNC_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates the C++ PyBoost functions and writes them to the specified files.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information
        such as operator names, arguments, and conversion types. It constructs the function definitions, includes,
        and registration code. The generated content is saved to the specified path as a C++ source file.

        Args:
            work_path (str): The file path where the generated files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.

        Returns:
            None
        """
        pyboost_func_include_headers_str = ''
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue

            if op_proto.op_dispatch.is_comm_op:
                pyboost_func_include_headers_str += self.pyboost_func_include_header_template.replace(
                    operator_name=op_proto.op_name)

            if op_proto.composite:
                pyboost_func_include_headers_str += self.composite_include_header_template.replace(
                    operator_name=op_proto.op_name
                )

        # generate pyboost core cc
        pyboost_core_body_str = self._get_pyboost_core_body_all_str(op_protos)
        pyboost_core_file \
            = self.PYBOOST_CORE_CC_TEMPLATE.replace(include_op_header=pyboost_func_include_headers_str,
                                                    function_body=pyboost_core_body_str)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_core.cc"
        save_file(save_path, file_name, pyboost_core_file)

    def _get_pyboost_core_body_all_str(self, op_protos):
        """
        Generates pyboost functions implementation string for all operations.

        Args:
            op_protos (list): A list of op prototypes.

        Returns:
            str: pyboost functions implementation string for all operations.
        """
        pyboost_core_body_str = ''
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            pyboost_core_body_str += self._get_pyboost_core_body_str(op_proto)

        return pyboost_core_body_str

    def _get_pyboost_core_body_str(self, op_proto):
        """
        Generates pyboost functions implementation string for specific operator.

        Args:
            op_proto (prototype): op prototype.

        Returns:
            str: pyboost functions implementation string of specific operator.
        """
        op_parser = OpTemplateParser(op_proto)
        op_pyboost_func_name = op_parser.get_pyboost_func_name()
        op_def_name_str = op_parser.get_op_def_name_str()
        type_num, same_type = op_parser.gen_signature_same_type_table()
        parser_body_str = self._generate_parser_func(op_proto)
        op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
        convert_stub_str = self._get_convert_stub_str(op_proto)
        call_args_str = self._get_call_args_str(op_proto)
        cast_args_str = self._get_cast_to_value_str(op_proto)
        op_input_args_str = self._get_input_args_str(op_proto)
        output_num_str = len(op_proto.op_returns)
        has_side_effect_str = 'true' if op_proto.op_view or op_proto.op_inplace else 'false'
        pyboost_core_body_tpl = self._get_pyboost_core_body_tpl(op_proto)
        if op_proto.op_view:
            implicit_cast_str = ''
        else:
            implicit_cast_str = self.implicit_cast_template.replace(cast_args=cast_args_str,
                                                                    type_num=type_num,
                                                                    call_args=call_args_str,
                                                                    same_type=same_type,
                                                                    class_name=op_proto.op_class.name)
        return pyboost_core_body_tpl.replace(func_name=op_pyboost_func_name,
                                             op_def_name=op_def_name_str,
                                             input_args=op_input_args_str,
                                             parser_body=parser_body_str,
                                             op_name=op_proto.op_class.name,
                                             class_name=op_proto.op_class.name,
                                             implicit_cast=implicit_cast_str,
                                             op_args=op_args_str,
                                             convert_stub=convert_stub_str,
                                             call_args=call_args_str,
                                             cast_args=cast_args_str,
                                             output_num=output_num_str,
                                             has_side_effect=has_side_effect_str,
                                             operator_name=op_proto.op_name)

    def _generate_parser_func(self, op_proto: OpProto) -> str:
        """
        Generates the parsing function for the operator's arguments.

        This method constructs the code for converting each argument in the operator prototype to its appropriate
        type, handling optional parameters as necessary.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            str: The generated parsing function code as a string.
        """
        parser_func_str = ''
        for index, op_arg in enumerate(op_proto.op_args):
            is_optional = is_optional_param(op_arg)
            if op_arg.is_type_id:
                convert_type_str = get_convert_type_str('type', is_optional, op_proto.op_view)
            else:
                convert_type_str = get_convert_type_str(op_arg.arg_dtype, is_optional, op_proto.op_view)

            parser_func_str += self.convert_template.replace(arg_name=op_arg.arg_name, convert_func=convert_type_str,
                                                             arg_index=pyboost_utils.get_index(index))
        return parser_func_str

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

    def _get_convert_stub_str(self, op_proto: OpProto):
        """
        Generates the conversion stub code for the operator's arguments.

        This method creates code for converting operator arguments to tensor format, depending on whether they
        are view operations or standard tensor operations.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            str: The generated conversion stub code as a string.
        """
        convert_stub_str = ''
        need_contiguous = 'true'
        is_inplace = 'false'
        convert_to_tensor_template = self.convert_to_tensor_template
        convert_to_tensor_list_template = self.convert_to_tensor_list_template
        if op_proto.op_view:
            # View/ACLNN op does not need to convert to contiguous tensor.
            need_contiguous = 'false'
            convert_to_tensor_template = self.convert_to_tensor_view_template
            convert_to_tensor_list_template = self.convert_to_tensor_list_view_template
        if op_proto.op_inplace:
            # Cpu inplace need contiguous tensor
            is_inplace = 'true'
            convert_to_tensor_template = self.convert_to_tensor_inplace_template
            convert_to_tensor_list_template = self.convert_to_tensor_list_inplace_template
        for op_arg in op_proto.op_args:
            if pyboost_utils.is_tensor(op_arg):
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor"
                convert_stub_str += convert_to_tensor_template.replace(input=op_arg.arg_name,
                                                                       output=convert_stub_output_name,
                                                                       need_contiguous=need_contiguous,
                                                                       is_inplace=is_inplace)
            elif pyboost_utils.is_tensor_list(op_arg):
                # To adapt the cases where TensorList is optional.
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor_list"
                convert_stub_str += convert_to_tensor_list_template.replace(input=op_arg.arg_name,
                                                                            output=convert_stub_output_name,
                                                                            need_contiguous=need_contiguous,
                                                                            is_inplace=is_inplace)
        return convert_stub_str

    def _get_call_args_str(self, op_proto: OpProto):
        """
        Generates the list of call arguments for the operator.

        This method constructs a list of argument names for the function call, adapting the names for
        optional tensors and tensor lists as needed.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            list: A list of formatted argument names for the function call.
        """
        call_args_str = []
        for op_arg in op_proto.op_args:
            if pyboost_utils.is_tensor(op_arg):
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor"
                call_arg = convert_stub_output_name
            elif pyboost_utils.is_tensor_list(op_arg):
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor_list"
                call_arg = convert_stub_output_name
            else:
                call_arg = op_arg.arg_name
            call_args_str.append(call_arg)
        return call_args_str

    def _get_cast_to_value_str(self, op_proto: OpProto):
        """
        Generates the list of cast arguments for the operator.

        This method constructs a list of argument names that need to be cast to their corresponding types.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            list: A list of formatted cast argument names.
        """
        cast_args_str = []
        for op_arg in op_proto.op_args:
            cast_str = 'cast_'
            if pyboost_utils.is_tensor(op_arg):
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor"
                cast_arg = cast_str + convert_stub_output_name
            elif pyboost_utils.is_tensor_list(op_arg):
                # To adapt the cases where TensorList is optional.
                convert_stub_output_name = op_arg.arg_name + '_optional' if is_optional_param(op_arg) \
                    else op_arg.arg_name + "_tensor_list"
                cast_arg = cast_str + convert_stub_output_name
            else:
                cast_arg = cast_str + op_arg.arg_name
            cast_args_str.append(cast_arg)
        return cast_args_str

    def _get_pyboost_core_body_tpl(self, op_proto: OpProto):
        if len(op_proto.op_returns) == 1 and is_tensor_list(op_proto.op_returns[0]):
            # op output size is unknown
            return self.PYBOOST_CORE_BODY_VIEW_SYNC_TEMPLATE\
                if op_proto.op_view else self.PYBOOST_CORE_BODY_SYNC_TEMPLATE
        if op_proto.op_view:
            return self.PYBOOST_CORE_BODY_VIEW_TEMPLATE
        if op_proto.op_dispatch.is_comm_op:
            return self.PYBOOST_CORE_BODY_COMM_TEMPLATE
        return self.PYBOOST_CORE_BODY_TEMPLATE
