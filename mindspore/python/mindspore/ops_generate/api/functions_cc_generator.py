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
This module provides classes for generating C++ header and implementation files for functions based on op_protos.
"""

import os

from common import template
from common.template import Template
import common.gen_constants as K
from common.gen_utils import save_file
from common.base_generator import BaseGenerator
from pyboost.op_template_parser import OpTemplateParser
from pyboost.pyboost_utils import is_optional_param, get_input_dtype, get_return_type


class FunctionsHeaderGenerator(BaseGenerator):
    """
    Generates C++ header files for backend functions based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the functions header generator with templates for code generation.
        """
        self.FUNCTIONS_H_TEMPLATE = template.FUNCTIONS_H_TEMPLATE
        self.function_interface_template = Template("${return_type} PYBOOST_API ${op_name}(${input_args});")
        self.function_interface_template_comm = Template(
            "${return_type} PYBOOST_API ${op_name}_inner(${input_args}," \
            "CommHandlePtr comm_handle, device::DeviceType target);"
        )
        self.function_interface_template_comm_return_handle = Template(
            "${return_type_with_handle} PYBOOST_API ${op_name}(${input_args});"
        )

    def generate(self, work_path, op_protos):
        """
        Generate the header file for backend functions.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the header.
        """
        functions_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or op_proto.composite is True:
                continue
            input_args_with_type_str = self._get_input_args(op_proto)
            return_type_str = _get_return_type_str(op_proto)
            function_template = (
                self.function_interface_template
                if not op_proto.op_dispatch.is_comm_op
                else self.function_interface_template_comm
            )
            functions = function_template.replace(op_name=op_proto.op_name,
                                                  input_args=input_args_with_type_str,
                                                  return_type=return_type_str)
            functions_list.append(functions)
            if op_proto.op_dispatch.is_comm_op:
                return_type_with_handle = _get_return_type_with_handle_str(return_type_str)
                functions_with_handle = (
                    self.function_interface_template_comm_return_handle.replace(
                        op_name=op_proto.op_name,
                        input_args=input_args_with_type_str,
                        return_type=return_type_str,
                        return_type_with_handle=return_type_with_handle,
                    )
                )
                functions_list.append(functions_with_handle)
        pyboost_func_h_str = self.FUNCTIONS_H_TEMPLATE.replace(op_call_with_grad=functions_list)
        save_path = os.path.join(work_path, K.MS_PYBOOST_FUNCTIONS_HEADER_AUTO_GEN_PATH)
        file_name = "functions.h"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_input_args(self, op_proto):
        """
        Get the input arguments with type information for the function interface.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: A string of input arguments with types.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg), op_proto.op_view)
            args_list.append("const " + input_dtype + " &" + op_arg.arg_name)
        return args_list


class FunctionsGenerator(BaseGenerator):
    """
    Generates C++ implementation files for backend functions based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the functions generator with templates for code generation.
        """
        self.FUNCTIONS_CC_TEMPLATE = template.FUNCTIONS_CC_TEMPLATE
        self.FUNCTION_BODY_TEMPLATE = template.FUNCTION_BODY_TEMPLATE
        self.FUNCTION_BODY_WRAPPER_TEMPLATE = template.FUNCTION_BODY_WRAPPER_TEMPLATE
        self.FUNCTION_VIEW_BODY_TEMPLATE = template.FUNCTION_VIEW_BODY_TEMPLATE
        self.FUNCTION_VIEW_CUSTOMIZE_BODY_TEMPLATE = template.FUNCTION_VIEW_CUSTOMIZE_BODY_TEMPLATE
        self.FUNCTION_COMM_BODY_TEMPLATE = template.FUNCTION_COMM_BODY_TEMPLATE
        self.pyboost_func_include_header_template = Template(
            f'#include "{K.MS_PYBOOST_BASE_HEADER_PATH}/auto_generate/${{operator_name}}.h"\n'
        )
        self.pyboost_view_func_include_header_template = Template(
            f'#include "{K.MS_OPS_VIEW_PATH}/${{operator_name}}_strides_calc.h"\n'
        )
        self.clone_inplace_input_template = Template(
            'GetCloneFunc()(op, prim::kPrim${class_name}, device_target, {${grad_args}});'
        )
        self.create_aclnn_op_template = Template(
            'auto op = CREATE_PYBOOST_OP(${class_name}, device_target);'
        )
        self.create_internal_op_template = Template(
            'auto op = CREATE_PYBOOST_SELECTED_OP(${class_name}, device_target);'
        )

    def generate(self, work_path, op_protos):
        """
        Generate the implementation file for backend functions.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the implementation.
        """
        func_include_headers_list = []
        op_call_with_grad_list = []
        ops_inc_head_set = set()
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if op_proto.composite:
                continue
            if op_proto.op_view:
                function_body, pyboost_func_include_header = self._get_function_view_body(op_proto)
            else:
                pyboost_func_include_header = self.pyboost_func_include_header_template.\
                    replace(operator_name=op_proto.op_name)
                function_body = self._get_function_body(op_proto)
            func_include_headers_list.append(pyboost_func_include_header)
            op_call_with_grad_list.append(function_body)
            ops_inc_head_set.add(
                template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_proto.op_class.name[0].lower()))
        pyboost_func_h_str = self.FUNCTIONS_CC_TEMPLATE.replace(op_call_with_grad=op_call_with_grad_list,
                                                                pyboost_op_header_include=func_include_headers_list,
                                                                ops_inc=list(sorted(ops_inc_head_set)))
        save_path = os.path.join(work_path, K.MS_PYBOOST_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.cc"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_function_view_body(self, op_proto):
        """
        Get the function body for a given view operator prototype.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: The generated function body.
        """
        function_body_template = self.FUNCTION_VIEW_BODY_TEMPLATE
        pyboost_func_include_header = self.pyboost_view_func_include_header_template.\
            replace(operator_name=op_proto.op_name)
        if not op_proto.bprop_expander or op_proto.op_name in ["reshape"]:
            function_body_template = self.FUNCTION_VIEW_CUSTOMIZE_BODY_TEMPLATE
            pyboost_func_include_header = ""
        op_parser = OpTemplateParser(op_proto)
        input_args = self._get_input_args(op_proto, False)
        input_args_with_type = self._get_input_args(op_proto, True)
        call_args_with_type = op_parser.parse_call_args_with_types(True)
        call_args = OpTemplateParser.parse_original_call_args(op_proto.op_args)
        call_args_tensors = op_parser.get_call_args_tensor()
        storage_calc_str = op_proto.op_class.name
        return_values, _ = op_parser.generate_pyboost_outputs()
        return_type_str = _get_return_type_str(op_proto)
        function_body = function_body_template.replace(op_name=op_proto.op_name,
                                                       class_name=op_proto.op_class.name,
                                                       input_args_with_type=input_args_with_type,
                                                       input_args=input_args,
                                                       storage_calc=storage_calc_str,
                                                       call_args_with_type=call_args_with_type,
                                                       call_args=call_args,
                                                       call_tensors=call_args_tensors,
                                                       input=call_args[0],
                                                       return_values=return_values,
                                                       return_type=return_type_str)
        return function_body, pyboost_func_include_header

    def _get_function_body(self, op_proto):
        """
        Get the function body for a given operator prototype.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: The generated function body.
        """
        input_args = self._get_input_args(op_proto, False)
        input_args_with_type = self._get_input_args(op_proto, True)
        inplace_clone_args = self._get_clone_input_args(op_proto, False, False)
        clone_func_str = self._get_clone_inplace_str(op_proto.op_inplace, op_proto.op_class.name, inplace_clone_args)
        return_type_str = _get_return_type_str(op_proto)
        if op_proto.op_dispatch.is_comm_op:
            return_type_with_handle = _get_return_type_with_handle_str(return_type_str)
            comm_body = self.FUNCTION_COMM_BODY_TEMPLATE.replace(op_name=op_proto.op_name,
                                                                 class_name=op_proto.op_class.name,
                                                                 input_args=input_args,
                                                                 clone_func=clone_func_str,
                                                                 input_args_with_type=input_args_with_type,
                                                                 return_type=return_type_str,
                                                                 return_type_with_handle=return_type_with_handle)
            return comm_body
        create_op_str = self.create_aclnn_op_template.replace(class_name=op_proto.op_class.name)
        if getattr(op_proto.op_dispatch, 'internal_op_ascend') != 'None':
            create_op_str = self.create_internal_op_template.replace(class_name=op_proto.op_class.name)

        function_body = self.FUNCTION_BODY_TEMPLATE.replace(op_name=op_proto.op_name,
                                                   class_name=op_proto.op_class.name,
                                                   create_op=create_op_str,
                                                   input_args=input_args,
                                                   clone_func=clone_func_str,
                                                   input_args_with_type=input_args_with_type,
                                                   return_type=return_type_str)

        function_body_wrapper \
            = self.FUNCTION_BODY_WRAPPER_TEMPLATE.replace(op_name=op_proto.op_name,
                                                          input_args=input_args,
                                                          input_args_with_type=input_args_with_type,
                                                          return_type=return_type_str)
        return function_body + function_body_wrapper


    def _get_input_args(self, op_proto, has_type):
        """
        Get the input arguments for the function body.

        Args:
            op_proto: The operator prototype.
            has_type (bool): Whether to include type information for the arguments.

        Returns:
            str: A string of input arguments, with or without types.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg), op_proto.op_view)
            if has_type:
                args_list.append("const " + input_dtype + " &" + op_arg.arg_name)
            else:
                args_list.append(op_arg.arg_name)
        return args_list

    def _get_clone_inplace_str(self, is_inplace_op: bool, class_name: str, grad_args: list):
        """
        Generates the view base str of arguments for the operator.

        This method constructs a list of argument names that need to be cast to their corresponding types.

        Args:
            is_view_or_inplace (bool): Whether the op is view op or inplace op.
            grad_args (list): grad args

        Returns:
            str: Formatted view or inplace first argument names.
        """
        if not is_inplace_op:
            return ''
        return self.clone_inplace_input_template.replace(class_name=class_name, grad_args=grad_args)

    def _get_clone_input_args(self, op_proto, has_type, with_optional):
        """
        Get the input arguments for the DoGrad function.

        Args:
            op_proto: The operator prototype.
            has_type (bool): Whether to include type information for the arguments.

        Returns:
            list: A list of input arguments for the DoGrad function.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg), op_proto.op_view)
            if has_type:
                args_list.append(f"const {input_dtype} &{op_arg.arg_name}")
            else:
                if not with_optional and is_optional_param(op_arg):
                    args_list.append(f"OptionalToValue({op_arg.arg_name})")
                else:
                    args_list.append(f"{op_arg.arg_name}")
        return args_list


def _get_return_type_with_handle_str(return_type_str):
    return f"std::tuple<{return_type_str}, CommHandlePtr>"


def _get_return_type_str(op_proto):
    """
    Get the return type string for the function.

    Args:
        op_proto: The operator prototype.

    Returns:
        str: The return type as a string.
    """
    returns_type = []
    type_convert_to_base = {
        'std::vector<mindspore::tensor::TensorPtr>': 'std::vector<mindspore::tensor::TensorPtr>',
        'mindspore::tensor::TensorPtr': 'mindspore::tensor::TensorPtr'
    }
    for return_obj in op_proto.op_returns:
        temp_return = get_return_type(return_obj.arg_dtype)
        if temp_return in type_convert_to_base:
            returns_type.append(type_convert_to_base[temp_return])
        else:
            raise Exception("Not return found")
    if len(returns_type) == 1:
        cpp_func_return = returns_type[0]
    elif len(returns_type) > 1:
        cpp_func_return = "std::tuple<"
        cpp_func_return += ', '.join(s for s in returns_type)
        cpp_func_return += ">"
    else:
        raise Exception("Not return found")
    return cpp_func_return
