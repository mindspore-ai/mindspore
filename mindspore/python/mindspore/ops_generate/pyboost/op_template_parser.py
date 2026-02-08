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
This module defines the OpTemplateParser class for parsing operator prototypes.

The OpTemplateParser class is responsible for converting attributes of OpProto instances into strings
that can be directly used to fill in code templates. It handles the parsing of argument types,
function signatures, and generates conversion stubs for PyBoost operations.
"""

import re

import common.gen_constants as K
from common.template import Template
from common.op_proto import OpProto

from . import pyboost_utils
from .pyboost_utils import get_input_dtype, tuple_input_to_cpp_type, get_return_type, \
    number_input_to_cpp_type, get_const_number_convert, get_tuple_input_convert, is_optional_param, \
    get_input_args_type_str, basic_type_convert_str, input_dtype_to_cpp_type


class OpTemplateParser:
    """
    Parses operator prototypes and generates template-compatible strings.

    This class converts the attributes of an OpProto instance into the strings needed for code generation
    in PyBoost operations.

    Attributes:
        op_proto (OpProto): The operator prototype containing the relevant information.
    """

    def __init__(self, op_proto: OpProto):
        self.op_proto = op_proto
        self.arg_handler_template = Template(
            "const auto arg_handler_input_${idx} = parse_args.arg_list_[${idx}];\n"
            "parse_args.arg_list_[${idx}] = "
            "pynative::${func_str}(\"${func_name}\", \"${op_arg_name}\", parse_args.arg_list_[${idx}]);\n"
            "trace::PassNodeInArg(arg_handler_input_${idx}, parse_args.arg_list_[${idx}]);\n"
            "parse_args.src_types_[${idx}] = ops::OP_DTYPE::DT_BEGIN;\n"
            "parse_args.dst_types_[${idx}] = ${new_type};\n"
        )
        self.arg_handler_optional_template = Template(
            'if (!py::isinstance<py::none>(parse_args.arg_list_[${idx}])) {\n'
            '  ${arg_handler_str}\n'
            '}\n'
        )
        self.arg_handler_type_map = {"to_2d_paddings": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "dtype_to_type_id": "ops::OP_DTYPE::DT_INT",
                                     "to_kernel_size": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "to_strides": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "str_to_enum": "ops::OP_DTYPE::DT_INT",
                                     "to_pair": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "to_dilations": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "to_output_padding": "ops::OP_DTYPE::DT_TUPLE_INT",
                                     "to_rates": "ops::OP_DTYPE::DT_TUPLE_INT"}
        self.black_list_for_arg_handlers = pyboost_utils.get_pyboost_arg_handlers_black_list()

    @staticmethod
    def _parse_call_args_types(op_proto, basic_type=False, is_convert=False):
        """
        Parses the data types of the call arguments for the operator.

        Args:
            op_args (list): A list of operator arguments.

        Returns:
            list: A list of data types for the call arguments.
        """
        call_args_types = []
        if basic_type and not op_proto.op_view:
            basic_type = False
            raise Exception("Only view op support basic type now, please check.")
        for op_arg in op_proto.op_args:
            is_optional = is_optional_param(op_arg)
            if is_convert:
                if op_arg.is_type_id:
                    call_args_types.append('TypeId')
                    continue
                call_args_types.append(input_dtype_to_cpp_type(
                    op_arg.arg_dtype, is_optional))
            else:
                call_args_types.append(get_input_dtype(
                    op_arg.arg_dtype, is_optional, basic_type))
        return call_args_types

    def parse_call_args_with_types(self, basic_type=False, is_convert=False):
        """
        Parses the original call arguments and their types for the operator.

        Returns:
            list: A list of formatted strings representing the call arguments with their types.
        """
        call_args = OpTemplateParser.parse_original_call_args(self.op_proto.op_args)
        call_args_after_convert, _, _ = self.op_args_converter()
        call_args_types = self._parse_call_args_types(self.op_proto, basic_type, is_convert)
        call_args_with_types = []
        if is_convert:
            for type_name, arg_name in zip(call_args_types, call_args_after_convert):
                call_args_with_types.append("const " + type_name + " &" + arg_name)
            return call_args_with_types
        for type_name, arg_name in zip(call_args_types, call_args):
            call_args_with_types.append("const " + type_name + " &" + arg_name)
        return call_args_with_types

    def parse_need_malloc_tensors(self):
        """
        Parses the operator arguments to identify which tensors require memory allocation.

        Returns:
            tuple: A tuple containing:
                - need_malloc_tensors (list): Arguments that need memory allocation.
                - tensor_list_convert (list): Conversions needed for tensor lists.
                - call_args_with_tensor (list): The call arguments formatted for tensors.
        """
        need_malloc_tensors = []
        tensor_list_convert = []
        call_args_with_tensor = []
        call_args = OpTemplateParser.parse_original_call_args(self.op_proto.op_args)
        for op_arg, call_arg in zip(self.op_proto.op_args, call_args):
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
                need_malloc_tensors.append(call_arg)
                call_args_with_tensor.append(call_arg)
            elif tuple_input_to_cpp_type(op_arg.arg_dtype) and pyboost_utils.is_tensor_list(op_arg):
                need_malloc_tensors.append(call_arg + "_vector")
                tensor_list_convert.append(
                    get_tuple_input_convert(call_arg, op_arg.arg_dtype))
                call_args_with_tensor.append(call_arg + "_vector")
            else:
                call_args_with_tensor.append(call_arg)
        return need_malloc_tensors, tensor_list_convert, call_args_with_tensor

    @staticmethod
    def parse_original_call_args(op_args):
        """
        Parses the original call arguments from the operator prototype.

        Args:
            op_args (list): A list of operator arguments.

        Returns:
            list: A list of formatted strings representing the original call arguments.
        """
        call_args = []
        for op_arg in op_args:
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
            elif pyboost_utils.is_tensor_list(op_arg):
                call_arg = op_arg.arg_name + "_tensor_list"
            else:
                call_arg = op_arg.arg_name
            call_args.append(call_arg)
        return call_args

    def op_args_converter(self):
        """
        Converts operator arguments to the corresponding C++ data types.

        Returns:
            tuple: A tuple containing:
                - call_args_after_convert (list): The converted call arguments.
                - value_tuple_convert (list): Conversions needed for value tuples.
                - const_number_convert (list): Conversions needed for constant numbers.
        """
        call_args_after_convert = []
        value_tuple_convert = []
        const_number_convert = []
        call_args = OpTemplateParser.parse_original_call_args(self.op_proto.op_args)
        for op_arg, call_arg in zip(self.op_proto.op_args, call_args):
            if number_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_imm")
                const_number_convert.append(
                    get_const_number_convert(call_arg, op_arg))
            elif tuple_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_vector")
                value_tuple_convert.append(
                    get_tuple_input_convert(call_arg, op_arg.arg_dtype))
            else:
                call_args_after_convert.append(call_arg)
        if const_number_convert:
            const_number_convert.insert(
                0, '// Convert ValuePtr to c++ scalar\n')
        if value_tuple_convert:
            value_tuple_convert.insert(0, '// ValueTuple to std::vector\n')
        return call_args_after_convert, value_tuple_convert, const_number_convert

    def get_pyboost_func_name(self):
        """
        Gets the PyBoost function name based on the operator's class name.

        Returns:
            str: The generated PyBoost function name.
        """
        return "Pyboost_" + self.op_proto.op_class.name

    def get_pyboost_name(self):
        """
        Gets the PyBoost name for the operator.

        Returns:
            str: The generated PyBoost name for the operator.
        """
        return "pyboost_" + self.op_proto.op_name

    def get_op_def_name_str(self):
        """
        Gets the operator definition name string.

        Returns:
            str: The generated operator definition name string.
        """
        return "g" + self.op_proto.op_class.name

    def gen_signature_same_type_table(self):
        """
        Generates a signature table for arguments of the same type.

        Returns:
            tuple: A tuple containing:
                - type_num (int): The number of argument types.
                - signature_table (str): The generated signature table as a string.
        """
        signature_table = ''
        type_num = 0
        args_signature = self.op_proto.op_args_signature
        if args_signature is not None:
            dtype_group = args_signature.dtype_group
            indexes = {arg.arg_name: index for index, arg in enumerate(self.op_proto.op_args)}
            if dtype_group is not None:
                match = re.findall(r'\((.*?)\)', dtype_group)
                for item in match:
                    name_args = item.replace(' ', '').split(",")
                    signature_table += '{'
                    for arg in name_args:
                        arg_index = indexes[arg]
                        signature_table += f"""{arg_index}, """
                    signature_table = signature_table[:-2]
                    signature_table += '}, '
                    type_num += 1
                signature_table = signature_table[:-2]
        return type_num, signature_table

    def get_call_args_tensor(self):
        """
        Retrieves the call arguments that are of tensor type.

        Returns:
            list: A list of call arguments that are tensors.
        """
        call_args_tensor = []
        call_args_types = self._parse_call_args_types(self.op_proto)
        call_args = OpTemplateParser.parse_original_call_args(self.op_proto.op_args)
        for _type, arg_name in zip(call_args_types, call_args):
            if _type in ("mindspore::tensor::TensorPtr", "std::optional<mindspore::tensor::TensorPtr>"):
                call_args_tensor.append(arg_name)
        return call_args_tensor

    def has_prim_init(self):
        """
        Checks if any arguments require primitive initialization.

        Returns:
            bool: True if any argument requires primitive initialization, otherwise False.
        """
        op_args = self.op_proto.op_args
        has_prim_init = False
        for op_arg in op_args:
            prim_init = op_arg.is_prim_init
            if prim_init:
                has_prim_init = True
                break
        return has_prim_init

    def generate_pyboost_op_func_return_type(self):
        """
        Generates the C++ return type for the PyBoost operator function.

        Returns:
            str: The generated C++ return type for the function.

        Raises:
            Exception: If no valid return type is found.
        """
        returns_type = []
        type_convert_to_base = {
            'std::vector<mindspore::tensor::TensorPtr>': 'std::vector<mindspore::tensor::TensorPtr>',
            'mindspore::tensor::TensorPtr': 'mindspore::tensor::TensorPtr'
        }
        for return_obj in self.op_proto.op_returns:
            temp_return = get_return_type(return_obj.arg_dtype)
            if temp_return in type_convert_to_base:
                returns_type.append(type_convert_to_base[temp_return])
            else:
                raise Exception("Not return found")
        if len(returns_type) == 1:
            cpp_func_return = returns_type[0]
        elif len(returns_type) > 1:
            cpp_func_return = "std::tuple<"
            cpp_func_return += ','.join(s for s in returns_type)
            cpp_func_return += ">"
        else:
            raise Exception("Not return found")
        return cpp_func_return

    def generate_pyboost_outputs(self):
        """
        Generates the output variables for the PyBoost operator function.

        Returns:
            tuple: A tuple containing:
                - op_outputs (str): The output variable representation for the operator.
                - call_outputs (str): The call output variable representation for the operator.
        """
        op_outputs = ''
        call_outputs = ''
        returns_type = []
        for return_obj in self.op_proto.op_returns:
            returns_type.append(get_return_type(return_obj.arg_dtype))

        if len(returns_type) == 1:
            if returns_type[0] == 'mindspore::tensor::TensorPtr':
                op_outputs = 'outputs[0]'
                call_outputs = 'outputs_[0]'
            elif returns_type[0] == "std::vector<mindspore::tensor::TensorPtr>":
                op_outputs = 'outputs'
                call_outputs = 'outputs_'
            else:
                raise Exception(
                    "Not support return type {}".format(returns_type[0]))
        elif len(returns_type) > 1:
            outputs_str = ''
            for i in range(len(returns_type)):
                outputs_str += 'outputs[{}],'.format(i)
            op_outputs = outputs_str[:-1]

            outputs_str = ''
            for i in range(len(returns_type)):
                outputs_str += 'outputs_[{}],'.format(i)
            outputs_str = outputs_str[:-1]
            call_outputs = "std::make_tuple(" + outputs_str + ")"

        return op_outputs, call_outputs

    def _generate_signature_arg_dtype_str(self, arg, disable_scalar_tensor):
        """
        Generate argument dtype string with arg_handler and disable_scalar_tensor configuration.

        Args:
            arg (OpArg): The operation argument object.
            disable_scalar_tensor (list[str]): List of argument names that disable scalar tensor conversion.

        Returns:
            str: Formatted argument dtype string.
        """
        arg_handler = arg.arg_handler
        if arg_handler and arg_handler not in self.black_list_for_arg_handlers:
            if arg_handler in K.ARG_HANDLER_MAP:
                arg_dtype = K.ARG_HANDLER_MAP[arg_handler]
            else:
                raise ValueError(
                    f"Generate failed. Check if {arg_handler} is registered in TensorFuncRegCppGenerator.")
        else:
            arg_dtype = arg.arg_dtype
            for cast_type in arg.type_cast:
                arg_dtype += f'|{cast_type}'

        # handle disable scalar tensor
        if disable_scalar_tensor and arg.arg_name in disable_scalar_tensor:
            arg_dtype += '|!tensor'

        return arg_dtype

    def generate_signature_str(self, kw_only_args=None, varargs=None,
                               disable_scalar_tensor=None, *, is_tensor_api: bool) -> str:
        """
        Generates a single function signature string for the given operation prototype.

        Args:
            kw_only_args (list[str]): List of keyword-only argument names.
            varargs (list[str]): List of variable args names.
            disable_scalar_tensor (list[str]): List of args names which disable tensor to scalar.

        Kwargs:
            is_tensor_api (bool): Whether this function is used in the Tensor API scenario.

        Returns:
            str: Generated function signature string.
        """

        op_name = self.op_proto.op_class.name
        args_str = f'"{op_name}('
        first_arg = True
        kw_args_init_flag = False

        arg_index = 0
        for arg in self.op_proto.op_args:
            arg_name = arg.arg_name

            if is_tensor_api and _is_input_arg(arg_name, op_name):
                continue

            single_arg = ''
            if not first_arg:
                single_arg = ', '

            arg_dtype = self._generate_signature_arg_dtype_str(arg, disable_scalar_tensor)
            # handle varargs params
            if varargs and arg_name in varargs and arg_index == 0:
                single_arg += f"{arg_dtype} *{arg_name}"
            else:
                single_arg += f"{arg_dtype} {arg_name}"

            if arg.as_init_arg:
                single_arg += f"={arg.default}"

            # handle keyword-only params
            if kw_only_args and not kw_args_init_flag and arg_name == kw_only_args[0]:
                single_arg = ("*, " if first_arg else ", *") + single_arg
                kw_args_init_flag = True

            args_str += single_arg
            first_arg = False
            arg_index += 1

        return args_str + ')"'

    def get_arg_handler_processor(self, func_name, op_proto, *, is_tensor_api):
        """
        Generates argument handler processing code for the given function prototype.

        Args:
            func_name (str): The name of the function.
            op_proto (OpProto): Operator prototype instance to generate argument processing for.

        Returns:
            str: Generated argument handler processing code.
        """
        arg_handler_processor = []
        op_args = op_proto.op_args
        for idx, op_arg in enumerate(op_args):
            arg_handler = op_arg.arg_handler
            if arg_handler and arg_handler not in self.black_list_for_arg_handlers:
                func_str = ''.join(word.capitalize() for word in arg_handler.split('_'))
                op_arg_name = op_arg.arg_name
                new_type = self.arg_handler_type_map.get(arg_handler, "Not exist")
                arg_handler_str = self.arg_handler_template.replace(func_str=func_str,
                                                                func_name=func_name,
                                                                op_arg_name=op_arg_name,
                                                                idx=idx,
                                                                new_type=new_type)

                if op_arg.default == "None":
                    arg_handler_str = self.arg_handler_optional_template.replace(idx=idx,
                                                                                 arg_handler_str=arg_handler_str)
                arg_handler_processor.append(arg_handler_str)

        return arg_handler_processor

    @staticmethod
    def get_input_tensor_index(op_proto):
        """
        Get index of input.

        Args:
            op_proto (OpProto): Function prototype to generate dispatch strings for.

        Returns:
            int: Index of input.
        """
        op_name = op_proto.op_class.name
        op_args = op_proto.op_args
        if op_name in K.INPUT_NAME_MAP:
            self_index = [i for i in range(
                len(op_args)) if op_args[i].arg_name == K.INPUT_NAME_MAP[op_name]]
        else:
            self_index = [i for i in range(
                len(op_args)) if op_args[i].arg_name in K.INPUT_ARGS_NAME]
        if len(self_index) != 1:
            raise ValueError(
                f'There must be only one field named \'input\'. But got {len(self_index)} in {op_name}')
        return self_index[0]

    def get_convert_args_str(self, op_proto, is_tensor_api):
        """
        Generates argument convert processing code for the given function prototype.

        Args:
            op_proto (OpProto): Operator prototype instance to generate argument processing for.

        Returns:
            str: Generated argument convert processing code.
        """
        self_index = 0
        if is_tensor_api:
            self_index = self.get_input_tensor_index(op_proto)
        convert_args_str = ""
        arg_basic_convert_template = Template("parse_args.${convert_func}(${index}), ")
        for idx, op_arg in enumerate(op_proto.op_args):
            if is_tensor_api:
                if self_index == idx:
                    convert_args_str += "input_tensor, "
                    continue
            is_optional = is_optional_param(op_arg)
            if op_proto.op_view:
                convert_func = basic_type_convert_str(op_arg.arg_dtype, is_optional)
                if convert_func != "":
                    arg_convert_str = arg_basic_convert_template.replace(convert_func=convert_func,
                                                                         index=idx)
                    convert_args_str += arg_convert_str
                    continue
            arg_convert_template = Template("parse_args.ConvertOptional<${des_type}>(${index}), ") if is_optional \
                else Template("parse_args.Convert<${des_type}>(${index}), ")
            if op_arg.is_type_id:
                arg_type_str = get_input_args_type_str('type', False)
            else:
                arg_type_str = get_input_args_type_str(op_arg.arg_dtype, False)
            convert_args_str += arg_convert_template.replace(index=idx,
                                                             des_type=arg_type_str[:-3])
        return convert_args_str[:-2]


def _is_input_arg(arg_name, op_name):
    res = False
    if op_name in K.INPUT_NAME_MAP and arg_name == K.INPUT_NAME_MAP[op_name]:
        res = True
    elif op_name not in K.INPUT_NAME_MAP and arg_name in K.INPUT_ARGS_NAME:
        res = True
    return res
