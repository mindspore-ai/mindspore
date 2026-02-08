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
from pyboost.pyboost_utils import get_convert_type_str, is_optional_param

from .op_template_parser import OpTemplateParser


class PyboostFunctionsGenerator(BaseGenerator):
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
        self.convert_template = Template("auto $arg_name = converter.${convert_func}(args, $arg_index);\n")
        self.PYBOOST_REGISTRY_BODY_CC_TEMPLATE = template.PYBOOST_REGISTRY_BODY_CC_TEMPLATE
        self.REGISTER_DEFINE_TEMPLATE = template.REGISTER_DEFINE_TEMPLATE
        self.REGISTER_TEMPLATE = template.REGISTER_TEMPLATE
        self.PYBOOST_REGISTRY_CC_TEMPLATE = template.PYBOOST_REGISTRY_CC_TEMPLATE
        self.TENSOR_FUNC_CLASS_REG = template.TENSOR_FUNC_CLASS_REG
        self.OP_DEF_INC_HEAD_TEMPLATE = template.OP_DEF_INC_HEAD_TEMPLATE
        self.MARK_SIDE_EFFECT_STR = "PyNativeAlgo::PyBoost::MarkSideEffect(PyList_GetItem(args, 0));"
        self.pyboost_api_body_template = template.PYBOOST_API_BODY_CC_TEMPLATE

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
        pyboost_registry_body_str = ''
        pyboost_func_pybind_def = ''
        pyboost_func_include_headers_str = ''
        ops_inc_head_set = set()
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue

            pyboost_registry_body_str += self._get_pyboost_registry_body_str(op_proto)
            pyboost_registry_body_str += template.NEW_LINE + template.NEW_LINE

            op_parser = OpTemplateParser(op_proto)
            pyboost_op_name = op_parser.get_pyboost_name()
            pyboost_func_name = op_parser.get_pyboost_func_name()
            pyboost_func_pybind_def += self.REGISTER_DEFINE_TEMPLATE.replace(
                pyboost_op_name=pyboost_op_name,
                pyboost_cfunc_name=pyboost_func_name,
                class_name=op_proto.op_class.name)
            pyboost_func_include_headers_str +=\
                self.pyboost_func_include_header_template.replace(operator_name=op_proto.op_name)
            ops_inc_head_set.add(self.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_proto.op_class.name[0].lower()))
        register_func_str = self.REGISTER_TEMPLATE.replace(register_func=pyboost_func_pybind_def)
        function_class_register = self._get_function_class_register(op_protos)
        pyboost_registry_file \
            = self.PYBOOST_REGISTRY_CC_TEMPLATE.replace(ops_inc=list(sorted(ops_inc_head_set)),
                                                        include_op_header=pyboost_func_include_headers_str,
                                                        function_body=pyboost_registry_body_str,
                                                        register_function_body=register_func_str,
                                                        function_class_register=function_class_register)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_registry.cc"
        save_file(save_path, file_name, pyboost_registry_file)

        pyboost_function_base_str = self.get_pyboost_api_body_str(op_protos)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_api.cc"
        save_file(save_path, file_name, pyboost_function_base_str)

    def get_pyboost_api_body_str(self, op_protos):
        """
        Generates pyboost function base string.

        Args:
            op_protos (list): A list of tensor op prototypes.

        Returns:
            str: pyboost function base string.
        """
        pyboost_api_cc_tpl = template.PYBOOST_API_CC_TEMPLATE
        pyboost_api_body_str = ''
        ops_inc_head_set = set()
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            op_parser = OpTemplateParser(op_proto)
            op_pyboost_func_name = op_parser.get_pyboost_func_name()
            op_def_name_str = op_parser.get_op_def_name_str()
            parser_body_str = self._generate_parser_func(op_proto)
            op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
            side_effect_str = self._generate_mark_side_effect_str(op_proto)
            pyboost_api_body_str += self.pyboost_api_body_template.replace(func_name=op_pyboost_func_name,
                                                                           op_def_name=op_def_name_str,
                                                                           parser_body=parser_body_str,
                                                                           class_name=op_proto.op_class.name,
                                                                           op_args=op_args_str,
                                                                           mark_side_effect=side_effect_str)

            ops_inc_head_set.add(self.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_proto.op_class.name[0].lower()))

        return pyboost_api_cc_tpl.replace(pyboost_api_body=pyboost_api_body_str)

    def _get_pyboost_registry_body_str(self, op_proto):
        op_parser = OpTemplateParser(op_proto)
        op_pyboost_func_name = op_parser.get_pyboost_func_name()
        op_def_name_str = op_parser.get_op_def_name_str()
        parser_body_str = self._generate_parser_func(op_proto)
        op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
        registry_body_tpl = self.PYBOOST_REGISTRY_BODY_CC_TEMPLATE
        return registry_body_tpl.replace(func_name=op_pyboost_func_name,
                                         op_def_name=op_def_name_str,
                                         parser_body=parser_body_str,
                                         class_name=op_proto.op_class.name,
                                         op_args=op_args_str)

    def _get_function_class_register(self, op_protos) -> str:
        """
        Generates a function class registration string for tensor functions.

        Args:
            op_protos (list): A list of tensor op prototypes.

        Returns:
            str: A concatenated string representing the registration information for tensor
                 function classes.
        """
        function_class_register = ''
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            class_name, op_name = op_proto.op_class.name, op_proto.op_name
            function_class_register += self.TENSOR_FUNC_CLASS_REG.replace(class_name=class_name,
                                                                          op_name=op_name)
        return function_class_register

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

    def _generate_mark_side_effect_str(self, op_proto: OpProto) -> str:
        """
        Generates the mark side effect str for the inplace operator.

        Args:
            op_proto (OpProto): The operator prototype containing the argument information.

        Returns:
            str: The generated mark side effect flag as a string.
        """
        if op_proto.op_inplace or op_proto.op_view:
            return self.MARK_SIDE_EFFECT_STR
        return ""

    def get_pyboost_registry_body_cc_tpl(self, op_proto: OpProto):
        return self.PYBOOST_REGISTRY_BODY_CC_TEMPLATE
