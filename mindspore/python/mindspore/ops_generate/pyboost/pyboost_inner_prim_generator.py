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
This module defines the PyboostInnerPrimGenerator class, which is responsible for generating Python primitive
wrappers for Pyboost operations. The generator constructs Python function definitions based on operator prototypes,
generates necessary import statements, and writes the generated content into Python source files.

The primary functionality is to take operator prototypes, extract relevant fields, and create Python function wrappers
that can be used to call the Pyboost primitive implementations.
"""

import os
from common import template
import common.gen_constants as K
from common.gen_utils import save_file

from common.op_proto import OpProto
from common.base_generator import BaseGenerator

from .op_template_parser import OpTemplateParser
from .pyboost_utils import get_pyboost_arg_handlers_black_list

class PyboostInnerPrimGenerator(BaseGenerator):
    """
    PyboostInnerPrimGenerator is responsible for generating Python primitive wrappers for Pyboost operators.

    This class processes operator prototypes (`op_protos`) to generate Python function implementations. It handles the
    inclusion of necessary headers, processes operator arguments, and creates Python functions that wrap Pyboost
    primitives.

    Attributes:
        IMPORT_PYBOOST_PRIM_HEADER (Template): Template for importing Pyboost primitive headers.
        PYBOOST_PY_FUNC_IMPORT_HEADER (Template): Template for importing Python functions related to Pyboost.
        PYTHON_PRIM_TEMPLATE (Template): Template for generating Python primitive functions.
    """

    def __init__(self):
        """
        Initializes the PyboostInnerPrimGenerator class.

        This constructor sets up the required templates for generating import headers, Python function imports,
        and Python primitive function wrappers.
        """
        self.IMPORT_PYBOOST_PRIM_HEADER = template.IMPORT_PYBOOST_PRIM_HEADER
        self.PYBOOST_PY_FUNC_IMPORT_HEADER = template.PYBOOST_PY_FUNC_IMPORT_HEADEAR
        self.PYTHON_PRIM_TEMPLATE = template.PYTHON_PRIM_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates Python wrappers for Pyboost primitives and writes them to a Python source file.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information such as
        class names, arguments, and handlers. It constructs Python function wrappers for the Pyboost primitives and
        generates the required import statements. The generated Python code is saved to a specified path.

        Args:
            work_path (str): The file path where the generated Python file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.

        Returns:
            None
        """
        gen_py = ''
        gen_header = template.PY_LICENSE_STR + self.IMPORT_PYBOOST_PRIM_HEADER
        for op_proto in op_protos:
            # only process pyboost enabled scenario
            if op_proto.op_dispatch is None:
                continue
            if not op_proto.op_dispatch.enable:
                continue
            op_parser = OpTemplateParser(op_proto)
            if not op_parser.has_prim_init():
                continue

            gen_header += self.PYBOOST_PY_FUNC_IMPORT_HEADER.replace(class_name=op_proto.op_class.name)
            input_args, process_func, processed_args = self._get_fields_for_prim_tpl(op_proto)
            gen_py += self.PYTHON_PRIM_TEMPLATE.replace(class_name=op_proto.op_class.name,
                                                        input_args=input_args,
                                                        process_func=process_func,
                                                        func_impl_name=op_proto.op_name,
                                                        processed_args=processed_args)

        save_file(os.path.join(work_path, K.PY_AUTO_GEN_PATH), "pyboost_inner_prim.py", gen_header + gen_py)

    def _get_fields_for_prim_tpl(self, op_proto: OpProto):
        """
        Extracts the necessary fields for the primitive template from the operator prototype.

        This method processes the arguments of the operator prototype and generates the input arguments, the function
        that handles argument processing, and the processed arguments list, which will be used in the final Python
        function definition.

        Args:
            op_proto (OpProto): The operator prototype from which the argument data will be extracted.

        Returns:
            tuple: A tuple containing three return values required for the primitive template to be generated:
                - input_args (list): List of input argument names for the Python function.
                - process_func (str): String representing the argument processing logic for the function.
                - processed_args (list): List of processed argument names used in the function call.
        """
        args = op_proto.op_args
        operator_name = op_proto.op_name

        input_args = []
        process_func = ''
        processed_args = []

        for arg in args:
            arg_name = arg.arg_name
            arg_handler = arg.arg_handler
            processed_arg = arg_name
            if arg_handler not in ('', 'dtype_to_type_id') and arg_handler not in get_pyboost_arg_handlers_black_list():
                process_func += \
                    f"""converted_{arg_name} = {arg_handler}('{operator_name}', '{arg_name}', {arg_name})\n"""
                processed_arg = 'converted_' + arg_name
            input_args.append(arg_name)
            processed_args.append(processed_arg)

        return input_args, process_func, processed_args
