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
This module defines the PyboostFunctionsPyGenerator class for generating Python bindings for PyBoost functions.

The PyboostFunctionsPyGenerator class processes operator prototypes and generates Python functions
that correspond to the PyBoost operations defined in the operator prototypes. It handles the necessary
argument processing and includes appropriate documentation descriptions.
"""

import os

from common import template
import common.gen_constants as K
from common.gen_utils import save_file
from common.op_proto import OpProto
from common.base_generator import BaseGenerator


class PyboostFunctionsPyGenerator(BaseGenerator):
    """
    Generates Python bindings for PyBoost functions.

    This class is responsible for creating Python function definitions that correspond to the PyBoost
    operations defined in operator prototypes. It generates a Python file that includes necessary function
    definitions and their descriptions.
    """

    def __init__(self):
        """Initializes the PyboostFunctionsPyGenerator with required templates."""
        self.IMPORT_PYBOOST_FUNC_HEADER = template.IMPORT_PYBOOST_FUNC_HEADER
        self.PYBOOST_PY_FUNC_TEMPLATE = template.PYBOOST_PY_FUNC_TEMPLATE

    def generate(self, work_path, op_protos, doc_data):
        """
        Generates the Python file containing PyBoost function definitions.

        This method processes the provided operator prototypes (`op_protos`), generates Python function
        definitions for each operator that meets the specified conditions, and saves the generated content
        to a Python file.

        Args:
            work_path (str): The directory path where the generated Python file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.
            doc_data (dict): A dictionary containing documentation data for the operators.

        Returns:
            None
        """
        gen_py = ''
        op_desc_dict = self._get_op_description_dict(doc_data)
        for op_proto in op_protos:
            # check if the operator is in pyboost scenario
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if op_proto.op_function.disable:
                continue
            if not op_proto.op_function.name.endswith("_ext") and not op_proto.op_name.endswith("_ext"):
                continue

            description = op_desc_dict.get(op_proto.op_name)
            func_args, input_args = self._process_args(op_proto.op_args)
            func_name, func_impl_name = self._get_func_impl_name(op_proto)
            gen_py += self.PYBOOST_PY_FUNC_TEMPLATE.replace(func_name=func_name,
                                                            description=description,
                                                            func_args=func_args,
                                                            input_args=input_args,
                                                            func_impl_name=func_impl_name)
        py_header = template.PY_LICENSE_STR + self.IMPORT_PYBOOST_FUNC_HEADER
        save_file(os.path.join(work_path, K.PY_AUTO_GEN_PATH), "gen_extend_func.py", py_header + gen_py)

    def _get_op_description_dict(self, doc_yaml_data):
        """
        Constructs a dictionary mapping operator names to their descriptions.

        Args:
            doc_yaml_data (dict): A dictionary containing YAML data for operator documentation.

        Returns:
            dict: A dictionary mapping operator names to their descriptions.
        """
        op_description_dict = {}
        for operator_name, operator_desc in doc_yaml_data.items():
            desc = operator_desc.get("description")
            op_description_dict[operator_name] = desc
        return op_description_dict

    def _process_args(self, op_args):
        """
        Processes the operator arguments to generate function argument strings.

        Args:
            op_args (list): A list of operator arguments to be processed.

        Returns:
            tuple: A tuple containing:
                - func_args (list): A list of formatted function argument strings.
                - input_args (list): A list of corresponding input argument names.
        """
        func_args = []
        input_args = []
        for op_arg in op_args:
            arg_handler = op_arg.arg_handler
            arg_name = op_arg.arg_name
            input_arg = arg_name
            if arg_handler not in ('', 'dtype_to_type_id'):
                input_arg = 'converted_' + arg_name
            input_args.append(input_arg)
            default_value = op_arg.default
            if default_value is not None:
                default_value = '=' + str(default_value)
                func_args.append(arg_name + default_value)
            else:
                func_args.append(arg_name)
        return func_args, input_args

    def _get_func_impl_name(self, op_proto: OpProto):
        """
        Retrieves the implementation function name based on the operator prototype.

        Args:
            op_proto (OpProto): The operator prototype containing function name information.

        Returns:
            tuple: A tuple containing:
                - func_name (str): The name of the function.
                - func_impl_name (str): The implementation name of the function.
        """
        func_name = op_proto.op_name if op_proto.op_function.name == '' \
            else op_proto.op_function.name
        if func_name.endswith("_ext"):
            func_name = func_name[:-4]
        func_impl_name = func_name
        if func_name.endswith("_"):
            func_impl_name = func_name[:-1]
        return func_name, func_impl_name
