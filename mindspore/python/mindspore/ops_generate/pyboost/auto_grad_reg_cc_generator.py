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
This module provides a generator class for creating C++ header files for AutoGrad registration functionality.
"""

import os

from common import template
from common.template import Template
import common.gen_constants as K
from common.gen_utils import save_file
from common.base_generator import BaseGenerator
from pyboost.pyboost_utils import is_optional_param, get_input_dtype, get_output_dtype


class AutoGradRegHeaderGenerator(BaseGenerator):
    """
    Generates C++ header files for the AutoGrad registration functionality based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the AutoGrad registration header generator with templates for code generation.
        """
        self.AUTO_GRAD_REG_H_TEMPLATE = template.AUTO_GRAD_REG_H_TEMPLATE
        self.op_type_enum_template = Template("k${class_name} = ${enum_val},\n")
        self.op_grad_func_template = Template("using ${class_name}GradFunc = std::function<void(${grad_func_args})>;")
        self.op_grad_func_obj_template = Template("${class_name}GradFunc ${class_name}GradFuncObj;")
        self.op_grad_func_args_template = Template(
            "const kernel::pyboost::OpPtr &, ${input_tensor_prt_args}"
        )
        self.op_view_grad_func_args_template = Template(
            "${output_tensor_prt_args}, ${input_tensor_prt_args}"
        )

    def generate(self, work_path, op_protos):
        """
        Generate the AutoGrad registration header file.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the header.
        """
        op_type_enum_list = []
        op_grad_func_list = []
        op_grad_func_obj_list = []
        index = 0
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            op_type_enum_list.append(self.op_type_enum_template.replace(class_name=op_proto.op_class.name,
                                                                        enum_val=index))
            # the backward func of flatten_ext and t_ext are implemented by other view ops, just continue
            if op_proto.op_view and not op_proto.bprop_expander:
                continue
            grad_func_args_with_type_str = self._get_grad_func_args_with_type_str(op_proto)
            op_grad_func_list.append(
                self.op_grad_func_template.replace(class_name=op_proto.op_class.name,
                                                   grad_func_args=grad_func_args_with_type_str))
            op_grad_func_obj_list.append(self.op_grad_func_obj_template.replace(class_name=op_proto.op_class.name))
            index += 1

        pyboost_func_h_str = self.AUTO_GRAD_REG_H_TEMPLATE.replace(op_enum=op_type_enum_list,
                                                                   op_grad_func=op_grad_func_list,
                                                                   op_grad_func_obj=op_grad_func_obj_list)

        save_path = os.path.join(work_path, K.MS_PYBOOST_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "auto_grad_op_reg.h"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_grad_func_args_with_type_str(self, op_proto):
        """
        Get the gradient function arguments with type information.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: A string of input tensor pointer arguments with types.
        """
        input_tensor_prt_args_str = ""
        for op_arg in op_proto.op_args:
            is_optional = is_optional_param(op_arg)
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional, op_proto.op_view)
            input_tensor_prt_args_str += f"const {input_dtype} &, "
        input_tensor_prt_args_str = input_tensor_prt_args_str.rstrip(', ')
        if not op_proto.op_view:
            return self.op_grad_func_args_template.replace(input_tensor_prt_args=\
                                                           input_tensor_prt_args_str)
        # for view operators, the output is tensor or vector<tensor>
        if len(op_proto.op_returns) != 1:
            raise ValueError(f"the output of {op_proto.op_name} is not tensor,",
                             "tuple[tensor] or list[tensor], which is not not as expected")
        output_dtype = get_output_dtype(op_proto.op_returns[0].arg_dtype)
        output_tensor_prt_args_str = f"const {output_dtype} &"
        return self.op_view_grad_func_args_template.replace(input_tensor_prt_args=input_tensor_prt_args_str,
                                                            output_tensor_prt_args=output_tensor_prt_args_str)
