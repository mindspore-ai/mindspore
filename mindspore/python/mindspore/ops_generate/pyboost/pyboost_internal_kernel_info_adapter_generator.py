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
This module defines the `PyboostKernelInfoAdapterGenerator` class, which is used to generate
files (`kernel_info_adapter.h`, `internal_kernel_info_adapter.h`, `internal_kernel_info_adapter.cc`)
that contains declarations and definitions for class 'InternalKernelInfoAdapter'.

The class uses templates and operation prototypes to create class declarations based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

from common import template
from common.template import Template
import common.gen_constants as K
from common.gen_utils import save_file
from common.base_generator import BaseGenerator

from .op_template_parser import OpTemplateParser

KERNEL_INFO_ADAPTER_REGISTER = \
"MS_KERNEL_INFO_ADAPTER_REG(${op_name}, Internal${op_name}KernelInfoAdapter, ${op_name}KernelInfoAdapter);\n"


class PyboostKernelInfoAdapterGenerator(BaseGenerator):
    """
    A class to generate `kernel_info_adapter.h`, `internal_kernel_info_adapter.h` and `internal_kernel_info_adapter.cc`
    which contains class declarations and definitions for internal op in Pyboost.
    """

    def __init__(self):
        """Initializes the PyboostKernelInfoAdapterGenerator with the necessary templates."""
        self.kernel_info_adapter_template = template.PYBOOST_KERNEL_INFO_ADAPTER_TEMPLATE
        self.kernel_info_adapter_h_template = template.PYBOOST_KERNEL_INFO_ADAPTER_H_TEMPLATE
        self.internal_kernel_info_adapter_template = template.PYBOOST_INTERNAL_KERNEL_INFO_ADAPTER_TEMPLATE
        self.internal_kernel_info_adapter_h_template = template.PYBOOST_INTERNAL_KERNEL_INFO_ADAPTER_H_TEMPLATE
        self.kernel_info_adapter_single_cpp_template = template.PYBOOST_INTERNAL_KERNEL_INFO_ADAPTER_SINGLE_CPP_TEMPLATE
        self.kernel_info_adapter_cpp_template = template.PYBOOST_INTERNAL_KERNEL_INFO_ADAPTER_CPP_TEMPLATE
        self.kernel_info_adapter_register_template = Template(KERNEL_INFO_ADAPTER_REGISTER)
        self.merged_op_headers_template = Template(
            "#include \"kernel/ascend/internal/pyboost/${operator_name}.h\"\n")

    def generate(self, work_path, op_protos):
        """
        Generates the class declarations and definitions for internal op in Pyboost (`kernel_info_adapter.h`,
        `internal_kernel_info_adapter.h`, `internal_kernel_info_adapter.cc`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse.

        Returns:
            None: The method writes the generated files to the specified directory.
        """
        kernel_info_adapter_list = []
        internal_kernel_info_adapter_list = []
        kernel_info_adapter_cpp_list = []
        kernel_info_adapter_register = []
        merged_op_headers_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            op_parser = OpTemplateParser(op_proto)
            call_args_after_convert, _, _ = op_parser.op_args_converter()
            call_args_with_type = op_parser.parse_call_args_with_types(is_convert=True)
            kernel_info_adapter_list.append(
                self.kernel_info_adapter_template.replace(
                    op_name=op_proto.op_class.name,
                    call_args_with_type=call_args_with_type))
            kernel_info_adapter_list.append(template.NEW_LINE)
            internal_kernel_info_adapter_list.append(
                self.internal_kernel_info_adapter_template.replace(
                    op_name=op_proto.op_class.name,
                    call_args_with_type=call_args_with_type))
            internal_kernel_info_adapter_list.append(template.NEW_LINE)
            kernel_info_adapter_cpp_list.append(
                self.kernel_info_adapter_single_cpp_template.replace(
                    op_name=op_proto.op_class.name,
                    call_args_with_type=call_args_with_type,
                    call_args_after_convert=call_args_after_convert))
            kernel_info_adapter_cpp_list.append(template.NEW_LINE)
            kernel_info_adapter_register.append(
                self.kernel_info_adapter_register_template.replace(op_name=op_proto.op_class.name))
            merged_op_headers_list.append(self.merged_op_headers_template.replace(operator_name=op_proto.op_name))

        if not kernel_info_adapter_list:
            return
        kernel_info_adapter_h_str = self.kernel_info_adapter_h_template.replace(
            kernel_info_adapter_list=kernel_info_adapter_list)
        internal_kernel_info_adapter_h_str = self.internal_kernel_info_adapter_h_template.replace(
            internal_kernel_info_adapter_list=internal_kernel_info_adapter_list,
            merged_op_headers=merged_op_headers_list
        )
        kernel_info_adapter_cpp_str = self.kernel_info_adapter_cpp_template.replace(
            kernel_info_adapter_cpp_list=kernel_info_adapter_cpp_list,
            kernel_info_adapter_register=kernel_info_adapter_register
        )

        self._save_files(work_path, kernel_info_adapter_h_str,
                         internal_kernel_info_adapter_h_str, kernel_info_adapter_cpp_str)

    @staticmethod
    def _save_files(work_path, kernel_info_adapter, kernel_info_adapter_h, kernel_info_adapter_cpp):
        """
        Save the generated files.
        """
        save_path = os.path.join(work_path, K.MS_INTERNAL_PYBOOST_GEN_PATH)
        save_file(save_path, "kernel_info_adapter.h", kernel_info_adapter)
        save_file(save_path, "internal_kernel_info_adapter.h", kernel_info_adapter_h)
        save_file(save_path, "internal_kernel_info_adapter.cc", kernel_info_adapter_cpp)
