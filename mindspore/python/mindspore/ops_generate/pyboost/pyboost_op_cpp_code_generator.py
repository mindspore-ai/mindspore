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
This module defines several classes and functions for generating C++ code for PyBoost operations,
including function headers, source files, and registration code. It handles the generation of code
for different devices (Ascend, CPU, GPU) and manages residual files associated with operator prototypes.
"""

import os
import re

from common import template
from common import gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.op_proto import OpProto
from common.base_generator import BaseGenerator

from .pyboost_utils import is_cube, AclnnUtils, get_return_type, merge_strings_by_chunk_size, is_op_multi_output, \
    chunk_list
from .op_template_parser import OpTemplateParser


def check_no_basic_int_type(op_args):
    for arg in op_args:
        if arg.arg_dtype in ["tuple[int]", "list[int]", "int"]:
            return False
    return True


def get_inplace_indices(op_proto):
    """
    Extracts the indices of inplace arguments from the operation prototype.

    Args:
        op_proto (OpProto): The operator prototype containing argument information.

    Returns:
        list: A list of indices for inplace arguments.
    """
    inplace_args = []
    for arg in op_proto.op_returns:
        if arg.inplace != '':
            inplace_args.append(arg.inplace)
    input_args = [arg.arg_name for arg in op_proto.op_args]
    inplace_indices = [input_args.index(arg) for arg in inplace_args]
    return inplace_indices


class PyboostCommonOpHeaderGenerator(BaseGenerator):
    """
    Generates common C++ headers for PyBoost operations.

    This class processes operator prototypes and generates header files containing function definitions
    based on templates provided. It specifically generates the headers that define operations for PyBoost.
    """

    def __init__(self):
        self.pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE
        self.pyboost_basic_type_func_template = Template(
            'virtual ${return_type} Call(${call_args_with_type}) {' \
            '\n   MS_EXCEPTION(NotImplementedError) << "Basic type func not implemented";' \
            '\n};'
        )

    def generate(self, work_path, op_protos):
        """
        Generates header files for the provided operator prototypes.

        Args:
            work_path (str): The directory path where the header files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        for op_proto in op_protos:
            if is_op_multi_output(op_proto.op_returns):
                print(op_proto.op_name)
            if op_proto.op_dispatch is None:
                continue
            if op_proto.op_view:
                continue
            op_parser = OpTemplateParser(op_proto)
            op_name_str = op_proto.op_class.name
            if op_proto.op_view and not check_no_basic_int_type(op_proto.op_args):
                call_args_with_type = op_parser.parse_call_args_with_types(basic_type=True)
            else:
                call_args_with_type = op_parser.parse_call_args_with_types(basic_type=False)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            output_is_tuple = "bool output_is_tuple() const override { return true; }" \
                if is_op_multi_output(op_proto.op_returns) else ''
            pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE.replace(op_name=op_name_str,
                                                                                     op_name_upper=op_name_str.upper(),
                                                                                     call_args=call_args_with_type,
                                                                                     return_type=cpp_func_return,
                                                                                     output_is_tuple=output_is_tuple)
            save_path = os.path.join(work_path, f"{K.MS_PYBOOST_BASE_HEADER_PATH}/auto_generate/")
            file_name = f"{op_proto.op_name}.h"
            save_file(save_path, file_name, pyboost_op_header_str)


class PyboostOpHeaderGenerator(BaseGenerator):
    """
    Generates device-specific C++ headers for PyBoost operations.

    This class generates header files for different devices (Ascend, CPU, GPU) and defines
    the operation functions accordingly.
    """

    def __init__(self, device):
        """
        Initializes the PyboostOpHeaderGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        template_dict = {"ascend": template.PYBOOST_ASCEND_OP_HEADER_TEMPLATE,
                         "gpu": template.PYBOOST_GPU_OP_HEADER_TEMPLATE,
                         "cpu": template.PYBOOST_CPU_OP_HEADER_TEMPLATE}
        if device not in template_dict:
            raise ValueError(
                f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_OP_HEADER_TEMPLATE = template_dict[device]
        if device == "ascend":
            self.code_generate_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
        else:
            self.code_generate_path = f"{K.MS_OPS_KERNEL_PATH}/{device}/pyboost/auto_generate/"
        self.hccl_code_generate_path = "mindspore/ops/kernel/ascend/hccl/pyboost/auto_generate/"
        self.device = device

    def generate(self, work_path, op_protos):
        """
        Generates header files for the provided operator prototypes based on the device.

        Args:
            work_path (str): The directory path where the header files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if op_proto.op_view:
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            is_ascend_comm_op = op_proto.op_dispatch.is_comm_op and self.device == 'ascend'
            op_parser = OpTemplateParser(op_proto)
            op_name_str = op_proto.op_class.name
            if op_proto.op_view and not check_no_basic_int_type(op_proto.op_args):
                call_args_with_type = op_parser.parse_call_args_with_types(basic_type=True)
            else:
                call_args_with_type = op_parser.parse_call_args_with_types(basic_type=False)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            pyboost_op_str = self.PYBOOST_OP_HEADER_TEMPLATE.replace(op_name=op_name_str,
                                                                     op_name_upper=op_name_str.upper(),
                                                                     operator_name=op_proto.op_name,
                                                                     call_args_with_type=call_args_with_type,
                                                                     return_type=cpp_func_return)
            save_path = os.path.join(work_path, self.code_generate_path if not is_ascend_comm_op \
                                     else self.hccl_code_generate_path)
            file_name = f"{op_proto.op_name}.h"
            save_file(save_path, file_name, pyboost_op_str)


class PyboostInternalOpHeaderGenerator(BaseGenerator):
    """
    Generates C++ headers for PyBoost internal operations.

    This class generates header files for Ascend and defines the operation functions accordingly.
    """

    def __init__(self, device):
        """
        Initializes the PyboostOpHeaderGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu), currently only support ascend.

        Raises:
            ValueError: If the device is not supported.
        """
        if device != 'ascend':
            raise ValueError(
                f"Currently, only support 'ascend' for internal operations, {device} is not supported.")
        self.pyboost_internal_op_header_template = template.PYBOOST_ASCEND_INTERNAL_OP_HEADER_TEMPLATE
        if device == "ascend":
            self.code_generate_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/internal/auto_generate/"
        else:
            self.code_generate_path = f"{K.MS_OPS_KERNEL_PATH}/{device}/pyboost/internal/auto_generate/"
        self.device = device

    def generate(self, work_path, op_protos):
        """
        Generates header files for the provided operator prototypes based on the device.

        Args:
            work_path (str): The directory path where the header files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            op_parser = OpTemplateParser(op_proto)
            op_name_str = op_proto.op_class.name
            call_args_with_type = op_parser.parse_call_args_with_types()
            cpp_func_return = _generate_cpp_func_return(op_proto)

            pyboost_op_str = self.pyboost_internal_op_header_template.replace(
                op_name=op_name_str,
                op_name_upper=op_name_str.upper(),
                operator_name=op_proto.op_name,
                call_args_with_type=call_args_with_type,
                return_type=cpp_func_return)

            save_path = os.path.join(work_path, self.code_generate_path)
            file_name = f"{op_proto.op_name}.h"
            save_file(save_path, file_name, pyboost_op_str)


class PyboostOpCppGenerator:
    """
    Generates C++ source files for PyBoost operations.

    This class generates the implementation of operations for different devices, handling function calls
    and registering custom kernels as necessary.
    """

    def __init__(self, device):
        """
        Initializes the PyboostOpCppGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_ASCEND_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
            self.device_reg_str = "Ascend"
        elif device == 'cpu':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_CPU_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
            self.device_reg_str = "CPU"
        elif device == 'gpu':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_GPU_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
            self.device_reg_str = "GPU"
        else:
            raise ValueError(
                f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_REG_OP_TEMPLATE = Template('MS_REG_PYBOOST_OP(${device}, ${op_name});' \
                                                '${register_custom_kernel}')
        self.PYBOOST_CUSTOMIZE_CALL_TEMPLATE = PYBOOST_CUSTOMIZE_CALL_TEMPLATE
        self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = PYBOOST_SINGLE_OP_HEADER_TEMPLATE
        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.PYBOOST_SINGLE_HCLL_OP_HEADER_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_HCLL_OP_HEADER_TEMPLATE
        self.gen_path = gen_path
        self.device = device

    def generate_customize_op_cpp_code(self, op_protos, merge_op_header, merge_op_function, merge_op_inc,
                                       merge_op_hccl_header=None, merge_op_hccl_function=None, merge_op_hccl_inc=None):
        """
        Generate C++ code for PyBoost operations using the provided operation prototypes.

        This method processes a list of operation prototypes, generates customized function call
        implementations, and updates the merged headers and functions for the specified device.

        Args:
            op_protos (list): A list of operation prototypes to process. Each prototype contains
                              metadata about the operation, including dispatch settings and arguments.
            merge_op_header (list): A list to store the generated C++ header code for operations.
            merge_op_function (list): A list to store the generated C++ source code for operations.
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if op_proto.composite:
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'default':
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            is_ascend_comm_op = op_proto.op_dispatch.is_comm_op and self.device == 'ascend'
            op_parser = OpTemplateParser(op_proto)
            call_args = OpTemplateParser.parse_original_call_args(op_proto.op_args)
            if op_proto.op_view and not check_no_basic_int_type(op_proto.op_args):
                call_args_with_type = op_parser.parse_call_args_with_types(True)
            else:
                call_args_with_type = op_parser.parse_call_args_with_types()
            _, call_func_outputs = op_parser.generate_pyboost_outputs()
            operator_name = op_proto.op_name
            op_name_str = op_proto.op_class.name
            check_inplace_func = ''
            for arg in op_proto.op_returns:
                if arg.inplace != '':
                    check_inplace_func = f'ThrowExpectionWhenInternalOverlap({arg.inplace}_tensor);'
                    break
            inplace_indices = get_inplace_indices(op_proto)
            inplace_indices_str = ', '.join(str(i) for i in inplace_indices)
            call_impl = self.PYBOOST_CUSTOMIZE_CALL_TEMPLATE.replace(
                call_args=call_args,
                return_values=call_func_outputs,
                inplace_indices=inplace_indices_str,
                customize_func=getattr(
                    op_proto.op_dispatch, self.device) + "Customize",
                check_expression=check_inplace_func,
            )
            if is_ascend_comm_op and ((merge_op_hccl_header is None) or (merge_op_hccl_function is None)):
                raise ValueError(f"merge_op_hccl_header and merge_op_hccl_function \
must be provided for comm op {operator_name}")

            if is_ascend_comm_op:
                customize_include = \
                    f'#include "mindspore/ops/kernel/ascend/hccl/pyboost/{operator_name.lower()}.h"\n'
            else:
                if self.device == 'ascend':
                    customize_include = (
                        f'#include "{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/customize/'
                        f'{operator_name.lower()}.h"\n'
                    )
                else:
                    customize_include = \
                        f'#include "{K.MS_OPS_KERNEL_PATH}/{self.device}/pyboost/customize/{operator_name.lower()}.h"\n'

            register_custom = self._get_register_custom_kernel(op_proto)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            op_register = self.PYBOOST_REG_OP_TEMPLATE.replace(op_name=op_name_str,
                                                               device=self.device_reg_str,
                                                               register_custom_kernel=register_custom)
            if is_ascend_comm_op:
                merge_op_hccl_header.append(
                    self.PYBOOST_SINGLE_HCLL_OP_HEADER_TEMPLATE.replace(operator_name=operator_name,
                                                                        customize_include=customize_include))
                merge_op_hccl_function.append(
                    self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_name_str,
                                                                   call_args_with_type=call_args_with_type,
                                                                   return_type=cpp_func_return, call_impl=call_impl,
                                                                   op_register=op_register,
                                                                   device=self.device_reg_str))
                merge_op_hccl_inc.append(op_name_str)
            else:
                merge_op_header.append(
                    self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE.replace(operator_name=operator_name,
                                                                   customize_include=customize_include))
                merge_op_function.append(
                    self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_name_str,
                                                                   call_args_with_type=call_args_with_type,
                                                                   return_type=cpp_func_return, call_impl=call_impl,
                                                                   op_register=op_register,
                                                                   device=self.device_reg_str))
                merge_op_inc.append(op_name_str)

    def _get_register_custom_kernel(self, op_proto: OpProto):
        """
        Generates the registration code for custom kernels based on the device.

        Args:
            op_proto (OpProto): The operator prototype to generate registration for.

        Returns:
            str: The registration code for the custom kernel.
        """
        if self.device == 'ascend':
            register_custom_kernel = ''
        elif self.device == 'cpu':
            register_custom_kernel = f"MS_REG_PYBOOST_CPU_CUSTOM_KERNEL({op_proto.op_class.name});"
        elif self.device == 'gpu':
            register_custom_kernel = f"MS_REG_PYBOOST_GPU_CUSTOM_KERNEL({op_proto.op_class.name});"
        else:
            raise ValueError(
                f"Device must be ascend, gpu, or cpu, {self.device} is not supported")
        return register_custom_kernel


class PyboostViewOpCppGenerator:
    """
    Generates C++ source files for view operations in PyBoost.

    This class handles the generation of source files for view operations, which have special handling
    compared to regular operations.
    """

    def __init__(self, device):
        """
        Initializes the PyboostViewOpCppGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
            self.device_reg_str = "Ascend"
        elif device == 'cpu':
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
            self.device_reg_str = "CPU"
        elif device == 'gpu':
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
            self.device_reg_str = "GPU"
        else:
            raise ValueError(
                f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_REG_OP_TEMPLATE = Template('MS_REG_PYBOOST_OP(${device}, ${op_name});' \
                                                '${register_custom_kernel}')
        self.PYBOOST_VIEW_CALL_TEMPLATE = PYBOOST_VIEW_CALL_TEMPLATE
        self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = PYBOOST_SINGLE_OP_HEADER_TEMPLATE
        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.gen_path = gen_path
        self.device = device


class AclnnOpCppCodeGenerator:
    """
    Generates C++ source files for ACLNN operations in PyBoost.

    This class handles the generation of source files for operations that utilize the ACLNN framework,
    including customized calls and tensor management.

    Attributes:
        PYBOOST_CALL_TEMPLATE (Template): Template for generating ACLNN operation calls.
        PYBOOST_OP_SOURCE_TEMPLATE (Template): Template for generating operation source files.
        gen_path (str): Path for saving the generated C++ source files.
        device (str): The target device (ascend, cpu, or gpu).
    """

    def __init__(self, device):
        """
        Initializes the AclnnOpCppCodeGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_ASCEND_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
            self.device_reg_str = "Ascend"
        elif device == 'cpu':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_CPU_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
            self.device_reg_str = "CPU"
        elif device == 'gpu':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_GPU_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
            self.device_reg_str = "GPU"
        else:
            raise ValueError(
                f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_REG_OP_TEMPLATE = Template('MS_REG_PYBOOST_OP(${device}, ${op_name});' \
                                                '${register_custom_kernel}')
        self.PYBOOST_CALL_TEMPLATE = PYBOOST_CALL_TEMPLATE
        if device == "ascend":
            self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.Template(
                '#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/${operator_name}.h"\n'
            )
        else:
            self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.Template(
                '#include "kernel/${device}/pyboost/auto_generate/${operator_name}.h"\n'
            )

        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.gen_path = gen_path
        self.device = device

    def generate_aclnn_op_cpp_code(self, op_protos, merge_op_header, merge_op_function, ascend_merge_op_inc):
        """
        Generate C++ code for ACLNN operations in PyBoost.

        This method processes a list of operation prototypes (`op_protos`) and generates C++ code
        for aclnn operations. The method filters the operation
        prototypes based on their dispatch and view settings, and then uses templates and metadata
        to generate the necessary implementation and header files.

        Args:
            op_protos (list): A list of operation prototypes. Each prototype includes metadata
                              such as operation name, dispatch settings, view attributes, and arguments.
            merge_op_header (list): A list to store the generated C++ header code for ACLNN operations.
            merge_op_function (list): A list to store the generated C++ source code for ACLNN operations.
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if op_proto.composite:
                continue
            if getattr(op_proto.op_dispatch, self.device) != 'default':
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            if op_proto.op_view:
                continue

            op_parser = OpTemplateParser(op_proto)
            aclnn_name = AclnnUtils.get_aclnn_interface(op_proto.op_class.name)

            call_args_tensor = op_parser.get_call_args_tensor()
            create_input_address = AclnnOpCppCodeGenerator._generate_create_input_address(
                op_parser)
            malloc_inputs = AclnnOpCppCodeGenerator._generate_malloc_input(op_parser)
            op_outputs, call_func_outputs = op_parser.generate_pyboost_outputs()
            get_inputs_kernel_tensors = AclnnOpCppCodeGenerator._generate_get_inputs_kernel_tensors(
                op_parser)

            cube_math_type, get_cube_math_type = '', ''
            if self.device == 'ascend' and is_cube(op_proto.op_class.name):
                get_cube_math_type = '// cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION\n'
                get_cube_math_type += "auto cube_math_type = GetCubeMathType();"
                cube_math_type = ', cube_math_type'

            real_output = ', ' + op_outputs \
                if _generate_inplace_process_cpp_code(op_proto) == '' else ''

            cast_input_code, real_call_args_tensor = AclnnOpCppCodeGenerator._generate_tensor_cpu_cast_input_code(
                op_parser)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            _, tensor_list_convert, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
            call_args_after_convert, value_tuple_convert, const_number_convert = op_parser.op_args_converter()
            call_args = OpTemplateParser.parse_original_call_args(op_proto.op_args)
            if op_proto.op_view and not check_no_basic_int_type(op_proto.op_args):
                call_args_with_type = op_parser.parse_call_args_with_types(True)
            else:
                call_args_with_type = op_parser.parse_call_args_with_types()
            inplace_process = _generate_inplace_process_cpp_code(op_proto)
            inplace_indices = get_inplace_indices(op_proto)
            inplace_indices_str = ', '.join(str(i) for i in inplace_indices)
            call_impl = self.PYBOOST_CALL_TEMPLATE.replace(aclnn_name=aclnn_name,
                                                           call_args=call_args,
                                                           call_tensors=call_args_tensor,
                                                           value_tuple_convert=value_tuple_convert,
                                                           const_number_convert=const_number_convert,
                                                           create_input_address=create_input_address,
                                                           tensor_list_convert=tensor_list_convert,
                                                           call_args_with_tensor=call_args_with_tensor,
                                                           malloc_inputs=malloc_inputs,
                                                           get_inputs_kernel_tensors=get_inputs_kernel_tensors,
                                                           get_cube_math_type=get_cube_math_type,
                                                           cube_math_type=cube_math_type,
                                                           real_call_args=call_args_after_convert,
                                                           return_values=call_func_outputs,
                                                           outputs=real_output,
                                                           inplace_process=inplace_process,
                                                           inplace_indices=inplace_indices_str,
                                                           cast_input_code=cast_input_code,
                                                           real_call_args_tensor=real_call_args_tensor,
                                                           class_name=op_proto.op_class.name,
                                                           op_name_str=op_proto.op_class.name)

            merge_op_header.append(self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE.replace(operator_name=op_proto.op_name,
                                                                                  device=self.device))
            op_register = self.PYBOOST_REG_OP_TEMPLATE.replace(op_name=op_proto.op_class.name,
                                                               device=self.device_reg_str,
                                                               register_custom_kernel="")
            merge_op_function.append(
                self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_proto.op_class.name,
                                                               call_args_with_type=call_args_with_type,
                                                               return_type=cpp_func_return,
                                                               call_impl=call_impl,
                                                               op_register=op_register,
                                                               device=self.device_reg_str))
            ascend_merge_op_inc.append(op_proto.op_class.name)

    @staticmethod
    def _generate_tensor_cpu_cast_input_code(op_parser: OpTemplateParser):
        """
        Generates the input casting code for CPU tensor operations.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            tuple: A tuple containing the casting code and the updated tensor call arguments.
        """
        _, _, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
        call_tensors = op_parser.get_call_args_tensor()
        cast_input = ""
        real_call_args_tensor = call_args_with_tensor.copy()
        for i, tensor in enumerate(call_args_with_tensor):
            is_tuple_tensor = real_call_args_tensor[i].endswith("_vector")
            is_tensor = real_call_args_tensor[i] in call_tensors
            if is_tensor:
                cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                              f'select_kernel.input_type()[{i}].dtype, device::DeviceType::kCPU);\n'
                real_call_args_tensor[i] = "real_" + real_call_args_tensor[i]
            if is_tuple_tensor:
                cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                              f'select_kernel.input_type()[{i}].dtype, device::DeviceType::kCPU);\n'
                real_call_args_tensor[i] = "PyBoostUtils::ConvertTensorVectorToTuple(real_" + real_call_args_tensor[
                    i] + ")"
        if cast_input != "":
            cast_input = "auto &select_kernel = kernel_attr_pair.second;\n" + cast_input
        return cast_input, real_call_args_tensor

    @staticmethod
    def _generate_create_input_address(op_parser: OpTemplateParser):
        need_malloc_tensors, _, _ = op_parser.parse_need_malloc_tensors()
        create_input_address = ''
        args_list = ', '.join(str(item) for item in need_malloc_tensors)
        if args_list:
            create_input_address = f'PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), {args_list});\n'
        return create_input_address

    @staticmethod
    def _generate_malloc_input(op_parser: OpTemplateParser):
        """
        Generates the code for creating input addresses for tensors that need to be allocated.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            str: The generated code for creating input addresses.
        """
        need_malloc_tensors, _, _ = op_parser.parse_need_malloc_tensors()
        malloc_inputs = ''
        args_list = ', '.join(str(item) for item in need_malloc_tensors)
        if args_list:
            malloc_inputs += f'PyBoostUtils::MallocOpInputs(device_context, {args_list});\n'
        return malloc_inputs

    @staticmethod
    def _generate_get_inputs_kernel_tensors(op_parser: OpTemplateParser):
        """
        Generates the code for retrieving input kernel tensors.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            str: The generated code for retrieving input kernel tensors.
        """
        _, _, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
        inputs_kernel_tensors = ''
        args_list = ', '.join(str(item) for item in call_args_with_tensor)
        if args_list:
            inputs_kernel_tensors += f'const auto &input_address_info = PyBoostUtils::GetAddressInfo(' \
                                     f'device_context, op->stream_id(), op->input_abs(), {args_list});\n'
        return inputs_kernel_tensors


class InternalOpCppCodeGenerator:
    """
    Generates C++ code files for internal operations in PyBoost.
    """

    def __init__(self, device):
        """
        Initializes the InternalOpCppCodeGenerator with the appropriate templates.
        """
        self.device = device
        self.internal_op_header_template = template.PYBOOST_INTERNAL_OP_HEADER_TEMPLATE
        self.internal_single_op_header_template = template.PYBOOST_INTERNAL_SINGLE_OP_HEADER_TEMPLATE
        self.internal_op_source_template = template.PYBOOST_INTERNAL_OP_SOURCE_TEMPLATE
        self.internal_single_op_source_template = template.PYBOOST_INTERNAL_SINGLE_OP_SOURCE_TEMPLATE
        self.internal_single_op_customize_source_template = template.PYBOOST_INTERNAL_SINGLE_OP_CUSTOMIZE_TEMPLATE
        self.customize_inc_template = Template(
            '#include "${ms_ops_kernel_path}/ascend/aclnn/pyboost_impl/internal/customize/${operator_name}.h"\n'
        )
        self.gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/internal/auto_generate/"

    def generate_internal_op_cpp_code(self, work_path, op_protos):
        """
        Generate internal op cpp code in pyboost.
        """
        merge_op_header = []
        merge_op_function = []
        ascend_merge_op_inc = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            internal_op_ascend = op_proto.op_dispatch.internal_op_ascend
            op_name = op_proto.op_class.name
            if internal_op_ascend == 'AutoGen':
                self.generate_default_call(work_path, op_proto, merge_op_header,
                                           merge_op_function, ascend_merge_op_inc)
            elif internal_op_ascend == 'Internal' + op_name + 'AscendCustomize':
                self.generate_customize_call(work_path, op_proto, merge_op_header,
                                             merge_op_function, ascend_merge_op_inc)

        if not ascend_merge_op_inc:
            return
        ops_inc_head_set = set()
        for op_name_inc in ascend_merge_op_inc:
            ops_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_name_inc[0].lower()))

        internal_op_source_str = self.internal_op_source_template.replace(ops_prim_inc=list(sorted(ops_inc_head_set)),
                                                                          merge_op_header=merge_op_header,
                                                                          merge_op_function=merge_op_function)
        save_path = os.path.join(work_path, self.gen_path)
        file_name = "pyboost_ascend_internal_ops.cc"
        save_file(save_path, file_name, internal_op_source_str)

    def generate_default_op_function(self, op_parser, op_proto):
        """
        Generate default op call function.
        """
        call_args_with_type = op_parser.parse_call_args_with_types()
        cpp_func_return = _generate_cpp_func_return(op_proto)
        _, call_func_outputs = op_parser.generate_pyboost_outputs()
        call_args = OpTemplateParser.parse_original_call_args(op_proto.op_args)
        call_args_after_convert, value_tuple_convert, const_number_convert = op_parser.op_args_converter()
        create_input_address, create_output_address = self._create_input_and_output_address(op_parser, op_proto)
        internal_op_source_str = self.internal_single_op_source_template.replace(
            op_name=op_proto.op_class.name,
            operator_name=op_proto.op_name,
            call_args_with_type=call_args_with_type,
            internal_call_args=call_args,
            internal_real_call_args=call_args_after_convert,
            create_input_address=create_input_address,
            create_output_address=create_output_address,
            value_tuple_convert=value_tuple_convert,
            const_number_convert=const_number_convert,
            return_type=cpp_func_return,
            return_values=call_func_outputs)
        return internal_op_source_str

    def generate_default_call(self, work_path, op_proto, merge_op_header,
                              merge_op_function, ascend_merge_op_inc):
        """
        Generate internal op default call function in pyboost.
        """
        op_parser = OpTemplateParser(op_proto)
        call_args_with_type = op_parser.parse_call_args_with_types()
        cpp_func_return = _generate_cpp_func_return(op_proto)

        # generate op header
        internal_op_header_str = self.internal_op_header_template.replace(
            operator_name=op_proto.op_name,
            op_name=op_proto.op_class.name,
            op_name_upper=op_proto.op_class.name.upper(),
            call_args_with_type=call_args_with_type,
            return_type=cpp_func_return)
        save_path = os.path.join(work_path, self.gen_path)
        save_file(save_path, f"{op_proto.op_name}.h", internal_op_header_str)
        merge_op_header.append(
            self.internal_single_op_header_template.replace(
                operator_name=op_proto.op_name,
                customize_inc=''))

        # generate op function
        internal_op_source_str = self.generate_default_op_function(op_parser, op_proto)
        merge_op_function.append(internal_op_source_str)

        ascend_merge_op_inc.append(op_proto.op_class.name)

    def generate_customize_call(self, work_path, op_proto, merge_op_header,
                                merge_op_function, ascend_merge_op_inc):
        """
        Generate internal op customize call function in pyboost.
        """
        op_parser = OpTemplateParser(op_proto)
        call_args_with_type = op_parser.parse_call_args_with_types()
        cpp_func_return = _generate_cpp_func_return(op_proto)

        # generate op header
        internal_op_header_str = self.internal_op_header_template.replace(
            operator_name=op_proto.op_name,
            op_name=op_proto.op_class.name,
            op_name_upper=op_proto.op_class.name.upper(),
            call_args_with_type=call_args_with_type,
            return_type=cpp_func_return)
        save_path = os.path.join(work_path, self.gen_path)
        save_file(save_path, f"{op_proto.op_name}.h", internal_op_header_str)
        customize_inc_str = self.customize_inc_template.replace(
            ms_ops_kernel_path=K.MS_OPS_KERNEL_PATH,
            operator_name=op_proto.op_name)
        merge_op_header.append(
            self.internal_single_op_header_template.replace(
                operator_name=op_proto.op_name,
                customize_inc=customize_inc_str))

        # generate op function
        _, call_func_outputs = op_parser.generate_pyboost_outputs()
        call_args = OpTemplateParser.parse_original_call_args(op_proto.op_args)
        internal_op_source_str = self.internal_single_op_customize_source_template.replace(
            op_name=op_proto.op_class.name,
            call_args=call_args,
            call_args_with_type=call_args_with_type,
            return_type=cpp_func_return,
            return_values=call_func_outputs)
        merge_op_function.append(internal_op_source_str)
        ascend_merge_op_inc.append(op_proto.op_class.name)

    @staticmethod
    def _create_input_and_output_address(op_parser: OpTemplateParser, op_proto):
        """
        Create input and output address.
        """
        need_malloc_tensors, _, _ = op_parser.parse_need_malloc_tensors()
        create_input_address = ''
        create_output_address = ''
        args_list = ''.join(f'{arg}, ' for arg in need_malloc_tensors)
        args_list = args_list[:-2]
        if args_list:
            create_input_address = f'PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), {args_list});\n'
        if op_proto.op_args_signature and op_proto.op_args_signature.rw_write:
            create_output_address = ''
        else:
            create_output_address = 'PyBoostUtils::PrepareOpOutputs(device_context_, op->stream_id(), outputs_);\n'

        return create_input_address, create_output_address


class PyboostOpFunctionGenerator(BaseGenerator):
    """
    Generates C++ source files for ACLNN operations in PyBoost.

    This class handles the generation of source files for operations that utilize the ACLNN framework,
    including customized calls and tensor management.

    Attributes:
        PYBOOST_CALL_TEMPLATE (Template): Template for generating ACLNN operation calls.
        PYBOOST_OP_SOURCE_TEMPLATE (Template): Template for generating operation source files.
        gen_path (str): Path for saving the generated C++ source files.
        device (str): The target device (ascend, cpu, or gpu).
    """

    def __init__(self):
        self.ascend_op_cpp_generator = PyboostOpCppGenerator('ascend')
        self.ascend_aclnn_cpp_generator = AclnnOpCppCodeGenerator('ascend')
        self.ascend_internal_op_cpp_generator = InternalOpCppCodeGenerator('ascend')

        self.cpu_op_cpp_generator = PyboostOpCppGenerator('cpu')
        self.cpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('cpu')

        self.gpu_op_cpp_generator = PyboostOpCppGenerator('gpu')
        self.gpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('gpu')

        self.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE
        self.PYBOOST_CPU_OP_SOURCE_TEMPLATE = template.PYBOOST_CPU_OP_SOURCE_TEMPLATE
        self.PYBOOST_GPU_OP_SOURCE_TEMPLATE = template.PYBOOST_GPU_OP_SOURCE_TEMPLATE
        self.ascend_gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
        self.cpu_gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
        self.gpu_gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
        self.hccl_gen_path = "mindspore/ops/kernel/ascend/hccl/pyboost/auto_generate/"

    def generate(self, work_path, op_protos):
        """
        Generate and save C++ source code for PyBoost operations across different devices.

        This method generates C++ source files for operations (`op_protos`) tailored to Ascend, CPU,
        and GPU devices. It combines headers and function implementations for each device, and then
        saves the final source files to the appropriate paths.

        Args:
            op_protos (list): A list of operation prototypes containing metadata such as
                              operation name, dispatch settings, arguments, and view attributes.
            work_path (str): The base working directory where the generated files will be saved.

        Generated Files:
            - Ascend: `pyboost_ascend_ops.cc`
            - CPU: `pyboost_cpu_ops.cc`
            - GPU: `pyboost_gpu_ops.cc`
        """
        self._generate_pyboost_ascend_ops(work_path, op_protos)
        self._generate_pyboost_cpu_ops(work_path, op_protos)
        self._generate_pyboost_gpu_ops(work_path, op_protos)

    def _generate_pyboost_ascend_ops(self, work_path, op_protos):
        """
        Generates Ascend PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        ascend_merge_op_header = []
        ascend_merge_op_function = []
        hccl_merge_op_header = []
        hccl_merge_op_function = []
        ascend_merge_op_inc = []
        ascend_merge_op_hccl_inc = []
        self.ascend_op_cpp_generator.generate_customize_op_cpp_code(op_protos, ascend_merge_op_header,
                                                                    ascend_merge_op_function, ascend_merge_op_inc,
                                                                    hccl_merge_op_header, hccl_merge_op_function,
                                                                    ascend_merge_op_hccl_inc)
        self.ascend_aclnn_cpp_generator.generate_aclnn_op_cpp_code(op_protos, ascend_merge_op_header,
                                                                   ascend_merge_op_function,
                                                                   ascend_merge_op_inc)
        self.ascend_internal_op_cpp_generator.generate_internal_op_cpp_code(work_path, op_protos)

        ascend_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(
            ascend_merge_op_header, chunk_size=120)
        ascend_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(
            ascend_merge_op_function, chunk_size=120)
        op_inc_list = chunk_list(ascend_merge_op_inc, n=120)

        new_gen_num = len(ascend_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(os.path.join(
            work_path, self.ascend_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(ascend_op_header_merge_by_chunk_size)),
                                             ascend_op_header_merge_by_chunk_size,
                                             ascend_op_function_merge_by_chunk_size):
            ops_inc_head_set = set()
            for op_name_inc in op_inc_list[i]:
                ops_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_name_inc[0].lower()))

            ascend_pyboost_op_source = self.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function, ops_inc=list(sorted(ops_inc_head_set)))
            save_file(os.path.join(work_path, self.ascend_gen_path), f"pyboost_ascend_ops_{i}.cc",
                      ascend_pyboost_op_source)

        ops_hccl_inc_head_set = set()
        for op_name_inc in ascend_merge_op_hccl_inc:
            ops_hccl_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_name_inc[0].lower()))
        hccl_pyboost_op_source = self.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE.replace(
            merge_op_header='\n'.join(hccl_merge_op_header), merge_op_function='\n'.join(hccl_merge_op_function),
            ops_inc=list(sorted(ops_hccl_inc_head_set)))
        save_file(os.path.join(work_path, self.hccl_gen_path), "pyboost_hccl_ops.cc", \
                  hccl_pyboost_op_source)

    def _generate_pyboost_cpu_ops(self, work_path, op_protos):
        """
        Generates CPU PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        cpu_merge_op_header = []
        cpu_merge_op_function = []
        cpu_merge_op_inc = []
        self.cpu_op_cpp_generator.generate_customize_op_cpp_code(
            op_protos, cpu_merge_op_header, cpu_merge_op_function, cpu_merge_op_inc)
        self.cpu_aclnn_cpp_generator.generate_aclnn_op_cpp_code(
            op_protos, cpu_merge_op_header, cpu_merge_op_function, cpu_merge_op_inc)
        cpu_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(
            cpu_merge_op_header, chunk_size=120)
        cpu_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(
            cpu_merge_op_function, chunk_size=120)
        op_inc_list = chunk_list(cpu_merge_op_inc, n=120)

        new_gen_num = len(cpu_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(
            os.path.join(work_path, self.cpu_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(cpu_op_header_merge_by_chunk_size)),
                                             cpu_op_header_merge_by_chunk_size,
                                             cpu_op_function_merge_by_chunk_size):
            ops_inc_head_set = set()
            for op_name_inc in op_inc_list[i]:
                ops_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_name_inc[0].lower()))
            op_header += '#include "kernel/cpu/pyboost/pyboost_op_plugin_utils.h"\n'
            cpu_pyboost_op_source = self.PYBOOST_CPU_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function, ops_inc=list(sorted(ops_inc_head_set)))
            save_file(os.path.join(work_path, self.cpu_gen_path), f"pyboost_cpu_ops_{i}.cc",
                      cpu_pyboost_op_source)

    def _generate_pyboost_gpu_ops(self, work_path, op_protos):
        """
        Generates GPU PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        gpu_merge_op_header = []
        gpu_merge_op_function = []
        gpu_merge_op_inc = []
        self.gpu_op_cpp_generator.generate_customize_op_cpp_code(
            op_protos, gpu_merge_op_header, gpu_merge_op_function, gpu_merge_op_inc)
        self.gpu_aclnn_cpp_generator.generate_aclnn_op_cpp_code(
            op_protos, gpu_merge_op_header, gpu_merge_op_function, gpu_merge_op_inc)
        gpu_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(
            gpu_merge_op_header, chunk_size=120)
        gpu_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(
            gpu_merge_op_function, chunk_size=120)
        op_inc_list = chunk_list(gpu_merge_op_inc, n=120)

        new_gen_num = len(gpu_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(
            os.path.join(work_path, self.gpu_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(gpu_op_header_merge_by_chunk_size)),
                                             gpu_op_header_merge_by_chunk_size,
                                             gpu_op_function_merge_by_chunk_size):
            ops_inc_head_set = set()
            for op_name_inc in op_inc_list[i]:
                ops_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_name_inc[0].lower()))
            gpu_pyboost_op_source = self.PYBOOST_GPU_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function, ops_inc=list(sorted(ops_inc_head_set)))
            save_file(os.path.join(work_path, self.gpu_gen_path), f"pyboost_gpu_ops_{i}.cc",
                      gpu_pyboost_op_source)

    def _delete_residual_merged_ops_files(self, files_path, new_gen_num):
        """
        Deletes residual merged operation files in the specified directory if the number of
        newly generated files does not match the number of existing ones.

        This method first lists all files in the specified directory, then filters out the files
        that match the pattern `pyboost_.*_ops_.*.cc` (i.e., files related to pyboost ops). It compares
        the number of such files (`old_files_num`) with the `new_gen_num` argument, which represents
        the expected number of new pyboost ops files. If the counts do not match, the method will
        delete all the existing pyboost ops files in the directory before any new ones can be generated.

        Args:
            files_path (str): The path to the directory containing the files to be checked and deleted.
            new_gen_num (int): The number of newly generated pyboost ops files expected to be in the directory.

        Returns:
            None
        """
        all_files = os.listdir(files_path)
        old_pyboost_ops_files = [file for file in all_files if re.match(r'pyboost_.*_ops_.*\.cc', file)]
        old_files_num = len(old_pyboost_ops_files)
        if new_gen_num != old_files_num:
            for file in old_pyboost_ops_files:
                os.remove(os.path.join(files_path, file))


def _generate_cpp_func_return(op_proto):
    """Generates the C++ return type for the given operator prototype.

    Args:
        op_proto (OpProto): The operator prototype containing return information.

    Returns:
        str: The C++ return type for the function based on the operator prototype.

    Raises:
        Exception: If no return type is found.
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
        cpp_func_return += ','.join(s for s in returns_type)
        cpp_func_return += ">"
    else:
        raise Exception("Not return found")
    return cpp_func_return


def _generate_inplace_process_cpp_code(op_proto):
    """Generates C++ code for updating outputs by input tensors for inplace processing.

    Args:
        op_proto (OpProto): The operator prototype containing return information.

    Returns:
        str: The C++ code for inplace processing, or an empty string if no inplace processing is needed.
    """
    inplace_process = '// RefOps update output by input tensor\n'
    has_ref = False
    for index, return_obj in enumerate(op_proto.op_returns):
        if return_obj.inplace != '':
            inplace_process += f'outputs_[{index}]->set_device_address(' \
                               f'{return_obj.inplace}_tensor->device_address()); '\
                               f'outputs_[{index}]->set_format(' \
                               f'{return_obj.inplace}_tensor->format()); '\
                               f'outputs_[{index}]->set_storage_info(' \
                               f'{return_obj.inplace}_tensor->storage_info()); '
            has_ref = True
            break
    if has_ref:
        return inplace_process
    return ''


def delete_residual_files(work_path, op_protos):
    """
    Deletes residual files generated for operator prototypes that are no longer needed.

    Args:
        work_path (str): The base directory path where generated files are located.
        op_protos (list): A list of operator prototypes that are currently valid.

    Returns:
        None
    """
    all_operator_name = []
    for op_proto in op_protos:
        all_operator_name.append(op_proto.op_name)
    devices = ["ascend", "gpu", "cpu"]
    code_generate_path_list = [f"{K.MS_OPS_KERNEL_PATH}/{device}/pyboost/auto_generate/"
                               if device != "ascend" else
                               f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/pyboost_impl/auto_generate/"
                               for device in devices]
    code_generate_path_list.append(
        f"{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/")
    for code_generate_path in code_generate_path_list:
        filter_files = []
        code_generate_path = os.path.join(work_path, code_generate_path)
        if os.path.exists(code_generate_path):
            all_files = os.listdir(code_generate_path)
            # No need to delete pyboost_.*_ops_.*.cc files and op_register.cc.
            # These residual files will be deleted before new files generate.
            filter_files = [file for file in all_files if
                            not re.match(r'pyboost_.*_ops_.*\.cc', file) and file != "op_register.cc"]
        registered_op_name = set(item.split(".")[0] for item in filter_files)
        need_clean_op = registered_op_name - set(all_operator_name)

        for file in filter_files:
            file_name = file.split(".")[0]
            if file_name in need_clean_op:
                file_path = os.path.join(code_generate_path, file)
                if os.path.exists(file_path):
                    os.remove(file_path)


class PyboostOpRegisterCppCodeGenerator:
    """
    Generates registration C++ code for PyBoost operations.

    This class is responsible for creating a registration source file that includes
    all the necessary headers and template instantiations for the registered operations.

    Attributes:
        PYBOOST_OP_REGISTER_TEMPLATE (Template): Template for generating the operation registration code.
    """

    def __init__(self):
        self.PYBOOST_OP_REGISTER_TEMPLATE = template.PYBOOST_OP_REGISTER_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates a C++ source file for registering all PyBoost operations.

        Args:
            work_path (str): The directory path where the registration file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operations.

        Returns:
            None
        """
        all_op_names = []
        internal_op_names = []
        all_functional_names = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if op_proto.op_view:
                continue
            op_name_str = op_proto.op_class.name
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') != 'None':
                internal_op_names.append(op_name_str)
            all_op_names.append(op_name_str)
            all_functional_names.append(op_proto.op_name)

        include_str = ''
        factory_str = ''
        for op_name in all_op_names:
            factory_str += "template class OpFactory<{0}>;\n".format(op_name)
        for op_name in internal_op_names:
            factory_str += "template class InternalOpFactory<{0}>;\n".format(op_name)
        for operator_name in all_functional_names:
            include_str += f'#include "{K.MS_PYBOOST_BASE_HEADER_PATH}/auto_generate/{operator_name}.h"\n'
        op_register_file_str = self.PYBOOST_OP_REGISTER_TEMPLATE.replace(op_includes=include_str,
                                                                         op_factory_templates=factory_str)
        save_path = os.path.join(work_path, f"{K.MS_PYBOOST_BASE_PATH}/auto_generate/")
        file_name = "op_register.cc"
        save_file(save_path, file_name, op_register_file_str)
