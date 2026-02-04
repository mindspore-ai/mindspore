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
Generates C++ registration code for ACL NN kernels based on operator prototypes.
"""

import os
import logging
import re

import common.gen_constants as K
from common import gen_utils
from common import template
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils
from aclnn.gen_aclnn_implement import gen_aclnn_kernel, skip_aclnn_list
from resources.resource_list import ResourceType


ACLNN_REG_CODE = """
#include "$ops_gen_kernel_path/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {

$aclnn_reg_code
}  // namespace kernel
}  // namespace mindspore
"""


class AclnnKernelRegisterAutoCcGenerator(BaseGenerator):
    """Generates ACL NN kernel registration code for Ascend devices."""

    def __init__(self):
        self.aclnn_reg_code_template = template.Template(ACLNN_REG_CODE)

    def generate(self, work_path, op_protos):
        """
        Generates registration code for ACL NN kernels and saves it to a file.

        Args:
            work_path (str): The directory to save the generated file.
            op_protos (list): List of operator prototypes.

        Returns:
            None
        """
        aclnn_reg_code = []
        for op_proto in op_protos:
            if not op_proto.op_dispatch or not op_proto.op_dispatch.enable:
                continue
            if op_proto.op_name in skip_aclnn_list:
                logging.warning(
                    "Operator {%s} has no aclnn interface, no aclnn kernel will be generated.", op_proto.op_name)
                continue
            # KernelMod is provided by yaml, don't auto generate it.
            if op_proto.op_dispatch.ascend != 'default':
                continue
            if check_op_registed(op_proto.op_name, True):
                logging.warning(
                    "Kernel {%s} is already registered.", op_proto.op_name)
                continue
            _, _, none_tensor_exist = pyboost_utils.get_dtypes(op_proto)
            if none_tensor_exist:
                # gen operator aclnn kernel c++ files
                gen_aclnn_kernel(op_proto, auto=True)
                continue

            class_name = op_proto.op_class.name
            inputs_outputs_num = len(op_proto.op_args) + \
                len(op_proto.op_returns)
            aclnn_name = pyboost_utils.AclnnUtils.get_aclnn_interface(
                class_name)
            aclnn_reg_code.append(
                f"MS_ACLNN_COMMON_KERNEL_FACTORY_REG({class_name}, {aclnn_name}, {inputs_outputs_num});\n")

        reg_code = self.aclnn_reg_code_template.replace(ops_gen_kernel_path=K.MS_OPS_KERNEL_PATH,
                                                        aclnn_reg_code=aclnn_reg_code)
        res_str = template.CC_LICENSE_STR + reg_code

        save_path = os.path.join(
            work_path, f"{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/auto_generate")
        file_name = "aclnn_kernel_register_auto.cc"
        gen_utils.save_file(save_path, file_name, res_str)


def get_registed_ops(file_path=f'{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/',
                     pattern=r"(?<=KERNEL_FACTORY_REG\()\w+(?=,)"):
    '''get registered ops by search files'''
    # default search in 'ops/kernel/ascend/aclnn/kernel_mod_impl/'
    search_path = os.path.join(K.WORK_DIR, file_path)
    ret = []
    try:
        for root_path, _, files in os.walk(search_path):
            for file_name in files:
                if file_name == 'aclnn_kernel_register_auto.cc':
                    continue

                with open(os.path.join(root_path, file_name), "r", encoding='utf-8') as f:
                    file_context = f.read()
                    search_re = re.search(pattern, file_context)
                    if search_re:
                        ret.append(search_re.group())
    except OSError:
        logging.warning("Something wrong in check op registered.")
        return ret
    return ret


registed_ops = get_registed_ops()
manual_registed_ops = get_registed_ops(f'{K.MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/customize/')
host_registed_ops = get_registed_ops(f'{K.MS_OPS_KERNEL_PATH}/host/view/',
                                     r"(?<=MS_HOST_REG_KERNEL\()\w+(?=,)")


def check_op_registed(op_name, manual=False):
    '''if op already registered return true'''
    class_name = ''.join(word.capitalize() for word in op_name.split('_'))
    if manual:
        return class_name in manual_registed_ops or class_name in host_registed_ops
    return class_name in registed_ops


def generate_aclnn_reg_file(resource_mgr):
    """
    Generate nnacl kernelmod register
    """
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    generator = AclnnKernelRegisterAutoCcGenerator()
    generator.generate(K.WORK_DIR, op_protos)
