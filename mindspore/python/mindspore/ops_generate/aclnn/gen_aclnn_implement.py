# Copyright 2023 Huawei Technologies Co., Ltd
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
Generate aclnn kernelmod or call func by input name in ops.yaml
"""
import argparse
import os
import re
import pathlib
import logging
import common.gen_utils as gen_utils
import common.template as template
import common.gen_constants as K
from common.op_proto import OpProto
from common.gen_constants import MS_OPS_KERNEL_PATH
from pyboost.pyboost_utils import AclnnUtils, get_dtypes

auto_gen = ''


def gen_h(kernelmod_name, aclnn_name, op_proto, kernelmod_h_path, need_update_shape):
    """generate h files"""
    op_name = op_proto.op_name
    update_shape = template.UPDATE_OUTPUT_SHAPE_AND_SIZE
    if not need_update_shape:
        update_shape = "\n  "

    file_path = kernelmod_h_path + ".h"
    aclnn_kernel_h_str = template.ACLNN_KERNEL_H_TEMPLATE.replace(aclnn_name=aclnn_name,
                                                                  op_name=op_name.upper(),
                                                                  auto_gen=auto_gen.upper(),
                                                                  kernelmod_name=kernelmod_name,
                                                                  update_shape=update_shape,
                                                                  ops_kernel_path=MS_OPS_KERNEL_PATH)
    gen_utils.save_file(os.path.dirname(file_path), os.path.basename(file_path), aclnn_kernel_h_str)


def gen_cc(kernelmod_name, aclnn_name, op_proto, kernelmod_cc_path, need_update_shape):
    """generate cc files"""
    op_name = op_proto.op_name
    tuple_tensor_not_supported = template.TUPLE_TENSOR_NOT_SUPPORTED.replace(
        op_name=op_name)
    input_templete = '\n  '
    inputs = ''
    input_dtypes, output_dtypes, _ = get_dtypes(op_proto)
    for idx, n in enumerate(input_dtypes):
        input_name = "inputs[kIndex" + str(idx) + "], "
        dtype = input_dtypes.get(n)
        if dtype != 'tensor':
            if dtype == 'int':
                dtype = 'int64_t'
            input_templete += "  auto {} = device::ascend::ConvertKernelTensor<{}>(inputs[kIndex{}]);\n".format(
                n.arg_name, dtype, idx)
            input_name = n.arg_name + ", "
        if dtype == 'tuple[tensor]' and auto_gen == "_auto_gen":
            raise NotImplementedError(tuple_tensor_not_supported)
        inputs += input_name
    input_templete = '' if input_templete == '\n  ' else input_templete
    for idx, n in enumerate(output_dtypes):
        output_name = "outputs[kIndex" + str(idx) + "], "
        dtype = output_dtypes.get(n)
        if dtype != 'tensor':
            if dtype == 'int':
                dtype = 'int64_t'
            input_templete += "  auto {} = device::ascend::ConvertKernelTensor<{}>(outputs[kIndex{}]);\n".format(
                n.arg_name, dtype, idx)
            output_name = n.arg_name + ", "
        if dtype == 'tuple[tensor]' and auto_gen == "_auto_gen":
            raise NotImplementedError(tuple_tensor_not_supported)
        inputs += output_name
    inputs = inputs[:-2]

    update_shape = template.update_output_shape_and_size_template.replace(
        kernelmod_name=kernelmod_name)
    if not need_update_shape:
        update_shape = ""

    file_path = kernelmod_cc_path + ".cc"
    aclnn_kernel_cc_str = template.ACLNN_KERNEL_CC_TEMPLATE.replace(kernelmod_name=kernelmod_name,
                                                                    input_templete=input_templete,
                                                                    inputs=inputs,
                                                                    update_shape=update_shape,
                                                                    class_name=aclnn_name,
                                                                    auto_gen_path=MS_OPS_KERNEL_PATH,
                                                                    op_name=op_name,
                                                                    auto_gen=auto_gen) + "    "
    gen_utils.save_file(os.path.dirname(file_path), os.path.basename(file_path), aclnn_kernel_cc_str)


def generate(kernelmod_name, class_name, op_proto, h_and_cc, need_update_shape):
    """generate cc and h files"""
    aclnn_name = AclnnUtils.get_aclnn_interface(class_name)
    gen_h(kernelmod_name, aclnn_name, op_proto, h_and_cc, need_update_shape)
    gen_cc(kernelmod_name, class_name, op_proto, h_and_cc, need_update_shape)


skip_aclnn_list = {"slice", "expand_dims", "squeeze", "split",
                   "generator", "view_as", "unstack_ext_view",
                   "reshape", "meshgrid", "flatten_ext"}


def gen_aclnn_kernel(op_proto: OpProto, need_update_shape=False, auto=False):
    """gen_aclnn_kernel function"""
    op_name = op_proto.op_name
    if op_name in skip_aclnn_list:
        logging.warning(
            "Operator {%s} has no aclnn interface, no aclnn kernel will be generated.", op_name)
        return
    if check_op_registed(op_proto.op_name) and not auto:
        logging.warning("Kernel {%s} is already registered.", op_name)
        return

    aclnn_path = f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/customize/'
    # merge inner ops
    dispatch = op_proto.op_dispatch
    aclnn_name = ''.join(word.capitalize() for word in op_name.split('_'))
    kernelmod_name = op_proto.op_dispatch.ascend
    if not dispatch or not op_proto.op_dispatch.enable:
        raise ValueError(
            "Op {} is not enabled dispatch, please check.".format(op_name))
    global auto_gen
    if auto:
        auto_gen = "_auto_gen"
        kernelmod_name = aclnn_name + "Ascend"
        aclnn_path = f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen/'
        pathlib.Path(os.path.join(K.WORK_DIR, aclnn_path)
                     ).mkdir(parents=True, exist_ok=True)
    if dispatch.ascend is None:
        raise ValueError("KernelMod {} is auto generated. If need achieve it, "
                         "please provide the KernelMod name in dispatch.".format(op_name))
    op_class = op_proto.op_class
    if op_class is not None and op_class.name is not None:
        aclnn_name = op_class.name
    kernelmod_h_and_cc_path = os.path.join(
        K.WORK_DIR, aclnn_path + '{}_aclnn_kernel'.format(op_name))
    generate(kernelmod_name, aclnn_name, op_proto,
             kernelmod_h_and_cc_path, need_update_shape)


def get_registed_ops(file_path=f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/'):
    '''get registered ops by search files'''
    # default search in 'ops/kernel/ascend/aclnn/kernel_mod_impl/'
    search_path = os.path.join(K.WORK_DIR, file_path)
    ret = []
    try:
        for root_path, _, files in os.walk(search_path):
            for file_name in files:
                with open(os.path.join(root_path, file_name), "r") as f:
                    file_context = f.read()
                    search_re = re.search(
                        r"(?<=KERNEL_FACTORY_REG\()\w+(?=,)", file_context)
                    if search_re:
                        ret.append(search_re.group())
    except OSError:
        logging.warning("Something wrong in check op registered.")
        return ret
    return ret


registed_ops = get_registed_ops()
manual_registed_ops = get_registed_ops(
    f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/customize/')


def check_op_registed(op_name, manual=False):
    '''if op already registered return true'''
    class_name = ''.join(word.capitalize() for word in op_name.split('_'))
    return (class_name in manual_registed_ops) if manual else (class_name in registed_ops)


def generate_aclnn_reg_code(yaml_data):
    """generate aclnn register code"""
    ops_yaml_path = os.path.join(K.WORK_DIR, K.PY_OPS_GEN_PATH, "ops.yaml")
    yaml_str = gen_utils.safe_load_yaml(ops_yaml_path)

    reg_code = f"""
#include "{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {{
namespace kernel {{
"""
    for operator_name, operator_data in yaml_data.items():
        dispatch = operator_data.get("dispatch")
        if not dispatch or not dispatch.get("enable"):
            continue
        if operator_name in skip_aclnn_list:
            logging.warning(
                "Operator {%s} has no aclnn interface, no aclnn kernel will be generated.", operator_name)
            continue
        Ascend = dispatch.get("Ascend")
        if Ascend is not None:  # KernelMod is provided by yaml, don't auto generate it.
            continue
        if check_op_registed(operator_name):
            logging.warning(
                "Kernel {%s} is already registered.", operator_name)
            continue
        _, _, none_tensor_exist = get_dtypes(operator_data)
        if none_tensor_exist:
            gen_aclnn_kernel(operator_name, yaml_str, auto=True)
            continue
        class_name = ''.join(word.capitalize()
                             for word in operator_name.split('_'))
        op_class = operator_data.get("class")
        if op_class and op_class.get("name") is not None:
            class_name = op_class.get("name")
        inputs_outputs_num = len(operator_data.get(
            "args")) + len(operator_data.get("returns"))
        aclnn_name = AclnnUtils.get_aclnn_interface(class_name)
        reg_code += f"""
MS_ACLNN_COMMON_KERNEL_FACTORY_REG({class_name}, {aclnn_name}, {inputs_outputs_num});"""
    reg_code += f"""
}}  // namespace kernel
}}  // namespace mindspore
"""
    return reg_code


def generate_aclnn_reg_file(work_path, yaml_str):
    """
    Generate nnacl kernelmod register
    """
    tmp_register_file = work_path + \
        f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/tmp_aclnn_kernel_register.cc'
    register_file = work_path + \
        f'{MS_OPS_KERNEL_PATH}/ascend/aclnn/kernel_mod_impl/aclnn_kernel_register_auto.cc'
    reg_code = generate_aclnn_reg_code(yaml_str)
    gen_utils.save_file(
        os.path.dirname(tmp_register_file), os.path.basename(tmp_register_file), gen_utils.cc_license_str + reg_code)
    gen_utils.check_change_and_replace_file(register_file, tmp_register_file)


def main(op_name, need_update_shape):
    '''main func'''
    gen_aclnn_kernel(op_name, need_update_shape)


parser = argparse.ArgumentParser(description="Generate aclnn KernelMod.")
parser.add_argument('-n', '--name', type=str, default=None,
                    help='Kernel name in yaml.')
parser.add_argument('-d', '--need_update_shape', type=bool, default=False,
                    help="Some kernel like:unique need update shape and size after launch. Default: False")
options, _ = parser.parse_known_args()

if __name__ == "__main__":
    try:
        name = options.name
        if name is None:
            raise ValueError(
                "Please provide op name to generate aclnn kernelmod.")
        is_need_update_shape = options.need_update_shape
        main(name, is_need_update_shape)
    except Exception as e:  # pylint: disable=W0703
        logging.exception("Generate aclnn kernelmod failed, err info: %s", e)
