# pylint: disable=broad-except
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
import logging
import gen_utils
from pyboost_utils import AclnnUtils


def gen_h(op_name, aclnn_name, op_yaml, kernelmod_h_path, need_update_shape):
    """generate h files"""
    kernelmod_name = op_yaml.get('dispatch').get("Ascend")
    h_head = f"""
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_{op_name.upper()}_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_{op_name.upper()}_ACLNN_KERNEL_MOD_H_
#include <vector>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"
"""
    update_shape = f"""
  bool IsNeedUpdateOutputShapeAndSize() override {{ return true; }}
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
"""
    if not need_update_shape:
        update_shape = ""
    h_body = f"""
namespace mindspore {{
namespace kernel {{

class {kernelmod_name} : public AclnnKernelMod {{
 public:
  explicit {kernelmod_name}() : AclnnKernelMod(std::move("{aclnn_name}")) {{}}
  ~{kernelmod_name}() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  {update_shape}
}};
}}  // namespace kernel
}}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_{op_name.upper()}_ACLNN_KERNEL_MOD_H_
"""
    fd = os.open(kernelmod_h_path, os.O_WRONLY | os.O_CREAT, 0o644)
    h_file = os.fdopen(fd, 'w')
    h_file.write(gen_utils.cc_license_str + h_head + h_body)
    h_file.close()

def gen_cc(op_name, class_name, op_yaml, kernelmod_cc_path, need_update_shape):
    """generate cc files"""
    kernelmod_name = op_yaml.get('dispatch').get("Ascend")
    cc_head = f"""
#include "plugin/device/ascend/kernel/opapi/aclnn/{op_name}_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {{
namespace kernel {{
"""
    inputs_num = len(op_yaml.get("args"))
    outputs_num = len(op_yaml.get("returns"))
    inputs = ""
    for i in range(inputs_num):
        inputs += "inputs[kIndex" + str(i) + "], "
    for i in range(outputs_num):
        inputs += "outputs[kIndex" + str(i) + "], "
    inputs = inputs[:-2]
    workspace_info = f"""
void {kernelmod_name}::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {{
  auto return_value = GEN_EXECUTOR(op_type_, {inputs});
  UpdateWorkspace(return_value);
}}
"""
    launch = f"""
bool {kernelmod_name}::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {{
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR(op_type_, {inputs}));
  RunOp(stream_ptr, workspace);
  return true;
}}
"""
    update_shape = f"""
void {kernelmod_name}::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &,
                                                const std::vector<KernelTensor *> &outputs) {{
  // Delete these comment and complete the function:
  // Using outputs[index_x]->SetShapeVector(update_shape) and outputs[index_x]->set_size(update_size)
}}
"""
    if not need_update_shape:
        update_shape = ""

    reg = f"""
MS_ACLLNN_KERNEL_FACTORY_REG({class_name}, {kernelmod_name});
}}  // namespace kernel
}}  // namespace mindspore

    """
    fd = os.open(kernelmod_cc_path, os.O_WRONLY | os.O_CREAT, 0o644)
    cc_file = os.fdopen(fd, 'w')
    cc_file.write(gen_utils.cc_license_str + cc_head + workspace_info + launch + update_shape + reg)
    cc_file.close()

def gen_nnacl_kernelmod(op_name, class_name, op_yaml, h_and_cc, need_update_shape):
    """generate cc and h files"""
    kernelmod_h_path = h_and_cc[0]
    kernelmod_cc_path = h_and_cc[1]
    aclnn_name = AclnnUtils.get_aclnn_interface(class_name)
    gen_h(op_name, aclnn_name, op_yaml, kernelmod_h_path, need_update_shape)
    gen_cc(op_name, class_name, op_yaml, kernelmod_cc_path, need_update_shape)

def main(op_name, need_update_shape):
    """main function"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../../')

    # get ops yaml
    ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops.yaml')
    aclnn_path = 'mindspore/ccsrc/plugin/device/ascend/kernel/opapi/aclnn/'
    kernelmod_cc_path = os.path.join(work_path, aclnn_path + '{}_aclnn_kernel.cc'.format(op_name))
    kernelmod_h_path = os.path.join(work_path, aclnn_path + '{}_aclnn_kernel.h'.format(op_name))
    yaml_str = gen_utils.safe_load_yaml(ops_yaml_path)
    op_yaml = yaml_str.get(op_name)
    if  op_yaml is None:
        raise ValueError("Input op {} is not find in ops.yaml.".format(op_name))
    dispatch = op_yaml.get("dispatch")
    if not dispatch or not dispatch.get("enable"):
        raise ValueError("Op {} is not enabled dispatch, please check.".format(op_name))
    if dispatch.get("Ascend") is None:
        raise ValueError("KernelMod {} is auto generated. If need achieve it, "
                         "please provide the KernelMod name in dispatch.".format(op_name))
    class_name = ''.join(word.capitalize() for word in op_name.split('_'))
    op_class = op_yaml.get("class")
    if op_class is not None and op_class.get("name") is not None:
        class_name = op_class.get("name")
    h_and_cc = [kernelmod_h_path, kernelmod_cc_path]
    gen_nnacl_kernelmod(op_name, class_name, op_yaml, h_and_cc, need_update_shape)


parser = argparse.ArgumentParser(description="Generate aclnn KernelMod.")
parser.add_argument('-n', '--name', type=str, default=None, help='Kernel name in yaml.')
parser.add_argument('-d', '--need_update_shape', type=bool, default=False,
                    help="Some kernel like:unique need update shape and size after launch. Default: False")
options, _ = parser.parse_known_args()

if __name__ == "__main__":
    try:
        name = options.name
        if name is None:
            raise ValueError("Please provide op name to generate aclnn kernelmod.")
        is_need_update_shape = options.need_update_shape
        main(name, is_need_update_shape)
    except Exception as e: # pylint: disable=W0703
        logging.exception("Generate aclnn kernelmod failed, err info: %s", e)
