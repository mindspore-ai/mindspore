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
Generate pyboost base call function
"""

import os
import sys

LINCENSE_CODE = "/**\n\
 * Copyright 2023 Huawei Technologies Co., Ltd\n\
 *\n\
 * Licensed under the Apache License, Version 2.0 (the \"License\");\n\
 * you may not use this file except in compliance with the License.\n\
 * You may obtain a copy of the License at\n\
 *\n\
 * http://www.apache.org/licenses/LICENSE-2.0\n\
 *\n\
 * Unless required by applicable law or agreed to in writing, software\n\
 * distributed under the License is distributed on an \"AS IS\" BASIS,\n\
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
 * See the License for the specific language governing permissions and\n\
 * limitations under the License.\n\
 */\n"

CPU = "CPU"
GPU = "GPU"
ASCEND = "Ascend"

def create_yaml(op_name, base_path):
    file_path = os.path.join(base_path, "mindspore/core/ops/ops_def/")
    if not os.path.exists(file_path):
        assert "yaml path is not exist: " + file_path
    yaml_name = op_name.lower() + "_op.yaml"
    yaml_path = os.path.join(file_path, yaml_name)
    yaml_template = "#operator {op_name_lower}\n\
{op_name}:\n\
  args:\n\
    // TODO: add input args, like: \n\
    input:\n\
      dtype: tensor\n\
  returns:\n\
    // TODO: add oputput, like: \n\
    output:\n\
      dtype: tensor\n\
  class:\n\
    name: {op_name}\n\
  pyboost: True\n".format(op_name_lower=op_name.lower(),  op_name=op_name)

    with open(yaml_path, "w") as file:
        file.write(yaml_template)

    yaml_doc = op_name.lower() + "_doc.yaml"
    yaml_doc_path = os.path.join(file_path, yaml_doc)
    yaml_doc_template = "{op_name}:\n\
    description: |\n\
        //TODO: Add the op description\n".format(op_name=op_name.lower())
    with open(yaml_doc_path, "w") as file:
        file.write(yaml_doc_template)

def create_base(op_name, base_path):
    file_path = os.path.join(base_path, "mindspore/ccsrc/kernel/pyboost/op/")
    if not os.path.exists(file_path):
        assert "base op path is not exist: " + file_path
    base_h = op_name.lower() + ".h"
    base_h_path = os.path.join(file_path, base_h)
    base_h_template = "#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_{op_name_upper}_H_\n\
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_{op_name_upper}_H_\n\
\n\
#include \"kernel/pyboost/op_register.h\"\n\
\n\
namespace mindspore {{\n\
namespace kernel {{\n\
namespace pyboost {{\n\
class BACKEND_EXPORT {op_name} : public pyboost::Op {{\n\
 public:\n\
  {op_name}() = default;\n\
  ~{op_name}() override = default;\n\
\n\
  virtual tensor::TensorPtr Call(const tensor::TensorPtr &x) = 0;\n\
}};\n\
}}  // namespace pyboost\n\
}}  // namespace kernel\n\
}}  // namespace mindspore\n\
#endif // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_{op_name_upper}_H_\n".format(op_name_upper=op_name.upper(), op_name=op_name)
    
    with open(base_h_path, "w") as file:
        file.write(LINCENSE_CODE)
        file.write("\n")
        file.write(base_h_template)

def create_kernel_file(op_name, base_path, type):
    file_path = os.path.join(base_path, "mindspore/ccsrc/plugin/device/{type}/kernel/pyboost/".format(type=type.lower()))
    if not os.path.exists(file_path):
        assert "file path is not exist: " + file_path
    file_h = op_name.lower() + "_" + type.lower() + ".h"
    file_h_path = os.path.join(file_path, file_h)

    file_cc = op_name.lower() + "_" + type.lower() + ".cc"
    file_cc_path = os.path.join(file_path, file_cc)
    return file_h_path, file_cc_path

def create_code(op_name, base_path, type):
    file_h, file_cc = create_kernel_file(op_name, base_path, type)
    type_upper = type.upper()
    op_name_upper = op_name.upper()
    op_name_lower = op_name.lower()
    op_type = op_name + type
    op_type_lower = op_name_lower + "_" + type.lower()

    file_h_template = "#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_{type_upper}_KERNEL_PYBOOST_{op_name_upper}_{type_upper}_H_\n\
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_{type_upper}_KERNEL_PYBOOST_{op_name_upper}_{type_upper}_H_\n\
\n\
#include \"kernel/pyboost/op/{op_name_lower}.h\"\n\
\n\
namespace mindspore {{\n\
namespace kernel {{\n\
namespace pyboost {{\n\
class {op_type} : public pyboost::{op_name} {{\n\
 public:\n\
  {op_type}() = default;\n\
  ~{op_type}() = default;\n\
\n\
  tensor::TensorPtr Call(/*TODO: add input paramters*/) override;\n\
}};\n\
MS_REG_PYBOOST_OP({type}, {op_name});\n\
}}  // namespace pyboost\n\
}}  // namespace kernel\n\
}}  // namespace mindspore\n\
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_{type_upper}_KERNEL_PYBOOST_{op_name_upper}_{type_upper}_H_\n".format(type_upper=type_upper,op_name_upper=op_name_upper,op_name_lower=op_name_lower, op_type=op_type, op_name=op_name, type=type)


    file_cc_template = "#include \"plugin/device/{type_lower}/kernel/pyboost/{op_type_lower}.h\"\n\
\n\
namespace mindspore {{\n\
namespace kernel {{\n\
namespace pyboost {{\n\
tensor::TensorPtr {op_type}::Call(/*TODO: add input paramters*/) {{\n\
  // TODO: add op\n\
  MS_LOG(DEBUG) << \"Call start\";\n\
  // Step1: add infer func: CPU/GPU/Ascend are same\n\
  // InferOutput(input, batch1, batch2, beta, alpha);\n\
  // Step2: add malloc func: CPU/GPU/Ascend are same\n\
  // Don't need to allocate memory for Scalar.\n\
  // DeviceMalloc(input, batch1, batch2);\n\
  // Step3: add stream func: CPU/GPU/Ascend are same\n\
  // auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);\n\
  // Step4: add launc func: CPU/GPU/Ascend are different\n\
  // Ascend: need to check cube_math_type, please refer to op document of CANN\n\
  // Ascend: LAUNCH_ACLNN_CUBE(aclnnBaddbmm, stream_ptr, input, batch1, batch2, beta, alpha, output(0));\n\
  // Ascend: LAUNCH_ACLNN(aclnnBaddbmm, stream_ptr, input, batch1, batch2, beta, alpha, output(0));\n\
  // CPU: framework will provide the func\n\
  // GPU: framework will provide the func\n\
  MS_LOG(DEBUG) << \"Launch end\";\n\
  return outputs_[0];\n\
}}\n\
}}  // namespace pyboost\n\
}}  // namespace kernel\n\
}}  // namespace mindspore\n".format(op_type_lower=op_type_lower, op_type=op_type, type_lower=type.lower())

    with open(file_h, "w") as file:
        file.write(LINCENSE_CODE)
        file.write("\n")
        file.write(file_h_template)

    with open(file_cc, "w") as file:
        file.write(LINCENSE_CODE)
        file.write("\n")
        file.write(file_cc_template)


def gen_code(op_name, base_path):
    create_yaml(op_name, base_path)
    create_base(op_name, base_path)
    create_code(op_name, base_path, CPU)
    create_code(op_name, base_path, GPU)
    create_code(op_name, base_path, ASCEND)
    
def check_valid(base_path):
    if not os.path.exists(base_path):
        assert "root path is not exist: " + base_path

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        assert "Please check input args: 1. op_name; 2. mindspore code root path! e.g. python gen_code.py SiLU /d/mindspore"
    op_name = sys.argv[1]
    base_path = sys.argv[2]
    check_valid(base_path)
    gen_code(op_name, base_path)
