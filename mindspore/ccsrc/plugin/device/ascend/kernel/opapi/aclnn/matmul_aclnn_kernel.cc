/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/ascend/kernel/opapi/aclnn/matmul_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void MMAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  const auto &attr_list = primitive()->attrs();
  bool trans_a = false;
  bool trans_b = false;
  if (attr_list.at("transpose_a") && attr_list.at("transpose_b")) {
    trans_a = GetValue<bool>(attr_list.at("transpose_a"));
    trans_b = GetValue<bool>(attr_list.at("transpose_b"));
  }
  auto shape_a = inputs[0]->GetShapeVector();
  auto shape_b = inputs[1]->GetShapeVector();
  if ((shape_a.size() == shape_b.size()) && (shape_a.size() == kIndex2)) {
    op_type_ = "aclnnMm";
  }
  input_a_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_a);
  input_b_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], trans_b);
  auto return_value = GEN_EXECUTOR(op_type_, input_a_, input_b_, outputs[kIndex0], OpApiUtil::GetCubeMathType());
  UpdateWorkspace(return_value);
}

bool MMAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR(op_type_, input_a_, input_b_, outputs[kIndex0], OpApiUtil::GetCubeMathType()));
  RunOp(stream_ptr, workspace);
  return true;
}
// MS_ACLLNN_KERNEL_FACTORY_REG(MatMul, MMAclnnKernelMod);
// MS_ACLLNN_KERNEL_FACTORY_REG(BatchMatMul, MMAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
