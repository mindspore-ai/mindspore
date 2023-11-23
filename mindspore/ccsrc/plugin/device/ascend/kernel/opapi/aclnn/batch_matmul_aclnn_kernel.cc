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
#include "plugin/device/ascend/kernel/opapi/aclnn/batch_matmul_aclnn_kernel.h"
#include <vector>
#include <algorithm>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
void BMMAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  const auto &attr_list = primitive()->attrs();
  if (attr_list.at("transpose_a") || attr_list.at("transpose_b")) {
    MS_LOG(EXCEPTION) << "Please check attr transpose_a or transpose_b in BatchMatMul";
  }
  auto trans_a = GetValue<bool>(attr_list.at("transpose_a"));
  auto trans_b = GetValue<bool>(attr_list.at("transpose_b"));
  input_a_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_a);
  input_b_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_b);

  auto return_value = GEN_EXECUTOR(op_type_, input_a_, input_b_, outputs[kIndex0], OpApiUtil::GetCubeMathType());
  UpdateWorkspace(return_value);
}

bool BMMAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR(op_type_, input_a_, input_b_, outputs[kIndex0], OpApiUtil::GetCubeMathType()));
  RunOp(stream_ptr, workspace);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
