/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/repeat_interleave_grad_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void RepeatInterleaveGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto dout_shape = inputs[kIndex0]->GetShapeVector();
  dim_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  auto rank = SizeToLong(dout_shape.size());
  dim_ = (dim_ < 0) ? (dim_ + rank) : dim_;
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], dim_, outputs[kIndex0]);
}

bool RepeatInterleaveGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], dim_, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RepeatInterleaveGrad, RepeatInterleaveGradAscend);
}  // namespace kernel
}  // namespace mindspore
