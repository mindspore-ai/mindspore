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
#include "plugin/device/ascend/kernel/opapi/aclnn/normal_float_float_aclnn_kernel.h"

namespace mindspore {
namespace kernel {
void NormalFloatFloatAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  mean_ = transform::ConvertKernelTensor<float>(inputs[kIndex0]);
  std_ = transform::ConvertKernelTensor<float>(inputs[kIndex1]);
  seed_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  offset_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  GetWorkspaceForResize(mean_, std_, seed_, offset_, outputs[0]);
}

bool NormalFloatFloatAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, mean_, std_, seed_, offset_, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NormalFloatFloat, NormalFloatFloatAscend);
}  // namespace kernel
}  // namespace mindspore
