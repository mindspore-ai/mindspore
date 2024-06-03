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
#include "plugin/device/ascend/kernel/opapi/aclnn/fill_scalar_aclnn_kernel.h"

namespace mindspore {
namespace kernel {

void FillScalarAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  value_ = transform::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  GetWorkspaceForResize(outputs[kIndex0], value_);
}

bool FillScalarAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, outputs[kIndex0], value_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FillScalar, FillScalarAscend);
}  // namespace kernel
}  // namespace mindspore
