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
#include <vector>
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_add_ext_common.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace inplace_add_ext {
void InplaceAddsExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  GetScalarFromInput(inputs[kIndex1], &this->other_scalar_);
  GetScalarFromInput(inputs[kIndex2], &this->alpha_scalar_);
  GetWorkspaceForResize(inputs[kIndex0], this->other_scalar_, this->alpha_scalar_);
}

bool InplaceAddsExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], this->other_scalar_, this->alpha_scalar_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceAddsExt, InplaceAddsExtAclnnKernelMod);
}  // namespace inplace_add_ext
}  // namespace kernel
}  // namespace mindspore
