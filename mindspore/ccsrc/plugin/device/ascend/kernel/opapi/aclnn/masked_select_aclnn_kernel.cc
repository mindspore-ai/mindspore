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
#include "plugin/device/ascend/kernel/opapi/aclnn/masked_select_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {

void MaskedSelectAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto return_value = GEN_EXECUTOR(aclnnMaskedSelect, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]);
  UpdateWorkspace(std::get<0>(return_value));
}

bool MaskedSelectAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto res = GEN_EXECUTOR_CUST(aclnnMaskedSelect, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]);
  executor_ = std::get<1>(res);
  auto &all_tensor = std::get<2>(res);
  RunOpSync("aclnnMaskedSelect", stream_ptr, workspace);

  // Update output shape.
  outputs_shape_.resize(1);
  outputs_shape_[kIndex0] = transform::UpdateOutputShape(all_tensor.get<2>());
  return true;
}

void MaskedSelectAclnnKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &,
                                                          const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(outputs_shape_[kIndex0]);
}

MS_ACLLNN_KERNEL_FACTORY_REG(MaskedSelect, MaskedSelectAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
