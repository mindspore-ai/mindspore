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
  ParseGenExecutor(GEN_EXECUTOR(aclnnMaskedSelect, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]));
}

bool MaskedSelectAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto [workspace_size, executor_, tensor_param] =
    GEN_EXECUTOR_CUSTOM(aclnnMaskedSelect, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]);
  if (workspace_size != 0) {
    std::vector<size_t> workspace_size_list = {workspace_size};
    SetWorkspaceSizeList(workspace_size_list);
  }
  if (workspace_size_list_.empty()) {
    RUN_OP_API_SYNC(aclnnMaskedSelect, stream_ptr, nullptr, 0, executor_);
  } else {
    RUN_OP_API_SYNC(aclnnMaskedSelect, stream_ptr, workspace[0]->device_ptr(), workspace_size_list_[0], executor_);
  }

  // Update output shape.
  outputs_[0]->SetShapeVector(transform::UpdateOutputShape(tensor_param.get<2>()));
  return true;
}

MS_ACLLNN_KERNEL_FACTORY_REG(MaskedSelect, MaskedSelectAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
