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
#include "plugin/device/ascend/kernel/opapi/aclnn/baddbmm_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
bool BaddbmmAclnnFunctionalKernelMod::Call(const tensor::TensorPtr &self, const tensor::TensorPtr &batch1,
                                           const tensor::TensorPtr &batch2, const ScalarPtr &beta,
                                           const ScalarPtr &alpha, const tensor::TensorPtr &out) {
  MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);

  CreateTensorAddress(self, "self");
  CreateTensorAddress(batch1, "batch1");
  CreateTensorAddress(batch2, "batch2");
  CreateTensorAddress(out, "out");

  size_t workspace_size = 0;
  // 910A not support 0
  int8_t cube_math_type = 0;
  tie(workspace_size, executor_, std::ignore) =
    GEN_EXECUTOR_CUSTOM(aclnnBaddbmm, self, batch1, batch2, beta, alpha, out, cube_math_type);

  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  if (workspace_size == 0) {
    RUN_OP_API_SYNC(aclnnBaddbmm, stream_ptr, nullptr, 0, executor_);
  } else {
    auto workspace_device_address = CreateWorkspaceAddress(workspace_size);
    RUN_OP_API_SYNC(aclnnBaddbmm, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor_);
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
