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

#include "src/extendrt/kernel/default/kernel_mod_kernel.h"
#include "src/extendrt/utils/tensor_utils.h"
#include "src/extendrt/kernel/default/cnode_infer_manager.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int KernelModKernel::Prepare() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  auto inputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(out_tensors_);

  bool ret = kernel_mod_->Init_(this->base_operator_, inputs, outputs);
  return ret ? ReSize() : RET_ERROR;
}

int KernelModKernel::ReSize() {
  auto inputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(out_tensors_);
  return kernel_mod_->Resize(inputs, outputs);
}

int KernelModKernel::Run() {
  auto inputs = CloudTensorUtils::LiteTensorToAddressPtrVec(in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToAddressPtrVec(out_tensors_);

  AddressPtrList workspace;
  auto workspace_size = kernel_mod_->GetWorkspaceSizeList();
  for (size_t &i : workspace_size) {
    auto buffer = context_->allocator->Malloc(i);
    std::shared_ptr<Address> address = std::make_shared<Address>(buffer, i);
    workspace.push_back(address);
  }

  auto ret = kernel_mod_->Launch(inputs, workspace, outputs, nullptr);

  for (const auto &address : workspace) {
    context_->allocator->Free(address->addr);
  }
  return ret ? RET_OK : RET_ERROR;
}

int KernelModKernel::InferShape() { return CNodeInferShape(cnode_, this->out_tensors_); }
}  // namespace mindspore::kernel
