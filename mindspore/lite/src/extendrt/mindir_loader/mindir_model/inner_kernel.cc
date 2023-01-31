/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <memory>

#include "extendrt/mindir_loader/mindir_model/inner_kernel.h"
#include "abstract/abstract_value.h"

namespace mindspore::kernel {
int InnerKernel::Prepare() {
  auto inputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(this->in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(this->out_tensors_);

  return this->kernel_mod_->Init(this->base_operator_, inputs, outputs) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}

int InnerKernel::Execute() {
  auto inputs = CloudTensorUtils::LiteTensorToAddressPtrVec(this->in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToAddressPtrVec(this->out_tensors_);

  std::vector<AddressPtr> workspace;

  return this->kernel_mod_->Launch(inputs, workspace, outputs, nullptr) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}

int InnerKernel::ReSize() {
  // use InitOp instead
  auto inputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(this->in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(this->out_tensors_);

  return this->kernel_mod_->Init(this->base_operator_, inputs, outputs) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}
}  // namespace mindspore::kernel
