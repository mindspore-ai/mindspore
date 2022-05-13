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
  auto inputs = LiteTensorToKernelTensorPtrVec(this->in_tensors_);
  auto outputs = LiteTensorToKernelTensorPtrVec(this->out_tensors_);

  return this->kernel_mod_->Init(this->base_operator_, inputs, outputs) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}

int InnerKernel::Execute() {
  auto inputs = LiteTensorToAddressPtrVec(this->in_tensors_);
  auto outputs = LiteTensorToAddressPtrVec(this->out_tensors_);

  std::vector<AddressPtr> workspace;

  return this->kernel_mod_->Launch(inputs, workspace, outputs, nullptr) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}

int InnerKernel::ReSize() {
  // use InitOp instead
  auto inputs = LiteTensorToKernelTensorPtrVec(this->in_tensors_);
  auto outputs = LiteTensorToKernelTensorPtrVec(this->out_tensors_);

  return this->kernel_mod_->Init(this->base_operator_, inputs, outputs) ? mindspore::lite::RET_OK
                                                                        : mindspore::lite::RET_ERROR;
}

std::vector<KernelTensorPtr> InnerKernel::LiteTensorToKernelTensorPtrVec(
  const std::vector<lite::Tensor *> &lite_tensors) {
  std::vector<KernelTensorPtr> ret_vec;

  for (auto lite_tensor : lite_tensors) {
    auto kernel_tensor_ptr = LiteTensorToKernelTensorPtr(lite_tensor);
    ret_vec.push_back(kernel_tensor_ptr);
  }

  return ret_vec;
}

KernelTensorPtr InnerKernel::LiteTensorToKernelTensorPtr(lite::Tensor *lite_tensor) {
  KernelTensorPtr kernel_tensor_ptr = std::make_shared<mindspore::kernel::KernelTensor>();
  auto address_ptr = LiteTensorToAddressPtr(lite_tensor);
  kernel_tensor_ptr->SetData(address_ptr);
  kernel_tensor_ptr->SetFormat(lite_tensor->format());

  auto kernel_tensor_abstract_ptr = std::make_shared<mindspore::abstract::AbstractScalar>();

  auto type_ptr = mindspore::TypeIdToType(lite_tensor->data_type());
  kernel_tensor_abstract_ptr->set_type(type_ptr);

  auto lite_shape = lite_tensor->shape();
  std::vector<int64_t> shape;
  for (size_t i = 0; i < lite_shape.size(); i++) {
    shape.push_back(lite_shape[i]);
  }
  kernel_tensor_abstract_ptr->set_shape(std::make_shared<abstract::Shape>(shape));
  kernel_tensor_ptr->SetAbstract(kernel_tensor_abstract_ptr);
  return kernel_tensor_ptr;
}

std::vector<AddressPtr> InnerKernel::LiteTensorToAddressPtrVec(const std::vector<lite::Tensor *> &lite_tensors) {
  std::vector<AddressPtr> ret_vec;

  for (auto lite_tensor : lite_tensors) {
    auto address_ptr = LiteTensorToAddressPtr(lite_tensor);
    ret_vec.push_back(address_ptr);
  }

  return ret_vec;
}

AddressPtr InnerKernel::LiteTensorToAddressPtr(lite::Tensor *lite_tensor) {
  AddressPtr address_ptr = std::make_shared<mindspore::kernel::Address>();
  address_ptr->addr = lite_tensor->data();
  address_ptr->size = lite_tensor->Size();
  return address_ptr;
}
}  // namespace mindspore::kernel
