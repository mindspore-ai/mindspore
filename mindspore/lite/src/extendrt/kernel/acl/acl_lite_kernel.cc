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

#include <string>
#include <algorithm>

#include "src/extendrt/kernel/acl/acl_lite_kernel.h"
#include "src/extendrt/utils/tensor_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
AclLiteKernel::AclLiteKernel(std::shared_ptr<mindspore::kernel::KernelMod> kernel_mod, BaseOperatorPtr base_operator,
                             std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
                             const lite::InnerContext *ctx)
    : BaseKernel({base_operator, nullptr}, std::move(in_tensors), std::move(out_tensors), ctx),
      kernel_mod_(std::move(kernel_mod)),
      base_operator_(std::move(base_operator)) {
  inputs_ = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(in_tensors_);
  outputs_ = CloudTensorUtils::LiteTensorToKernelTensorPtrVec(out_tensors_);
}

int AclLiteKernel::Prepare() {
  bool ret = kernel_mod_->Init_(this->base_operator_, inputs_, outputs_);
  return ret ? ReSize() : RET_ERROR;
}

int AclLiteKernel::ReSize() {
  // acl custom kernel last input is om data, do not pass to resize
  std::vector<KernelTensorPtr> kernel_inputs;
  kernel_inputs.assign(inputs_.begin(), inputs_.end() - 1);

  return kernel_mod_->Resize(kernel_inputs, outputs_);
}

int AclLiteKernel::InferShape() {
  // new shape is already updated in in_tensors_, infer shape base on in_tensors_

  // current acl do not support change static shape

  // if infer of in_tensors_ is not changed, do nothing
  bool shape_changed = false;
  if (inputs_.size() != in_tensors_.size()) {
    MS_LOG(ERROR) << "New shape size " << in_tensors_.size() << " is not the same with old shape size "
                  << inputs_.size();
    return lite::RET_ERROR;
  }
  // in_tensors_ last is om data, delete it
  for (size_t i = 0; i < inputs_.size() - 1; i++) {
    auto new_input = in_tensors_.at(i);
    auto old_input = inputs_.at(i);

    auto new_shape = new_input->shape();
    auto is_dynamic = std::any_of(new_shape.begin(), new_shape.end(), [i](auto dim) { return dim < 0; });
    if (is_dynamic) {
      MS_LOG(ERROR) << "New shape of input " << i << " cannot be dynamic, new shape: " << new_shape;
      return lite::RET_NOT_SUPPORT;
    }
    if (old_input->GetShapeVector() != new_input->shape64()) {
      shape_changed = true;
    }
  }
  if (!shape_changed) {
    for (size_t i = 0; i < outputs_.size(); i++) {
      auto new_output = out_tensors_.at(i);
      auto old_output = outputs_.at(i);
      new_output->set_shape64(old_output->GetShapeVector());
      new_output->set_data_type(old_output->GetDtype());
    }
    return lite::RET_OK;
  }

  return lite::RET_NOT_SUPPORT;
}

int AclLiteKernel::Run() {
  auto inputs = CloudTensorUtils::LiteTensorToAddressPtrVec(in_tensors_);
  auto outputs = CloudTensorUtils::LiteTensorToAddressPtrVec(out_tensors_);

  // acl custom kernel last input is om data, do not pass to run
  std::vector<AddressPtr> kernel_inputs;
  kernel_inputs.assign(inputs.begin(), inputs.end() - 1);

  AddressPtrList workspace;
  auto workspace_size = kernel_mod_->GetWorkspaceSizeList();
  for (size_t i = 0; i < workspace_size.size(); i++) {
    auto buffer = context_->allocator->Malloc(workspace_size.at(i));
    std::shared_ptr<Address> address = std::make_shared<Address>(buffer, workspace_size.at(i));
    workspace.push_back(address);
  }

  if (in_tensors_.size() != inputs_.size()) {
    MS_LOG(ERROR) << "Given inputs size " << in_tensors_.size() << " != graph inputs size " << inputs_.size();
    return kLiteError;
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto &input = in_tensors_[i];
    auto &kernel_input = inputs_[i];
    if (input->Size() != kernel_input->GetSizeInBytes()) {
      MS_LOG(ERROR) << "Byte size of input " << i << " != the size expected, given size " << input->Size()
                    << ", expected size " << kernel_input->GetSizeInBytes()
                    << ", input shape: " << kernel_input->GetShapeVector();
      return kLiteError;
    }
    auto input_device_address = input->device_data();
    if (input_device_address != nullptr) {
      auto device_ptr = input_device_address;
      kernel_input->SetData(std::make_shared<kernel::Address>(device_ptr, input->Size()));
      kernel_input->SetHostData(nullptr);
    } else {
      kernel_input->SetHostData(std::make_shared<kernel::Address>(input->data(), input->Size()));
      kernel_input->SetData(nullptr);
    }
  }
  // solve out_tensors empty case
  if (out_tensors_.empty()) {
    std::transform(outputs_.begin(), outputs_.end(), std::back_inserter(out_tensors_), [](auto &item) {
      auto shape64 = item->GetShapeVector();
      std::vector<int> shape;
      std::transform(shape64.begin(), shape64.end(), std::back_inserter(shape),
                     [](auto &value) { return static_cast<int>(value); });
      return new lite::Tensor(item->GetDtype(), shape);
    });
  }
  if (out_tensors_.size() != outputs_.size()) {
    MS_LOG(ERROR) << "Given outputs size " << outputs.size() << " != graph inputs size " << outputs_.size();
    return kLiteError;
  }
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    auto output = out_tensors_[i];
    auto kernel_output = outputs_[i];
    if (output->Size() != kernel_output->GetSizeInBytes()) {
      MS_LOG(ERROR) << "Byte size of output " << i << " != the size expected, given size " << output->Size()
                    << ", expected size " << kernel_output->GetSizeInBytes()
                    << ", output shape: " << kernel_output->GetShapeVector();
      return kLiteError;
    }
    auto output_device_address = output->device_data();
    if (output_device_address != nullptr) {
      auto device_ptr = output_device_address;
      kernel_output->SetData(std::make_shared<kernel::Address>(device_ptr, output->Size()));
      kernel_output->SetHostData(nullptr);
    } else {
      kernel_output->SetHostData(std::make_shared<kernel::Address>(output->data(), output->Size()));
      kernel_output->SetData(nullptr);
    }
  }

  auto ret = kernel_mod_->Launch(kernel_inputs, workspace, outputs, nullptr);

  for (auto address : workspace) {
    context_->allocator->Free(address->addr);
  }
  return ret ? RET_OK : RET_ERROR;
}
}  // namespace mindspore::kernel
