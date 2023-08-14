/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/registry/kernel_interface_registry.h"
#include "src/common/log_adapter.h"

namespace mindspore::kernel {
void Kernel::Initialize() {
  if (primitive_ == nullptr) {
    return;
  }
  type_ = primitive_->value_type();
  if (type_ == schema::PrimitiveType_Custom) {
    auto param = primitive_->value_as_Custom();
    if (param != nullptr && param->type() != nullptr) {
      SetAttr("type", param->type()->str());
    }
  }
}

int Kernel::InferShape() {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  std::shared_ptr<KernelInterface> kernel_interface = nullptr;
  if (type() == schema::PrimitiveType_Custom) {
    kernel_interface = registry::KernelInterfaceRegistry::Instance()->GetKernelInterface("", nullptr, this);
  } else {
    auto device_list = const_cast<mindspore::Context *>(context_)->MutableDeviceInfo();
    for (auto &device : device_list) {
      MS_CHECK_TRUE_RET(device != nullptr, lite::RET_NULL_PTR);
      kernel_interface =
        registry::KernelInterfaceRegistry::Instance()->GetKernelInterface(device->GetProvider(), nullptr, this);
      if (kernel_interface != nullptr) {
        break;
      }
    }
  }

  if (kernel_interface == nullptr) {
    MS_LOG(ERROR) << "op_type: " << schema::EnumNamePrimitiveType(type_) << " can not find infer interface.";
    return lite::RET_NOT_SUPPORT;
  }
  auto ret = kernel_interface->Infer(&inputs_, &outputs_, static_cast<const schema::Primitive *>(primitive_), this);
  if (ret == kLiteInferInvalid) {
    for (auto output : outputs_) {
      output.SetShape({-1});
    }
    return lite::RET_INFER_INVALID;
  }
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "op_type: " << schema::EnumNamePrimitiveType(type_) << " infer fail!ret: " << ret;
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
#endif
  return lite::RET_NOT_SUPPORT;
}
}  // namespace mindspore::kernel
