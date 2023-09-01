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

#include "extendrt/kernel/ascend_native/ascend_native_copy_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ops/copy.h"

namespace mindspore::kernel {
int AscendNativeCopyKernel::InferShape() {
  if (out_tensors_[0]->shape().size() == 0) {
    if (in_tensors_[0] != nullptr) {
      std::vector<int> shape;
      for (size_t j = 0; j < in_tensors_[0]->shape().size(); j++) {
        shape.push_back(in_tensors_[0]->shape()[j]);
      }
      out_tensors_[0]->set_shape(shape);
    }
  }
  return kSuccess;
}

int AscendNativeCopyKernel::Prepare() {
  auto prim = GetValueNode<PrimitivePtr>(primitive_.cnode->input(0));
  copy_type_ = static_cast<ops::Copy::CopyFormatType>(GetValue<int64_t>(prim->GetAttr(mindspore::ops::kCopyFormat)));
  auto ret = InferShape();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ascend native copy kernel inferShape failed.";
    return kLiteError;
  }
  return kSuccess;
}

int AscendNativeCopyKernel::PreProcess() {
  switch (copy_type_) {
    case ops::Copy::CopyFormatType::HOST_DEVICE: {
      if (out_tensors_[0]->device_data() == nullptr) {
        auto device_data = ascend_native::MallocDevice(out_tensors_[0]->Size(), const_cast<void *>(stream_));
        if (device_data == nullptr) {
          MS_LOG(ERROR) << "fail to allocate " << out_tensors_[0]->Size() << "Bytes for device";
          return kLiteError;
        }
        out_tensors_[0]->set_device_data(device_data);
      }
      break;
    }
    case ops::Copy::CopyFormatType::DEVICE_HOST: {
      if (out_tensors_[0]->data() == nullptr) {
        out_tensors_[0]->MallocData();
        if (out_tensors_[0]->data() == nullptr) {
          MS_LOG(ERROR) << "fail to allocate " << out_tensors_[0]->Size() << "Bytes for host";
          return kLiteError;
        }
      }
      break;
    }
    case ops::Copy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return kLiteError;
    }
      out_tensors_[0]->ResetRefCount();
  }
  return kSuccess;
}

int AscendNativeCopyKernel::Run() {
  MS_LOG(INFO) << "AscendNativeCopyKernel::Execute";
  auto elem = out_tensors_[0]->ElementsNum();
  // Execute copy
  switch (copy_type_) {
    case ops::Copy::CopyFormatType::HOST_DEVICE: {
      if (in_tensors_[0]->data() == nullptr) {
        MS_LOG(ERROR) << "no host data to tensor " << in_tensors_[0]->tensor_name();
        return kLiteError;
      }
      void *dst = out_tensors_[0]->device_data();
      if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
        ascend_native::CopyHostFp16ToDeviceFp16(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(stream_));
      } else {
        ascend_native::CopyHostFp32ToDeviceFp16(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(stream_));
      }
      break;
    }
    case ops::Copy::CopyFormatType::DEVICE_HOST: {
      if (in_tensors_[0]->device_data() == nullptr) {
        MS_LOG(ERROR) << "no device data to tensor " << in_tensors_[0]->tensor_name();
        return kLiteError;
      }
      out_tensors_[0]->set_data_type(kNumberTypeFloat32);
      ascend_native::CopyDeviceFp16ToHostFp32(in_tensors_[0]->device_data(), out_tensors_[0]->data(), elem,
                                              const_cast<void *>(stream_));
      break;
    }
    case ops::Copy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported. " << copy_type_;
      return kLiteError;
    }
  }
  return kSuccess;
}

int AscendNativeCopyKernel::PostProcess() {
  switch (copy_type_) {
    case ops::Copy::CopyFormatType::HOST_DEVICE: {
      in_tensors_[0]->DecRefCount();
      break;
    }
    case ops::Copy::CopyFormatType::DEVICE_HOST: {
      auto ref = in_tensors_[0]->ref_count();
      in_tensors_[0]->set_ref_count(--ref);
      if (ref <= 0) {
        ascend_native::FreeDevice(in_tensors_[0]->device_data(), const_cast<void *>(stream_));
      }
      break;
    }
    case ops::Copy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return kLiteError;
    }
  }
  return kSuccess;
}

REGISTER_ASCEND_NATIVE_CREATOR(ops::kNameCopy, AscendNativeCopyKernel)
}  // namespace mindspore::kernel
