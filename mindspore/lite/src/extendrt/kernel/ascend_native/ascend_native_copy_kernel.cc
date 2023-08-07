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
#include "extendrt/kernel/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ops/copy.h"

namespace mindspore::kernel {
int AscendNativeCopyKernel::Prepare() {
  auto prim = GetValueNode<PrimitivePtr>(primitive_.cnode->input(0));
  copy_type_ = static_cast<ops::Copy::CopyFormatType>(GetValue<int64_t>(prim->GetAttr(mindspore::ops::kCopyFormat)));
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

int AscendNativeCopyKernel::Execute() {
  MS_LOG(INFO) << "AscendNativeCopyKernel::Execute";
  void *dst = out_tensors_[0]->device_data();
  if (copy_type_ == ops::Copy::CopyFormatType::DEVICE_HOST) {
    dst = out_tensors_[0]->data();
  }
  auto elem = out_tensors_[0]->ElementsNum();
  // Check if output should be allocated
  if (dst == nullptr) {
    if (copy_type_ == ops::Copy::CopyFormatType::HOST_DEVICE) {
      // Allocate device memory
      auto size = out_tensors_[0]->Size();
      out_tensors_[0]->set_device_data(ascend_native::MallocDevice(size, const_cast<void *>(stream_)));
    } else if (copy_type_ == ops::Copy::CopyFormatType::DEVICE_HOST) {
      // Allocate host memory
      out_tensors_[0]->MallocData();
    } else {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return kLiteError;
    }
  }

  // Execute copy
  switch (copy_type_) {
    case ops::Copy::CopyFormatType::HOST_DEVICE: {
      if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
        ascend_native::CopyHostFp16ToDeviceFp16(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(get_stream()));
      } else {
        ascend_native::CopyHostFp32ToDeviceFp32(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(get_stream()));
      }
      break;
    }
    case ops::Copy::CopyFormatType::DEVICE_HOST: {
      if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
        ascend_native::CopyDeviceFp16ToHostFp16(in_tensors_[0]->device_data(), out_tensors_[0]->data(), elem,
                                                const_cast<void *>(get_stream()));
      } else {
        ascend_native::CopyDeviceFp32ToHostFp32(in_tensors_[0]->device_data(), out_tensors_[0]->data(), elem,
                                                const_cast<void *>(get_stream()));
      }
      break;
    }
    case ops::Copy::CopyFormatType::NONE:
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    default:
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return kLiteError;
  }
  return kSuccess;
}

REGISTER_ASCEND_NATIVE_CREATOR(ops::kNameCopy, AscendNativeCopyKernel)
}  // namespace mindspore::kernel
