/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include <algorithm>
#include "runtime/device/move_to.h"
#include "include/backend/device_type.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace {
void MoveToD2H(const DeviceAddressPtr &src_device_ptr, const tensor::TensorPtr &dst_tensor, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_device_ptr);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  if (blocking && !src_device_ptr->SyncDeviceToHost(dst_tensor->Size(), dst_tensor->data_c())) {
    MS_LOG(EXCEPTION) << "SyncDeviceToHost failed.";
  } else if (!src_device_ptr->AsyncDeviceToHost(dst_tensor->Size(), dst_tensor->data_c())) {
    MS_LOG(EXCEPTION) << "AsyncDeviceToHost failed.";
  }
}

void MoveToH2D(const DeviceAddressPtr &src_device_ptr, const DeviceAddressPtr &dst_device_ptr, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_device_ptr);
  MS_EXCEPTION_IF_NULL(dst_device_ptr);
  size_t size = std::min(src_device_ptr->GetSize(), dst_device_ptr->GetSize());
  if (blocking && !dst_device_ptr->SyncHostToDevice(size, src_device_ptr->GetPtr())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
  } else if (!dst_device_ptr->AsyncHostToDevice(size, src_device_ptr->GetPtr())) {
    MS_LOG(EXCEPTION) << "AsyncHostToDevice failed.";
  }
}

}  // namespace

void MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, const std::string &to,
            bool blocking, bool *return_self) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(return_self);

  const auto &device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (to != "CPU" && to != device) {
    MS_LOG(EXCEPTION) << "The value of arg 'to' of method 'move_to' should be same with device target, bug got to:"
                      << to << ", device target: " << device;
  }

  auto src_addr = src_tensor->device_address();
  MS_EXCEPTION_IF_NULL(src_addr);
  auto src_device_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(src_addr);
  MS_EXCEPTION_IF_NULL(src_device_ptr);
  auto src_type = GetDeviceNameByType(src_device_ptr->GetDeviceType());
  if (to == src_type) {
    MS_LOG(WARNING) << "The value of src tensor type:[" << src_type << "] equal to 'move_to' target type:[" << to
                    << ", no need to move tensor.";
    *return_self = true;
    return;
  }
  // D2H copy, src_device_ptr: GPU/ASCEND; dst_device_ptr: CPU.
  if (to == "CPU") {
    MoveToD2H(src_device_ptr, dst_tensor, blocking);
    return;
  }
  // H2D src_device_ptr: CPU; dst_device_ptr: GPU/ASCEND.
  auto dst_addr = dst_tensor->device_address();
  if (dst_addr == nullptr) {
    auto size = src_device_ptr->GetSize();
    auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto target_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({to, device_id});
    MS_EXCEPTION_IF_NULL(target_context);
    target_context->Initialize();
    auto stream_id = target_context->device_res_manager_->GetCurrentStreamId();
    if (target_context->device_res_manager_->GetStream(stream_id) == nullptr) {
      stream_id = kDefaultStreamIndex;
    }
    auto new_ptr = target_context->device_res_manager_->AllocateMemory(size, stream_id);
    MS_EXCEPTION_IF_NULL(new_ptr);
    auto kernel_tensor =
      std::make_shared<kernel::KernelTensor>(new_ptr, size, kernel::GetFormatFromStrToEnum(kOpFormat_DEFAULT),
                                             src_device_ptr->type_id(), src_device_ptr->host_shape(), to, device_id);
    dst_addr = target_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    dst_tensor->set_device_address(dst_addr);
  }
  auto dst_device_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(dst_addr);
  MS_EXCEPTION_IF_NULL(dst_device_ptr);
  MoveToH2D(src_device_ptr, dst_device_ptr, blocking);
}
}  // namespace device
}  // namespace mindspore
