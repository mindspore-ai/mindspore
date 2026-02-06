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
#include "include/backend/ms_backend/runtime/graph_scheduler/base/move_to.h"
#include <string>
#include <memory>
#include <algorithm>
#include "utils/stream_guard.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "device_address/device_type.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "device_address/device_address.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace device {
namespace {
bool MoveToD2H(const tensor::TensorPtr &src_tensor, const DeviceAddressPtr &src_device_ptr,
               const tensor::TensorPtr &dst_tensor, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  if (src_device_ptr == nullptr) {
    MS_LOG(DEBUG) << "Origin tensor has no device address, just copy host value";
    size_t size = dst_tensor->Size();
    auto ret = memcpy_s(dst_tensor->data_c(), size, src_tensor->data_c(), size);
    return ret == EOK;
  }
  auto ret = true;
  std::string status;
  if (blocking) {
    status = "SyncDeviceToHost";
    device::DeviceContextKey host_key = {src_device_ptr->GetDeviceType(), src_device_ptr->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    (void)host_context->device_res_manager_->SyncAllStreams();
    MS_EXCEPTION_IF_NULL(dst_tensor->device_address());
    MS_EXCEPTION_IF_NULL(src_tensor->device_address());
    ret = SyncCopy(dst_tensor, src_tensor, CurrentStream::id());
  } else {
    status = "AsyncDeviceToHost";
    MS_EXCEPTION_IF_NULL(dst_tensor->device_address());
    ret = AsyncCopy(dst_tensor, src_tensor, CurrentStream::id());
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << status << " failed.";
  }
  return true;
}

void MoveToH2D(const tensor::TensorPtr &src_tensor, const DeviceAddressPtr &src_device_ptr,
               const tensor::TensorPtr &dst_tensor, const DeviceAddressPtr &dst_device_ptr, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_device_ptr);
  auto ret = true;
  std::string status;
  if (blocking) {
    DeviceContextKey host_key = {dst_device_ptr->GetDeviceType(), dst_device_ptr->device_id()};
    DeviceContext *host_context = DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    (void)host_context->device_res_manager_->SyncAllStreams();
    ret = AsyncCopy(dst_tensor, src_tensor, CurrentStream::id());
    status = "SyncHostToDevice";
  } else {
    ret = AsyncCopy(dst_tensor, src_tensor, CurrentStream::id());
    status = "AsyncHostToDevice";
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << status << " failed.";
  }
}
}  // namespace

void MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, const std::string &to,
            bool blocking, bool *return_self) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(return_self);

  auto src_addr = src_tensor->device_address();
  device::DeviceAddressPtr src_device_ptr = nullptr;
  if (src_addr != nullptr) {
    src_device_ptr = src_addr;
    MS_EXCEPTION_IF_NULL(src_device_ptr);
    auto src_type = GetDeviceNameByType(src_device_ptr->GetDeviceType());
    if (to == src_type) {
      MS_LOG(DEBUG) << "The tensor is already on: " << to << ", no need move again";
      *return_self = true;
      return;
    }
  }

  // Need to create cpu device address even if the tensor is on CPU.
  // H2D src_device_ptr: CPU; dst_device_ptr: GPU/ASCEND.
  auto dst_addr = dst_tensor->device_address();
  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto target_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({GetDeviceTypeByName(to), device_id});
  MS_EXCEPTION_IF_NULL(target_context);
  target_context->Initialize();
  auto stream_id = CurrentStream::id();
  if (target_context->device_res_manager_->GetStream(stream_id) == nullptr) {
    stream_id = kDefaultStreamIndex;
  }

  if (dst_addr == nullptr) {
    auto size = src_device_ptr != nullptr ? src_device_ptr->GetSize() : src_tensor->Size();
    dst_addr = std::make_shared<device::DeviceAddress>(nullptr, size, GetDeviceTypeByName(to), 0);
    MS_EXCEPTION_IF_NULL(dst_addr);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   dst_addr->GetSize(), dst_addr.get());
    if (!target_context->device_res_manager_->AllocateMemory(dst_addr.get(), stream_id)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, maybe device memory(device id:" << device_id
                        << ") isn't enough. Allocate size: " << size;
    }
    dst_tensor->set_device_address(dst_addr);
    dst_tensor->set_format(Format::DEFAULT_FORMAT);
    dst_tensor->set_storage_info(nullptr);
  } else if (dst_addr->GetMutablePtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   dst_addr->GetSize(), dst_addr.get());
    if (!target_context->device_res_manager_->AllocateMemory(dst_addr.get(), stream_id)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, maybe device memory(device id:" << device_id
                        << ") isn't enough. Allocate size: " << dst_addr->GetSize();
    }
  } else {
    MS_LOG(DEBUG) << "Dst Address already have allocated memory, " << dst_addr->ToString();
  }

  // D2H copy, src_device_ptr: GPU/ASCEND; dst_device_ptr: CPU.
  if (to == "CPU") {
    if (src_device_ptr == nullptr) {
      MS_LOG(INFO) << "Src tensor device ptr is null, means tensor on: " << to << ", no need move again!";
      *return_self = true;
      return;
    }
    if (!MoveToD2H(src_tensor, src_device_ptr, dst_tensor, blocking)) {
      MS_LOG(EXCEPTION) << "Move tensor to " << to << "failed.";
    }
    dst_tensor->set_sync_status(kNeedSyncHostToDevice);
    return;
  }

  MoveToH2D(src_tensor, src_device_ptr, dst_tensor, dst_addr, blocking);
  dst_tensor->set_sync_status(kNeedSyncDeviceToHost);
}
}  // namespace device
}  // namespace mindspore
