/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/gpu/gpu_device_address.h"
#include <vector>
#include <memory>
#include "runtime/device/gpu/gpu_device_manager.h"
#include "utils/log_adapter.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "ir/tensor.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#include "debug/tensor_load.h"
#include "debug/debugger/debugger.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
bool GPUDeviceAddress::SyncDeviceToHost(const std::vector<int> &, size_t size, TypeId, void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = GPUDeviceManager::GetInstance().SyncStream(stream);
  if (!ret) {
    MS_LOG(ERROR) << "SyncStream failed";
    return ret;
  }
  if (size != size_) {
    MS_LOG(WARNING) << "SyncDeviceToHost ignored, host size: " << size << ", device size " << size_;
    return true;
  }
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(host_ptr, ptr_, size_);
}

bool GPUDeviceAddress::SyncHostToDevice(const std::vector<int> &, size_t, TypeId, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(ptr_, host_ptr, size_, stream)) {
    MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncStream(stream);
}

GPUDeviceAddress::~GPUDeviceAddress() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}
#ifdef ENABLE_DEBUGGER
bool GPUDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                                     const std::vector<int> &host_shape, TypeId host_type, size_t slot,
                                     Debugger *debugger, bool keep_prev) const {
  bool ret = false;
  if (size_ == 0) {
    return true;
  }
  DebugServices *debug_services = debugger->debug_services();
  TensorLoader *tensor_loader = debug_services->tensor_loader();

  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(type_id_, host_shape);
  size_t host_size = out_tensor->data().nbytes();
  auto ret_rt_memcpy = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  if (!ret_rt_memcpy) {
    MS_LOG(ERROR) << "Copy device mem to host failed";
    return ret;
  }
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetSlot(slot);
  ret = tensor_loader->LoadNewTensor(tensor_data, keep_prev);
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  return ret;
}
#endif
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
