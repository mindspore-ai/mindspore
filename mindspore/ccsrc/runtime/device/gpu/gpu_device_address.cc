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
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
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
bool GPUDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  bool need_sync = (size != 0) && (size_ != 0) && (size <= size_);
  if (!need_sync) {
    return true;
  }
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = GPUDeviceManager::GetInstance().SyncStream(stream);
  if (!ret) {
    MS_LOG(ERROR) << "SyncStream failed";
    return ret;
  }
  if (size != size_) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << size_;
  }
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(host_ptr, ptr_, size);
}

bool GPUDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  bool need_sync = (size != 0) && (size_ != 0) && (size <= size_);
  if (!need_sync) {
    return true;
  }
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  if (size != size_) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << size_;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (execution_mode != kPynativeMode) {
    if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(ptr_, host_ptr, size, stream)) {
      MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
      return false;
    }
    return GPUDeviceManager::GetInstance().SyncStream(stream);
  } else {
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kGPUDevice, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    return runtime_instance->MemcpyAsync(ptr_, host_ptr, size, 0);
  }
}

void GPUDeviceAddress::ClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}

GPUDeviceAddress::~GPUDeviceAddress() { ClearDeviceMemory(); }
#ifdef ENABLE_DEBUGGER
bool GPUDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                                     const ShapeVector &host_shape, TypeId host_type, size_t slot,
                                     bool keep_prev) const {
  bool ret = false;
  if (size_ == 0) {
    return true;
  }

  if (Debugger::GetInstance()->TensorExistsInCurrent(tensor_name)) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }

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
  ret = Debugger::GetInstance()->LoadNewTensor(tensor_data, keep_prev);
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  return ret;
}
#endif
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
