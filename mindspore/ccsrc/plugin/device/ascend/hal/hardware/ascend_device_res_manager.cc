/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "runtime/rt.h"

namespace mindspore {
namespace device {
namespace ascend {
void AscendDeviceResManager::Initialize() {
  MS_LOG(INFO) << "Device resource manager Initialize start...";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = dynamic_cast<AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id));
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  if (!runtime_instance_->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  mem_manager_ = runtime_instance_->GetMemoryManager();
  MS_EXCEPTION_IF_NULL(mem_manager_);

  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    rank_id_ = GetRankId();
  }

  compute_stream_ = runtime_instance_->compute_stream();
  MS_EXCEPTION_IF_NULL(compute_stream_);
  communication_stream_ = runtime_instance_->communication_stream();
  MS_EXCEPTION_IF_NULL(communication_stream_);

  MS_LOG(INFO) << "Device resource manager Initialize success.";
}

void AscendDeviceResManager::Destroy() {
  MS_LOG(INFO) << "Device resource manager Destroy start...";
  if (DataQueueMgr::GetInstance().IsInit()) {
    MS_EXCEPTION_IF_CHECK_FAIL(DataQueueMgr::GetInstance().Destroy(), "Could not destroy ascend data queue.");
  }

  rank_id_ = 0;
  if (runtime_instance_) {
    runtime_instance_ = nullptr;
  }
  MS_LOG(INFO) << "Device resource manager Destroy success.";
}

bool AscendDeviceResManager::BindDeviceToCurrentThread() const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  return true;
}

void *AscendDeviceResManager::AllocateMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  runtime_instance_->SetContext();
  return mem_manager_->MallocMemFromMemPool(size, false);
}

void AscendDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

bool AscendDeviceResManager::AllocateMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (device_name_in_address != device_context_->device_context_key().device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_->device_context_key().device_name_;
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  runtime_instance_->SetContext();
  auto device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem());
  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  return true;
}

std::vector<void *> AscendDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  std::vector<size_t> align_size_list;
  for (size_t i = 0; i < size_list.size(); i++) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size_list[i]);
    align_size_list.emplace_back(align_size);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(align_size_list);
}

DeviceAddressPtr AscendDeviceResManager::CreateDeviceAddress(void *const device_ptr, size_t device_size,
                                                             const string &format, TypeId type_id,
                                                             const ShapeVector &shape) const {
  auto device_address = std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id,
                                                              device_context_->device_context_key().device_name_,
                                                              device_context_->device_context_key().device_id_);
  if (shape.empty()) {
    MS_LOG(DEBUG) << "shape size is empty.";
  }
  device_address->set_host_shape(shape);
  return device_address;
}

bool AscendDeviceResManager::SyncStream(size_t stream_id) const {
  auto iter = stream_ids_.find(stream_id);
  if (iter != stream_ids_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    auto ret = rtStreamSynchronize(iter->second);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to synchronize ascend stream, ret[" << ret << "]";
      return false;
    }
    return true;
  }

  if (runtime_instance_ != nullptr) {
    return runtime_instance_->SyncStream();
  }

  return true;
}

bool AscendDeviceResManager::CreateStream(void **stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = rtStreamCreate(stream, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to create ascend stream, ret[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceResManager::DestroyStream(void *stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = rtStreamDestroy(stream);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to destroy ascend stream, ret[" << ret << "]";
    return false;
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
