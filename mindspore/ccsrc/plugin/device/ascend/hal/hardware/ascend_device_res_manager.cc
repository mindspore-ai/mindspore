/**
 *
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
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

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

  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    // get actual rank id if it's distribution training case.
    rank_id_ = GetRankId();
  }
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    auto_mem_offload_ =
      std::make_shared<MindRTAutoOffloadAdapter>(&AscendMemoryPool::GetInstance(), kDefaultStreamIndex);
  }
  MS_LOG(INFO) << "Device resource manager Initialize success.";
}

void AscendDeviceResManager::Destroy() {
  MS_LOG(INFO) << "Device resource manager Destroy start...";
  if (DataQueueMgr::GetInstance().IsInit()) {
    DataQueueMgr::GetInstance().Release();
  }

  rank_id_ = 0;
  if (runtime_instance_) {
    runtime_instance_ = nullptr;
  }
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    MS_LOG(EXCEPTION) << "Fail to destroy all streams when destroy DeviceResManager.";
  }
  MS_LOG(INFO) << "Device resource manager Destroy success.";
}

bool AscendDeviceResManager::BindDeviceToCurrentThread(bool /* force_bind */) const {
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
  MS_EXCEPTION_IF_NULL(mem_manager_);
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
  if (auto_mem_offload_ != nullptr) {
    return auto_mem_offload_->Malloc(address);
  }
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
  MS_EXCEPTION_IF_NULL(mem_manager_);
  runtime_instance_->SetContext();
  std::vector<size_t> align_size_list;
  for (size_t size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    align_size_list.emplace_back(align_size);
  }
  if (auto_mem_offload_ != nullptr) {
    return auto_mem_offload_->MallocContinuousMem(align_size_list);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(align_size_list);
}

DeviceAddressPtr AscendDeviceResManager::CreateDeviceAddress(void *const device_ptr, size_t device_size,
                                                             const string &format, TypeId type_id,
                                                             const ShapeVector &shape,
                                                             const UserDataPtr &user_data) const {
  auto device_address = std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id,
                                                              device_context_->device_context_key().device_name_,
                                                              device_context_->device_context_key().device_id_);
  if (shape.empty()) {
    MS_LOG(DEBUG) << "shape size is empty.";
  }
  device_address->set_host_shape(shape);
  return device_address;
}

bool AscendDeviceResManager::CreateStream(size_t *stream_id) const {
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool AscendDeviceResManager::DestroyStream(size_t stream_id) const {
  return AscendStreamMng::GetInstance().DestroyStream(stream_id);
}

bool AscendDeviceResManager::SyncStream(size_t stream_id) const {
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool AscendDeviceResManager::SyncAllStreams() const { return AscendStreamMng::GetInstance().SyncAllStreams(); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
