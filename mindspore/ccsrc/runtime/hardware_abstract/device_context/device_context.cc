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

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include <memory>
#include <utility>
#include <vector>
#include "utils/ms_context.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace device {
namespace {
std::vector<std::shared_ptr<ExecutorCallback<PreLaunchKernelFunc>>> g_pre_launch_kernel_callbacks;
std::vector<std::shared_ptr<ExecutorCallback<PostLaunchKernelFunc>>> g_post_launch_kernel_callbacks;

template <typename CallbackType>
void RegisterKernelCallback(std::vector<std::shared_ptr<ExecutorCallback<CallbackType>>> *ptr_cb_vector,
                            const std::shared_ptr<ExecutorCallback<CallbackType>> &callback) {
  ptr_cb_vector->emplace_back(callback);
}

template <typename CallbackType>
const std::vector<std::shared_ptr<ExecutorCallback<CallbackType>>> &GetKernelCallbacks(
  const std::vector<std::shared_ptr<ExecutorCallback<CallbackType>>> &cb_list) {
  static std::vector<std::shared_ptr<ExecutorCallback<CallbackType>>> callbacks;
  static std::once_flag init_flag{};
  std::call_once(init_flag, [&]() {
    // filter enabled callbacks
    for (auto &obj : cb_list) {
      if (obj->fn_is_enabled_()) {
        callbacks.emplace_back(obj);
      }
    }
    // sort callbacks by priority from high to low, larger value means higher priority
    std::sort(callbacks.begin(), callbacks.end(),
              [](const auto &a, const auto &b) -> bool { return a->priority_ > b->priority_; });
  });
  return callbacks;
}
}  // namespace

template <>
RUNTIME_HARDWARE_EXPORT void KernelExecutor::RegisterLaunchKernelCallback<PreLaunchKernelFunc>(
  const std::shared_ptr<ExecutorCallback<PreLaunchKernelFunc>> &callback) {
  RegisterKernelCallback<PreLaunchKernelFunc>(&g_pre_launch_kernel_callbacks, callback);
}

template <>
RUNTIME_HARDWARE_EXPORT void KernelExecutor::RegisterLaunchKernelCallback<PostLaunchKernelFunc>(
  const std::shared_ptr<ExecutorCallback<PostLaunchKernelFunc>> &callback) {
  RegisterKernelCallback<PostLaunchKernelFunc>(&g_post_launch_kernel_callbacks, callback);
}

template <>
RUNTIME_HARDWARE_EXPORT const std::vector<std::shared_ptr<ExecutorCallback<PreLaunchKernelFunc>>> &
KernelExecutor::GetLaunchKernelCallbacks() {
  return GetKernelCallbacks<PreLaunchKernelFunc>(g_pre_launch_kernel_callbacks);
}

template <>
RUNTIME_HARDWARE_EXPORT const std::vector<std::shared_ptr<ExecutorCallback<PostLaunchKernelFunc>>> &
KernelExecutor::GetLaunchKernelCallbacks() {
  return GetKernelCallbacks<PostLaunchKernelFunc>(g_post_launch_kernel_callbacks);
}

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
template class RUNTIME_HARDWARE_EXPORT ExecutorCallback<PreLaunchKernelFunc>;
template class RUNTIME_HARDWARE_EXPORT ExecutorCallback<PostLaunchKernelFunc>;
#endif

DeviceResManager::DeviceResManager() {
  collective_comm_lib_ = nullptr;
  device_context_ = nullptr;
}

bool DeviceContext::initialized() const {
  runtime::Pipeline::Get().WaitForward();
  return initialized_;
}

bool DeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected in device address:" << address->ToString();
    return false;
  }

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }
  const auto &allocator = address->allocator();
  void *device_ptr = nullptr;
  if (allocator != nullptr) {
    device_ptr = allocator->Alloc(address->GetSize(), stream_id);
  } else {
    device_ptr = AllocateMemory(address->GetSize(), stream_id);
  }
  if (device_ptr == nullptr) {
    MS_LOG(WARNING) << "Allocate memory failed for size: " << address->GetSize();
    return false;
  }
  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  return true;
}

void DeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr is null in device address:" << address << " to release!";
  }

  if (!address->from_mem_pool()) {
    MS_LOG(DEBUG) << "device address:" << address << " ptr:" << address->GetMutablePtr() << " not from pool";
    return;
  }
  MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << address->GetMutablePtr();
  std::shared_ptr<AddressAllocator> allocator = address->allocator();
  if (allocator != nullptr) {
    allocator->Free(address->GetMutablePtr());
  } else {
    FreeMemory(address->GetMutablePtr());
  }
  address->set_ptr(nullptr);
}

uint8_t *DeviceResManager::MallocWorkSpaceMem(size_t size) { return MallocDynamicMem(size, false); }

uint8_t *DeviceResManager::MallocDynamicMem(size_t size, bool communication_mem) {
  MS_LOG(INFO) << "Call default dynamic malloc " << size << " v " << communication_mem;
  return nullptr;
}
}  // namespace device
}  // namespace mindspore
