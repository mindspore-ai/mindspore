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

#include "runtime/device/kernel_runtime_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
void KernelRuntimeManager::ClearRuntimeResource() {
  std::lock_guard<std::mutex> guard(lock_);
  for (auto &iter : runtime_map_) {
    MS_LOG(INFO) << "Release device " << iter.first;
    MS_EXCEPTION_IF_NULL(iter.second);
    iter.second->ReleaseDeviceRes();
  }
  runtime_map_.clear();
}

void KernelRuntimeManager::ClearGraphResource(uint32_t graph_id) {
  std::lock_guard<std::mutex> guard(lock_);
  for (auto &iter : runtime_map_) {
    MS_LOG(INFO) << "Clear device " << iter.first << " graph " << graph_id << " runtime resource";
    if (!iter.second) {
      MS_LOG(ERROR) << "Kernel runtime is nullptr";
      continue;
    }
    iter.second->ClearGraphRuntimeResource(graph_id);
  }
}

KernelRuntimeManager &KernelRuntimeManager::Instance() {
  static KernelRuntimeManager instance{};
  return instance;
}

void KernelRuntimeManager::Register(const std::string &device_name, KernelRuntimeCreator &&runtime_creator) const {
  if (runtime_creators_.find(device_name) == runtime_creators_.end()) {
    (void)runtime_creators_.emplace(device_name, runtime_creator);
  }
}

std::string KernelRuntimeManager::GetDeviceKey(const std::string &device_name, uint32_t device_id) const {
  std::string device_key = device_name + "_" + std::to_string(device_id);
  return device_key;
}

KernelRuntime *KernelRuntimeManager::GetSingleKernelRuntime(const std::string &device_name, uint32_t device_id) {
  auto runtime_key = GetDeviceKey(device_name, device_id);
  auto runtime_iter = runtime_map_.find(runtime_key);
  if (runtime_iter != runtime_map_.end()) {
    return runtime_iter->second.get();
  } else if (!runtime_map_.empty()) {
    auto cur_runtime_key = runtime_map_.begin()->first;
    auto find_pos = cur_runtime_key.rfind('_');
    if (find_pos != std::string::npos) {
      if (cur_runtime_key.size() > find_pos + 1) {
        auto cur_device_id = cur_runtime_key.substr(find_pos + 1);
        MS_LOG(EXCEPTION) << "Can't change device id in runtime, already set device id: " << cur_device_id
                          << ", set device id: " << device_id << " failed";
      } else {
        MS_LOG(EXCEPTION) << "Can't change device id in runtime, current runtime_key size error, set device id: "
                          << device_id << " failed";
      }
    }
  }
  return GetKernelRuntime(device_name, device_id);
}

KernelRuntime *KernelRuntimeManager::GetKernelRuntime(const std::string &device_name, uint32_t device_id) {
  std::string runtime_key = GetDeviceKey(device_name, device_id);
  std::lock_guard<std::mutex> guard(lock_);
  auto runtime_iter = runtime_map_.find(runtime_key);
  if (runtime_iter != runtime_map_.end()) {
    return runtime_iter->second.get();
  }
  std::shared_ptr<KernelRuntime> kernel_runtime;
  auto creator_iter = runtime_creators_.find(device_name);
  if (creator_iter != runtime_creators_.end()) {
    MS_EXCEPTION_IF_NULL(creator_iter->second);
    kernel_runtime = (creator_iter->second)();
    MS_EXCEPTION_IF_NULL(kernel_runtime);
    kernel_runtime->set_device_id(device_id);
    runtime_map_[runtime_key] = kernel_runtime;
  } else {
    MS_LOG(EXCEPTION) << "No kernel runtime creator for " << device_name << " with device id " << device_id;
  }

  return kernel_runtime.get();
}

KernelRuntime *KernelRuntimeManager::GetCurrentKernelRuntime() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  return GetKernelRuntime(device_name, device_id);
}

void KernelRuntimeManager::ReleaseKernelRuntime(const std::string &device_name, uint32_t device_id) {
  std::string runtime_key = GetDeviceKey(device_name, device_id);
  std::lock_guard<std::mutex> guard(lock_);
  auto runtime_iter = runtime_map_.find(runtime_key);
  if (runtime_iter == runtime_map_.end()) {
    return;
  }
  auto runtime = runtime_iter->second.get();
  if (runtime == nullptr) {
    return;
  }
  runtime->ReleaseDeviceRes();
  runtime_map_.erase(runtime_iter);
}

void KernelRuntimeManager::WaitTaskFinishOnDevice() const {
  for (const auto &iter : runtime_map_) {
    auto kernel_runtime = iter.second;
    try {
      if (kernel_runtime != nullptr && !kernel_runtime->SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return;
      }
    } catch (const std::exception &ex) {
      MS_LOG(ERROR) << "SyncStream failed, exception:" << ex.what();
      return;
    }
  }
}
}  // namespace device
}  // namespace mindspore
