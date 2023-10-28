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
#include "plugin/device/ascend/hal/device/ascend_runtime_manager.h"

namespace mindspore::device::ascend {
AscendRuntimeManager &AscendRuntimeManager::Instance() {
  static AscendRuntimeManager instance{};
  return instance;
}

void AscendRuntimeManager::Register(const std::string &device_name, AscendRuntimeCreator &&runtime_creator) const {
  if (runtime_creators_.find(device_name) == runtime_creators_.end()) {
    (void)runtime_creators_.emplace(device_name, runtime_creator);
  }
}

AscendKernelRuntime *AscendRuntimeManager::GetAscendRuntime(const std::string &device_name, uint32_t device_id) {
  auto runtime_key = device_name + "_" + std::to_string(device_id);
  std::lock_guard<std::mutex> guard(lock_);
  auto runtime_iter = runtime_map_.find(runtime_key);
  if (runtime_iter != runtime_map_.end()) {
    return runtime_iter->second.get();
  }
  std::shared_ptr<AscendKernelRuntime> ascend_runtime;
  auto creator_iter = runtime_creators_.find(device_name);
  if (creator_iter != runtime_creators_.end()) {
    MS_EXCEPTION_IF_NULL(creator_iter->second);
    ascend_runtime = (creator_iter->second)();
    MS_EXCEPTION_IF_NULL(ascend_runtime);
    ascend_runtime->set_device_id(device_id);
    runtime_map_[runtime_key] = ascend_runtime;
  } else {
    MS_LOG(INFO) << "No ascend runtime creator for " << device_name << " with device id " << device_id;
  }
  return ascend_runtime.get();
}
}  // namespace mindspore::device::ascend
