/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "aicpu_sharder/aicpu_pulse.h"
#include <unordered_map>
#include <mutex>
#include <string>
#include "common/kernel_log.h"

namespace {
static std::unordered_map<std::string, PulseNotifyFunc> pulse_notify_func_map;
static std::mutex mtx;
}  // namespace

__attribute__((visibility("default"))) void AicpuPulseNotify() {
  std::unique_lock<std::mutex> lck(mtx);
  AICPU_LOGD("Aicpu pulse notify start, notify func num=%zu.", pulse_notify_func_map.size());
  for (auto &notify_func : pulse_notify_func_map) {
    AICPU_LOGD("Aicpu pulse notify %s start.", notify_func.first.c_str());
    notify_func.second();
    AICPU_LOGD("Aicpu pulse notify %s end.", notify_func.first.c_str());
  }
  AICPU_LOGD("Aicpu pulse notify end.");
}

__attribute__((visibility("default"))) int32_t RegisterPulseNotifyFunc(const char *name, PulseNotifyFunc func) {
  if (name == nullptr) {
    AICPU_LOGE("Register pulse notify func failed as param name is null");
    return -1;
  }

  if (func == nullptr) {
    AICPU_LOGE("Register pulse notify func for %s failed as param func is null", name);
    return -1;
  }

  std::unique_lock<std::mutex> lck(mtx);
  auto ret = pulse_notify_func_map.emplace(name, func);
  if (!ret.second) {
    AICPU_LOGE("Register pulse notify func for %s failed.", name);
    return -1;
  }
  AICPU_LOGI("Register pulse notify func for %s success.", name);
  return 0;
}
