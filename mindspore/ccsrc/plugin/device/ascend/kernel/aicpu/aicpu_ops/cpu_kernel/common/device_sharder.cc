/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/common/device_sharder.h"

#include <dlfcn.h>
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace {
const char *kSharderPath = "/usr/lib64/libaicpu_sharder.so";
const char *kParallelForFunc = "ParallelFor";
const char *kGetCPUNumFunc = "GetCPUNum";
}  // namespace

namespace aicpu {
DeviceSharder::DeviceSharder(DeviceType device) : Sharder(device) {
  sharder_ = dlopen(kSharderPath, RTLD_LAZY | RTLD_GLOBAL);
  if (sharder_ == nullptr) {
    KERNEL_LOG_WARN("Device sharder dlopen so[%s] failed, error[%s]", kSharderPath, dlerror());
    parallel_for_ = nullptr;
    get_cpu_num_ = nullptr;
  } else {
    parallel_for_ = reinterpret_cast<ParallelForFunc>(dlsym(sharder_, kParallelForFunc));
    if (parallel_for_ == nullptr) {
      KERNEL_LOG_WARN("Get function[%s] address failed, error[%s]", kParallelForFunc, dlerror());
    }

    get_cpu_num_ = reinterpret_cast<GetCPUNumFunc>(dlsym(sharder_, kGetCPUNumFunc));
    if (get_cpu_num_ == nullptr) {
      KERNEL_LOG_WARN("Get function[%s] address failed, error[%s]", kGetCPUNumFunc, dlerror());
    }
    KERNEL_LOG_INFO("Device sharder dlopen so[%s] success", kSharderPath);
  }
}

DeviceSharder::~DeviceSharder() {
  if (sharder_ != nullptr) {
    (void)dlclose(sharder_);
    sharder_ = nullptr;
  }
  parallel_for_ = nullptr;
}

/*
 * ParallelFor shards the "total" units of work.
 */
void DeviceSharder::ParallelFor(int64_t total, int64_t perUnitSize,
                                const std::function<void(int64_t, int64_t)> &work) const {
  if (parallel_for_ != nullptr) {
    parallel_for_(total, perUnitSize, work);
    return;
  }

  KERNEL_LOG_WARN("Function[%s] is null", kParallelForFunc);
  work(0, total);
}

/*
 * Get CPU number
 */
uint32_t DeviceSharder::GetCPUNum() const {
  if (get_cpu_num_ != nullptr) {
    return get_cpu_num_();
  }

  KERNEL_LOG_WARN("Function[%s] is null", kGetCPUNumFunc);
  return 1;
}
}  // namespace aicpu
