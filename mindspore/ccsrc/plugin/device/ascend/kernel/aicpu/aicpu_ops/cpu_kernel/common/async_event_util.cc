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

#include "cpu_kernel/common/async_event_util.h"
#include <dlfcn.h>
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace {
const char *kSharderPath = "/usr/lib64/libaicpu_sharder.so";
const char *kNotifyWaitFunc = "AicpuNotifyWait";
const char *kRegEventCbFunc = "AicpuRegEventCb";
const char *kRegEventCbWithTimesFunc = "AicpuRegEventCbWithTimes";
const char *kUnregEventCbFunc = "AicpuUnregEventCb";
}  // namespace

namespace aicpu {
AsyncEventUtil &AsyncEventUtil::GetInstance() {
  static AsyncEventUtil async_event_util;
  return async_event_util;
}

void AsyncEventUtil::InitEventUtil() {
  notify_wait_func_ = reinterpret_cast<NotifyWaitFunc>(dlsym(sharder_, kNotifyWaitFunc));
  if (notify_wait_func_ == nullptr) {
    KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kNotifyWaitFunc, dlerror());
  }
  reg_event_cb_func_ = reinterpret_cast<RegEventCbFunc>(dlsym(sharder_, kRegEventCbFunc));
  if (reg_event_cb_func_ == nullptr) {
    KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kRegEventCbFunc, dlerror());
  }
  reg_event_cb_with_times_func_ = reinterpret_cast<RegEventCbWithTimesFunc>(dlsym(sharder_, kRegEventCbWithTimesFunc));
  if (reg_event_cb_with_times_func_ == nullptr) {
    KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kRegEventCbWithTimesFunc, dlerror());
  }
  unreg_event_cb_func_ = reinterpret_cast<UnregEventCbFunc>(dlsym(sharder_, kUnregEventCbFunc));
  if (unreg_event_cb_func_ == nullptr) {
    KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kUnregEventCbFunc, dlerror());
  }
}

AsyncEventUtil::AsyncEventUtil() {
  sharder_ = dlopen(kSharderPath, RTLD_LAZY | RTLD_GLOBAL);
  if (sharder_ == nullptr) {
    KERNEL_LOG_WARN("Device sharder dlopen so [%s] failed, error[%s]", kSharderPath, dlerror());
    notify_wait_func_ = nullptr;
    reg_event_cb_func_ = nullptr;
    reg_event_cb_with_times_func_ = nullptr;
    unreg_event_cb_func_ = nullptr;
  } else {
    InitEventUtil();
    KERNEL_LOG_INFO("Device sharder dlopen so[%s] success.", kSharderPath);
  }
}

AsyncEventUtil::~AsyncEventUtil() {
  if (sharder_ != nullptr) {
    (void)dlclose(sharder_);
  }
}

void AsyncEventUtil::NotifyWait(void *notify_param, const uint32_t param_len) const {
  if (notify_wait_func_ != nullptr) {
    notify_wait_func_(notify_param, param_len);
    return;
  }
  KERNEL_LOG_WARN("Function[%s] is null", kNotifyWaitFunc);
}

bool AsyncEventUtil::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb) {
  if (reg_event_cb_func_ != nullptr) {
    return reg_event_cb_func_(event_id, sub_event_id, cb);
  }
  KERNEL_LOG_WARN("Function[%s] is null.", kRegEventCbFunc);
  return false;
}

bool AsyncEventUtil::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb, const int32_t times) {
  if (reg_event_cb_with_times_func_ != nullptr) {
    return reg_event_cb_with_times_func_(event_id, sub_event_id, cb, times);
  }
  KERNEL_LOG_WARN("Function[%s] is null.", kRegEventCbWithTimesFunc);
  return false;
}

void AsyncEventUtil::UnregEventCb(const uint32_t event_id, const uint32_t sub_event_id) {
  if (unreg_event_cb_func_ != nullptr) {
    return unreg_event_cb_func_(event_id, sub_event_id);
  }
  KERNEL_LOG_WARN("Function[%s] is null.", kUnregEventCbFunc);
}
}  // namespace aicpu
