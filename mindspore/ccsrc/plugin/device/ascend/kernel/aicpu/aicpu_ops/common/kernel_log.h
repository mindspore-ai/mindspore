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
#ifndef AICPU_OPS_AICPU_COMMON_KERNEL_LOG_H_
#define AICPU_OPS_AICPU_COMMON_KERNEL_LOG_H_

#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <utility>
#include "common/kernel_errcode.h"

inline int64_t GetTid(void) {
  thread_local static const int64_t tid = syscall(__NR_gettid);
  return tid;
}
static const int LOG_COUNT = 0;

namespace aicpu {
#define AICPU_LOG_DEBUG 0
#define AICPU_LOG_INFO 1
#define AICPU_LOG_WARN 2
#define AICPU_LOG_ERROR 3
#define AICPU_LOG_EVENT 0x10

inline void PrintLog(const int level) { std::cerr << level << std::endl; }

template <typename T, typename... Args>
inline void PrintLog(const int level, T &&head, Args &&... tail) {
  std::cerr << std::forward<T>(head) << " ";
  PrintLog(level, std::forward<Args>(tail)...);
}

int LogSetLevel(int level);

int LogGetLevel(void);

bool CheckLogLevel(int log_level_check);

#define AICPU_LOGD(fmt, ...) \
  AICPU_LOG(AICPU_LOG_DEBUG, "%s:%s:%d[tid:%lu]:" #fmt, __FUNCTION__, __FILE__, __LINE__, GetTid(), ##__VA_ARGS__);
#define AICPU_LOGI(fmt, ...) \
  AICPU_LOG(AICPU_LOG_INFO, "%s:%s:%d[tid:%lu]:" #fmt, __FUNCTION__, __FILE__, __LINE__, GetTid(), ##__VA_ARGS__);
#define AICPU_LOGW(fmt, ...) \
  AICPU_LOG(AICPU_LOG_WARN, "%s:%s:%d[tid:%lu]:" #fmt, __FUNCTION__, __FILE__, __LINE__, GetTid(), ##__VA_ARGS__);
#define AICPU_LOGE(fmt, ...) \
  AICPU_LOG(AICPU_LOG_ERROR, "%s:%s:%d[tid:%lu]:" #fmt, __FUNCTION__, __FILE__, __LINE__, GetTid(), ##__VA_ARGS__);
#define AICPU_LOGEVENT(fmt, ...) \
  AICPU_LOG(AICPU_LOG_EVENT, "%s:%s:%d[tid:%lu]:" #fmt, __FUNCTION__, __FILE__, __LINE__, GetTid(), ##__VA_ARGS__);
#define AICPU_LOG(level, fmt, ...)                                              \
  do {                                                                          \
    if (aicpu::CheckLogLevel(level)) {                                          \
      aicpu::PrintLog(level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    }                                                                           \
  } while (LOG_COUNT != 0)

#define AICPU_CHK_STATUS_RET(expr...)        \
  do {                                       \
    const uint32_t status = (expr);          \
    if (status != kAicpuKernelStateSucess) { \
      return status;                         \
    }                                        \
  } while (0);

#define AICPU_CHECK_NULLPTR_VOID(value, logText...) \
  if (value == nullptr) {                           \
    AICPU_LOGE(logText);                            \
    return;                                         \
  }

#define AICPU_CHECK_FALSE(condition, errorCode, logText...) \
  if (!(condition)) {                                       \
    AICPU_LOGE(logText);                                    \
    return errorCode;                                       \
  }

#define AICPU_CHECK_NULLPTR(value, errorCode, logText...) \
  if (value == nullptr) {                                 \
    AICPU_LOGE(logText);                                  \
    return errorCode;                                     \
  }
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KERNEL_LOG_H_
