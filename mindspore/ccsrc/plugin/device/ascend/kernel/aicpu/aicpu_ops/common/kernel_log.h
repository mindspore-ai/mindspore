/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "toolchain/slog.h"

inline int64_t GetTid(void) {
  thread_local static const int64_t tid = syscall(__NR_gettid);
  return tid;
}

namespace aicpu {
#define AICPU_MODULE_NAME static_cast<int32_t>(AICPU)
#define KERNEL_MODULE "AICPU"

#define AICPU_LOGD(fmt, ...)                                                                                  \
  dlog_debug(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);
#define AICPU_LOGI(fmt, ...)                                                                                 \
  dlog_info(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
            ##__VA_ARGS__);
#define AICPU_LOGW(fmt, ...)                                                                                 \
  dlog_warn(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
            ##__VA_ARGS__);
#define AICPU_LOGE(fmt, ...)                                                                                  \
  dlog_error(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);
#define AICPU_LOGEVENT(fmt, ...)                                                                              \
  dlog_event(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);

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

#define KERNEL_LOG_DEBUG(fmt, ...)                                                                            \
  dlog_debug(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);
#define KERNEL_LOG_INFO(fmt, ...)                                                                            \
  dlog_info(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
            ##__VA_ARGS__);
#define KERNEL_LOG_WARN(fmt, ...)                                                                            \
  dlog_warn(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
            ##__VA_ARGS__);
#define KERNEL_LOG_ERROR(fmt, ...)                                                                            \
  dlog_error(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);
#define KERNEL_LOG_EVENT(fmt, ...)                                                                            \
  dlog_event(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GetTid(), \
             ##__VA_ARGS__);

#define KERNEL_CHECK_NULLPTR_VOID(value, logText...) \
  if (value == nullptr) {                            \
    AICPU_LOGE(logText);                             \
    return;                                          \
  }

#define KERNEL_CHECK_FALSE(condition, errorCode, logText...) \
  if (!(condition)) {                                        \
    AICPU_LOGE(logText);                                     \
    return errorCode;                                        \
  }

#define KERNEL_CHECK_NULLPTR(value, errorCode, logText...) \
  if (value == nullptr) {                                  \
    AICPU_LOGE(logText);                                   \
    return errorCode;                                      \
  }

#define KERNEL_CHECK_ASSIGN_64S_MULTI(A, B, result, errorCode)              \
  do {                                                                      \
    if ((A) != 0 && (B) != 0 && ((INT64_MAX) / (A)) <= (B)) {               \
      AICPU_LOGE("Integer reversed multiA: %llu * multiB: %llu", (A), (B)); \
      return errorCode;                                                     \
    }                                                                       \
    (result) = ((A) * (B));                                                 \
  } while (0)

#define KERNEL_CHECK_FALSE_VOID(condition, logText...) \
  if (!(condition)) {                                  \
    AICPU_LOGE(logText);                               \
    return;                                            \
  }

#define KERNEL_HANDLE_ERROR(expression, logText...)       \
  ;                                                       \
  do {                                                    \
    uint32_t ret = expression;                            \
    if (ret != static_cast<uint32_t>(KERNEL_STATUS_OK)) { \
      AICPU_LOGE(logText);                                \
      return ret;                                         \
    }                                                     \
  } while (0)

#define KERNEL_CHECK_FALSE_EXEC(condition, execExpr...) \
  if (!(condition)) {                                   \
    execExpr;                                           \
  }
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KERNEL_LOG_H_
