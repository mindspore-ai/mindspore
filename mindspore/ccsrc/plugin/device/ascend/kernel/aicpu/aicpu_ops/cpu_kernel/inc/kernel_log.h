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
#include "inc/cust_cpu_utils.h"

namespace aicpu {
#define CUST_AICPU_LOGD(ctx, fmt, ...)                                                                     \
  CustCpuKernelUtils::CustLogDebug(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                   ##__VA_ARGS__)
#define CUST_AICPU_LOGI(ctx, fmt, ...) \
  CustCpuKernelUtils::CustLogInfo(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define CUST_AICPU_LOGW(ctx, fmt, ...)                                                                       \
  CustCpuKernelUtils::CustLogWarning(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                     ##__VA_ARGS__)
#define CUST_AICPU_LOGE(ctx, fmt, ...)                                                                     \
  CustCpuKernelUtils::CustLogError(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                   ##__VA_ARGS__)

#define CUST_KERNEL_LOG_WARN(ctx, fmt, ...) CUST_KERNEL_LOG_WARNING(ctx, fmt, ##__VA_ARGS__)

#define CUST_KERNEL_CHECK_NULLPTR_VOID(ctx, value, logText...) \
  if (value == nullptr) {                                      \
    CUST_AICPU_LOGE(ctx, logText);                             \
    return;                                                    \
  }

#define CUST_KERNEL_CHECK_FALSE(ctx, condition, errorCode, logText...) \
  if (!(condition)) {                                                  \
    CUST_AICPU_LOGE(ctx, logText);                                     \
    return errorCode;                                                  \
  }

#define CUST_KERNEL_CHECK_NULLPTR(ctx, value, errorCode, logText...) \
  if (value == nullptr) {                                            \
    CUST_AICPU_LOGE(ctx, logText);                                   \
    return errorCode;                                                \
  }

#define CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, A, B, result, errorCode)              \
  do {                                                                                \
    if ((A) != 0 && (B) != 0 && ((INT64_MAX) / (A)) <= (B)) {                         \
      CUST_AICPU_LOGE(ctx, "Integer reversed multiA: %llu * multiB: %llu", (A), (B)); \
      return errorCode;                                                               \
    }                                                                                 \
    (result) = ((A) * (B));                                                           \
  } while (0)

#define CUST_KERNEL_CHECK_FALSE_VOID(ctx, condition, logText...) \
  if (!(condition)) {                                            \
    CUST_AICPU_LOGE(ctx, logText);                               \
    return;                                                      \
  }

#define CUST_KERNEL_HANDLE_ERROR(ctx, expression, logText...) \
  ;                                                           \
  do {                                                        \
    uint32_t ret = expression;                                \
    if (ret != static_cast<uint32_t>(KERNEL_STATUS_OK)) {     \
      CUST_AICPU_LOGE(ctx, logText);                          \
      return ret;                                             \
    }                                                         \
  } while (0)

#define RETURN_IF_FAILURE(expr)    \
  do {                             \
    auto ret = (expr);             \
    if (ret != KERNEL_STATUS_OK) { \
      return ret;                  \
    }                              \
  } while (0)
/*
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

extern "C" {
__attribute__((visibility("default"))) void DlogInner(int moduleId, int level, const char *fmt, ...);
}

namespace aicpu {
#define AICPU_MODULE_NAME static_cast<int32_t>(AICPU)
#define KERNEL_MODULE "AICPU"

#define AICPU_LOG(level, fmt, ...)                                                                                \
  do {                                                                                                            \
    if (CheckLogLevel(AICPU, level) == 1) {                                                                       \
      auto log_func = DlogRecord == nullptr ? DlogInner : DlogRecord;                                             \
      log_func(AICPU, level, "[%s:%d][%s][tid:%lu]:" fmt, __FILE__, __LINE__, __func__, GetTid(), ##__VA_ARGS__); \
    }                                                                                                             \
  } while (0)

#define AICPU_LOGD(fmt, ...) AICPU_LOG(DLOG_DEBUG, fmt, ##__VA_ARGS__);
#define AICPU_LOGI(fmt, ...) AICPU_LOG(DLOG_INFO, fmt, ##__VA_ARGS__);
#define AICPU_LOGW(fmt, ...) AICPU_LOG(DLOG_WARN, fmt, ##__VA_ARGS__);
#define AICPU_LOGE(fmt, ...) AICPU_LOG(DLOG_ERROR, fmt, ##__VA_ARGS__);
#define AICPU_EVENT(fmt, ...) AICPU_LOG(DLOG_EVENT, fmt, ##__VA_ARGS__);

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

#define KERNEL_LOG_DEBUG(fmt, ...) AICPU_LOGD(fmt, ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...) AICPU_LOGI(fmt, ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...) AICPU_LOGW(fmt, ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...) AICPU_LOGE(fmt, ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...) AICPU_EVENT(fmt, ##__VA_ARGS__)

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

#define RETURN_IF_FAILURE(expr)    \
  do {                             \
    auto ret = (expr);             \
    if (ret != KERNEL_STATUS_OK) { \
      return ret;                  \
    }                              \
  } while (0)
*/
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KERNEL_LOG_H_
