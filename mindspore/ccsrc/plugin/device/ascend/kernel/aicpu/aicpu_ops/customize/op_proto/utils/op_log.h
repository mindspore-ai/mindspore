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

/*!
 * \file op_log.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTILS_OP_LOG_H
#define CUSTOMIZE_OP_PROTO_UTILS_OP_LOG_H

#include <string>
#include <type_traits>
#include <cinttypes>
#include "graph/operator.h"

#define LOG_CPP
#if !defined(__ANDROID__) && !defined(ANDROID)
#include "toolchain/slog.h"
#else
#include <utils/Log.h>
#endif

#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#define OPPROTO_SUBMOD_NAME "OP_PROTO"

class OpLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline const char *get_cstr(const std::string &str) { return str.c_str(); }

inline const char *get_cstr(const char *str) { return str; }

inline const std::string &get_op_info(const std::string &str) { return str; }

inline const char *get_op_info(const char *str) { return str; }

template <class T>
constexpr bool is_ge_operator_type() {
  return std::is_base_of<ge::Operator, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<is_ge_operator_type<T>(), std::string>::type get_op_info(const T &op) {
  ge::AscendString name;
  ge::AscendString type;
  auto get_name_ret = op.GetName(name);
  auto get_type_ret = op.GetOpType(type);
  std::string op_info = get_type_ret == ge::GRAPH_SUCCESS ? type.GetString() : "nil";
  op_info += ":";
  op_info += get_name_ret == ge::GRAPH_SUCCESS ? name.GetString() : "nil";
  return op_info;
}

template <class T>
constexpr bool is_context_type() {
  return !std::is_base_of<ge::Operator, typename std::decay<T>::type>::value &&
         !std::is_same<const char *, typename std::decay<T>::type>::value &&
         !std::is_same<char *, typename std::decay<T>::type>::value &&
         !std::is_same<std::string, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<is_context_type<T>(), std::string>::type get_op_info(T context) {
  if (context == nullptr) {
    return "nil:nil";
  }
  std::string op_info = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
  op_info += ":";
  op_info += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
  return op_info;
}

template <typename T>
std::string TbeGetName(const T &op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

template <typename T>
std::string TbeGetOpType(const T &op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetOpType(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

#define CHECK_DIVISOR_ZERO(divisor) \
  if ((divisor) == 0) {             \
    return;                         \
  }

#define CHECK_DIVISOR_ZERO_RET(divisor, ret) \
  if ((divisor) == 0) {                      \
    return ret;                              \
  }

#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
  }

#if !defined(__ANDROID__) && !defined(ANDROID)
#define AICPU_OP_LOGI(opname, ...) AICPU_D_OP_LOGI(get_op_info(opname), __VA_ARGS__)
#define AICPU_OP_LOGW(opname, ...) AICPU_D_OP_LOGW(get_op_info(opname), __VA_ARGS__)
#define AICPU_OP_LOGD(opname, ...) AICPU_D_OP_LOGD(get_op_info(opname), __VA_ARGS__)
#define AICPU_OP_LOGE_WITHOUT_REPORT(opname, ...) AICPU_D_OP_LOGE(get_op_info(opname), __VA_ARGS__)
#define AICPU_OP_LOGE(op_name, ...)                       \
  do {                                                    \
    AICPU_OP_LOGE_WITHOUT_REPORT(op_name, ##__VA_ARGS__); \
  } while (0)

#define OP_LOGI(opname, ...) D_OP_LOGI(get_op_info(opname), __VA_ARGS__)
#define OP_LOGW(opname, ...) D_OP_LOGW(get_op_info(opname), __VA_ARGS__)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) D_OP_LOGE(get_op_info(opname), __VA_ARGS__)
#define OP_LOGE(op_name, ...)                       \
  do {                                              \
    OP_LOGE_WITHOUT_REPORT(op_name, ##__VA_ARGS__); \
  } while (0)

#define OP_LOGD(opname, ...) D_OP_LOGD(get_op_info(opname), __VA_ARGS__)
#define OP_EVENT(opname, ...) D_OP_EVENT(get_op_info(opname), __VA_ARGS__)
#define GE_OP_LOGI(opname, ...) GE_D_OP_LOGI(get_op_info(opname), __VA_ARGS__)
#define GE_OP_LOGW(opname, ...) GE_D_OP_LOGW(get_op_info(opname), __VA_ARGS__)
#define GE_OP_LOGE(opname, ...) GE_D_OP_LOGE(get_op_info(opname), __VA_ARGS__)
#define GE_OP_LOGD(opname, ...) GE_D_OP_LOGD(get_op_info(opname), __VA_ARGS__)
#define FUSION_PASS_LOGI(...) D_FUSION_PASS_LOGI(__VA_ARGS__)
#define FUSION_PASS_LOGW(...) D_FUSION_PASS_LOGW(__VA_ARGS__)
#define FUSION_PASS_LOGE(...) D_FUSION_PASS_LOGE(__VA_ARGS__)
#define FUSION_PASS_LOGD(...) D_FUSION_PASS_LOGD(__VA_ARGS__)
#else
#define AICPU_OP_LOGI(opname, ...)
#define AICPU_OP_LOGW(opname, ...)
#define AICPU_OP_LOGE(opname, ...)
#define AICPU_OP_LOGD(opname, ...)
#define AICPU_OP_LOGE_WITHOUT_REPORT(opname, ...)
#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)
#define OP_LOGE_WITHOUT_REPORT(opname, ...)
#define OP_LOGE(opname, ...)
#define OP_LOGD(opname, ...)
#define OP_EVENT(opname, ...)
#define FUSION_PASS_LOGI(...)
#define FUSION_PASS_LOGW(...)
#define FUSION_PASS_LOGE(...)
#define FUSION_PASS_LOGD(...)
#endif

#define OpLogSub(moduleId, level, op_info, fmt, ...)                                                                   \
  DlogSub(static_cast<int>(moduleId), OPPROTO_SUBMOD_NAME, level, "[%s][%" PRIu64 "] OpName:[%s] " #fmt, __FUNCTION__, \
          OpLog::GetTid(), get_cstr(op_info), ##__VA_ARGS__)

#if !defined(__ANDROID__) && !defined(ANDROID)
#define AICPU_D_OP_LOGI(opname, fmt, ...) OpLogSub(AICPU, DLOG_INFO, opname, fmt, ##__VA_ARGS__)
#define AICPU_D_OP_LOGW(opname, fmt, ...) OpLogSub(AICPU, DLOG_WARN, opname, fmt, ##__VA_ARGS__)
#define AICPU_D_OP_LOGE(opname, fmt, ...) OpLogSub(AICPU, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define AICPU_D_OP_LOGD(opname, fmt, ...) OpLogSub(AICPU, DLOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGI(opname, fmt, ...) OpLogSub(OP, DLOG_INFO, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) OpLogSub(OP, DLOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opname, fmt, ...) OpLogSub(OP, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) OpLogSub(OP, DLOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_OP_EVENT(opname, fmt, ...) OpLogSub(OP, DLOG_EVENT, opname, fmt, ##__VA_ARGS__)
#define GE_D_OP_LOGI(opname, fmt, ...) OpLogSub(GE, DLOG_INFO, opname, fmt, ##__VA_ARGS__)
#define GE_D_OP_LOGW(opname, fmt, ...) OpLogSub(GE, DLOG_WARN, opname, fmt, ##__VA_ARGS__)
#define GE_D_OP_LOGE(opname, fmt, ...) OpLogSub(GE, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define GE_D_OP_LOGD(opname, fmt, ...) OpLogSub(GE, DLOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGI(fmt, ...) \
  DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_INFO, " %s:%d " #fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGW(fmt, ...) \
  DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_WARN, " %s:%d " #fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGE(fmt, ...) \
  DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_ERROR, " %s:%d " #fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGD(fmt, ...) \
  DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_DEBUG, " %s:%d " #fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define AICPU_D_OP_LOGI(opname, fmt, ...)
#define AICPU_D_OP_LOGW(opname, fmt, ...)
#define AICPU_D_OP_LOGE(opname, fmt, ...)
#define AICPU_D_OP_LOGD(opname, fmt, ...)
#define D_OP_LOGI(opname, fmt, ...)
#define D_OP_LOGW(opname, fmt, ...)
#define D_OP_LOGE(opname, fmt, ...)
#define D_OP_LOGD(opname, fmt, ...)
#define D_OP_EVENT(opname, fmt, ...)
#define D_FUSION_PASS_LOGI(fmt, ...)
#define D_FUSION_PASS_LOGW(fmt, ...)
#define D_FUSION_PASS_LOGE(fmt, ...)
#define D_FUSION_PASS_LOGD(fmt, ...)
#endif

#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OP_LOGW_IF(condition, op_name, fmt, ...)                                                               \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGW(op_name, fmt, ##__VA_ARGS__);                                                                    \
    }                                                                                                          \
  } while (0)

#define OP_LOGI_IF_RETURN(condition, return_value, op_name, fmt, ...)                                          \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGI(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#endif  // OPS_COMMON_INC_OP_LOG_H_
