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

#ifndef MINDSPORE_CCSRC_UTILS_LOG_ADAPTER_H_
#define MINDSPORE_CCSRC_UTILS_LOG_ADAPTER_H_

#include <stdarg.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <memory>
#include "./overload.h"
#include "./securec.h"
#ifdef USE_GLOG
#include "glog/logging.h"
#else
#include "toolchain/slog.h"
#endif
// NOTICE: when relative path of 'log_adapter.h' changed, macro 'LOG_HDR_FILE_REL_PATH' must be changed
#define LOG_HDR_FILE_REL_PATH "mindspore/ccsrc/utils/log_adapter.h"

// Get start index of file relative path in __FILE__
static constexpr int GetRelPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(LOG_HDR_FILE_REL_PATH) ? sizeof(__FILE__) - sizeof(LOG_HDR_FILE_REL_PATH) : 0;
}

namespace mindspore {
#define FILE_NAME                                                                             \
  (sizeof(__FILE__) > GetRelPathPos() ? static_cast<const char *>(__FILE__) + GetRelPathPos() \
                                      : static_cast<const char *>(__FILE__))
enum ExceptionType {
  NoExceptionType = 0,
  UnknownError,
  ArgumentError,
  NotSupportError,
  NotExistsError,
  AlreadyExistsError,
  UnavailableError,
  DeviceProcessError,
  AbortedError,
  TimeOutError,
  ResourceUnavailable,
  NoPermissionError,
  IndexError,
  ValueError,
  TypeError,
};

struct LocationInfo {
  LocationInfo(const char *file, int line, const char *func) : file_(file), line_(line), func_(func) {}
  ~LocationInfo() = default;

  const char *file_;
  int line_;
  const char *func_;
};

class LogStream {
 public:
  LogStream() { sstream_ = std::make_shared<std::stringstream>(); }
  ~LogStream() = default;

  template <typename T>
  LogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  LogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }

  friend class LogWriter;

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

template <class T, typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
constexpr std::ostream &operator<<(std::ostream &stream, const T &value) {
  return stream << static_cast<typename std::underlying_type<T>::type>(value);
}

enum MsLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR, EXCEPTION };

enum SubModuleId : int {
  SM_UNKNOWN = 0,  // unknown submodule
  SM_ANALYZER,     // static analyzer
  SM_COMMON,       // common
  SM_DEBUG,        // debug
  SM_DEVICE,       // device
  SM_GE_ADPT,      // ge adapter
  SM_IR,           // IR
  SM_KERNEL,       // kernel
  SM_MD,           // MindData
  SM_ME,           // MindExpression
  SM_ONNX,         // ONNX
  SM_OPTIMIZER,    // optimzer
  SM_PARALLEL,     // parallel
  SM_PARSER,       // parser
  SM_PIPELINE,     // ME pipeline
  SM_PRE_ACT,      // pre-activate
  SM_PYNATIVE,     // PyNative
  SM_SESSION,      // session
  SM_UTILS,        // utils
  SM_VM,           // VM
  NUM_SUBMODUES    // number of submodules
};

#ifndef SUBMODULE_ID
#define SUBMODULE_ID mindspore::SubModuleId::SM_ME
#endif

#if defined(_WIN32) || defined(_WIN64)
extern int g_ms_submodule_log_levels[] __attribute__((dllexport));
#else
extern int g_ms_submodule_log_levels[] __attribute__((visibility("default")));
#endif

class LogWriter {
 public:
  LogWriter(const LocationInfo &location, MsLogLevel log_level, SubModuleId submodule,
            ExceptionType excp_type = NoExceptionType)
      : location_(location), log_level_(log_level), submodule_(submodule), exception_type_(excp_type) {}
  ~LogWriter() = default;

  void operator<(const LogStream &stream) const noexcept __attribute__((visibility("default")));
  void operator^(const LogStream &stream) const __attribute__((noreturn, visibility("default")));

 private:
  void OutputLog(const std::ostringstream &msg) const;

  LocationInfo location_;
  MsLogLevel log_level_;
  SubModuleId submodule_;
  ExceptionType exception_type_;
};

#define MSLOG_IF(level, condition, excp_type)                                                                       \
  static_cast<void>(0), !(condition)                                                                                \
                          ? void(0)                                                                                 \
                          : mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), level, \
                                                 SUBMODULE_ID, excp_type) < mindspore::LogStream()
#define MSLOG_THROW(excp_type)                                                                                         \
  mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), mindspore::EXCEPTION, SUBMODULE_ID, \
                       excp_type) ^                                                                                    \
    mindspore::LogStream()

#define IS_OUTPUT_ON(level) (level) >= mindspore::g_ms_submodule_log_levels[SUBMODULE_ID]

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_DEBUG MSLOG_IF(mindspore::DEBUG, IS_OUTPUT_ON(mindspore::DEBUG), mindspore::NoExceptionType)
#define MS_LOG_INFO MSLOG_IF(mindspore::INFO, IS_OUTPUT_ON(mindspore::INFO), mindspore::NoExceptionType)
#define MS_LOG_WARNING MSLOG_IF(mindspore::WARNING, IS_OUTPUT_ON(mindspore::WARNING), mindspore::NoExceptionType)
#define MS_LOG_ERROR MSLOG_IF(mindspore::ERROR, IS_OUTPUT_ON(mindspore::ERROR), mindspore::NoExceptionType)

#define MS_LOG_EXCEPTION MSLOG_THROW(mindspore::NoExceptionType)
#define MS_EXCEPTION(type) MSLOG_THROW(type)
}  // namespace mindspore

#define MS_EXCEPTION_IF_NULL(ptr)                                    \
  do {                                                               \
    if ((ptr) == nullptr) {                                          \
      MS_LOG(EXCEPTION) << ": The pointer[" << #ptr << "] is null."; \
    }                                                                \
  } while (0)

#ifdef DEBUG
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

#endif  // MINDSPORE_CCSRC_UTILS_LOG_ADAPTER_H_
