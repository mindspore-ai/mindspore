/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_LOG_ADAPTER_H_
#define MINDSPORE_CORE_UTILS_LOG_ADAPTER_H_

#include <cstdarg>
#include <cstdint>
#include <string>
#include <sstream>
#include <memory>
#include <map>
#include <vector>
#include <thread>
#include <functional>
#include "utils/macros.h"
#include "utils/os.h"
#include "utils/overload.h"
#include "./securec.h"
#ifdef USE_GLOG
#define GLOG_NO_ABBREVIATED_SEVERITIES
#define google mindspore_private
#include "glog/logging.h"
#undef google
#elif defined(BUILD_CORE_RUNTIME)
#else
#include "toolchain/slog.h"
#endif

#undef SM_DEBUG

// NOTICE: when relative path of 'log_adapter.h' changed, macro 'LOG_HDR_FILE_REL_PATH' must be changed
#ifndef LOG_HDR_FILE_REL_PATH
#define LOG_HDR_FILE_REL_PATH "mindspore/core/utils/log_adapter.h"
#endif
// Get start index of file relative path in __FILE__
static constexpr size_t GetRelPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(LOG_HDR_FILE_REL_PATH) ? sizeof(__FILE__) - sizeof(LOG_HDR_FILE_REL_PATH) : 0;
}
namespace mindspore {
/// \brief The handler map for ACL.
#define FILE_NAME                                                                             \
  (sizeof(__FILE__) > GetRelPathPos() ? static_cast<const char *>(__FILE__) + GetRelPathPos() \
                                      : static_cast<const char *>(__FILE__))
enum ExceptionType {
  NoExceptionType = 0,
  UnknownError,
  ArgumentError,
  NotSupportError,
  NotExistsError,
  DeviceProcessError,
  AbortedError,
  IndexError,
  ValueError,
  TypeError,
  KeyError,
  AttributeError,
  NameError,
  AssertionError,
  BaseException,
  KeyboardInterrupt,
  Exception,
  StopIteration,
  OverflowError,
  ZeroDivisionError,
  EnvironmentError,
  IOError,
  OSError,
  ImportError,
  MemoryError,
  UnboundLocalError,
  RuntimeError,
  NotImplementedError,
  IndentationError,
  RuntimeWarning,
};

static const inline std::map<std::string, ExceptionType> exception_types_map = {
  {"IndexError", IndexError},
  {"ValueError", ValueError},
  {"TypeError", TypeError},
  {"KeyError", KeyError},
  {"AttributeError", AttributeError},
  {"NameError", NameError},
  {"AssertionError", AssertionError},
  {"BaseException", BaseException},
  {"KeyboardInterrupt", KeyboardInterrupt},
  {"Exception", Exception},
  {"StopIteration", StopIteration},
  {"OverflowError", OverflowError},
  {"ZeroDivisionError", ZeroDivisionError},
  {"EnvironmentError", EnvironmentError},
  {"IOError", IOError},
  {"OSError", OSError},
  {"MemoryError", MemoryError},
  {"UnboundLocalError", UnboundLocalError},
  {"RuntimeError", RuntimeError},
  {"NotImplementedError", NotImplementedError},
  {"IndentationError", IndentationError},
  {"RuntimeWarning", RuntimeWarning}};

static inline std::string SupportedExceptionsToString() {
  std::ostringstream oss;
  size_t index = 0;
  for (auto iter = exception_types_map.cbegin(); iter != exception_types_map.cend(); ++iter) {
    oss << iter->first;
    if (index != exception_types_map.size() - 1) {
      oss << ", ";
    }
    ++index;
  }
  oss << ". ";
  return oss.str();
}

struct LocationInfo {
  LocationInfo(const char *file, int line, const char *func) : file_(file), line_(line), func_(func) {}
  ~LocationInfo() = default;

  const char *file_;
  int line_;
  const char *func_;
};

template <class T, typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
constexpr std::ostream &operator<<(std::ostream &stream, const T &value) {
  return stream << static_cast<typename std::underlying_type<T>::type>(value);
}

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

enum MsLogLevel : int { kDebug = 0, kInfo, kWarning, kError, kException };

enum SubModuleId : int {
  SM_UNKNOWN = 0,        // unknown submodule
  SM_CORE,               // core
  SM_ANALYZER,           // static analyzer
  SM_COMMON,             // common
  SM_DEBUG,              // debug
  SM_OFFLINE_DEBUG,      // offline debug
  SM_DEVICE,             // device
  SM_GE_ADPT,            // ge adapter
  SM_IR,                 // IR
  SM_KERNEL,             // kernel
  SM_MD,                 // MindData
  SM_ME,                 // MindExpression
  SM_EXPRESS,            // EXPRESS_IR
  SM_OPTIMIZER,          // optimzer
  SM_PARALLEL,           // parallel
  SM_PARSER,             // parser
  SM_PIPELINE,           // ME pipeline
  SM_PRE_ACT,            // pre-activate
  SM_PYNATIVE,           // PyNative
  SM_SESSION,            // session
  SM_UTILS,              // utils
  SM_VM,                 // VM
  SM_PROFILER,           // profiler
  SM_PS,                 // Parameter Server
  SM_FL,                 // Federated Learning
  SM_DISTRIBUTED,        // Distributed
  SM_LITE,               // LITE
  SM_ARMOUR,             // ARMOUR
  SM_HCCL_ADPT,          // Hccl Adapter
  SM_RUNTIME_FRAMEWORK,  // Runtime framework
  SM_GE,                 // GraphEngine
  SM_API,                // MindAPI
  NUM_SUBMODUES          // number of submodules
};

#ifndef SUBMODULE_ID
#ifndef BUILD_LITE_INFERENCE
#define SUBMODULE_ID mindspore::SubModuleId::SM_ME
#else
#define SUBMODULE_ID mindspore::SubModuleId::SM_LITE
#endif
#endif

/// \brief Get sub-module name by the module id.
///
/// \param[in] module_id The module id.
///
/// \return The sub-module name.
MS_EXPORT const std::string GetSubModuleName(SubModuleId module_id);

MS_CORE_API void InitSubModulesLogLevel();

/// \brief Get current time as a string.
///
/// \return The string presents current time.
MS_EXPORT std::string GetTimeString();

/// \brief The log levels of mindspore sub-module.
MS_EXPORT extern int g_ms_submodule_log_levels[];

#if defined(_WIN32) || defined(_WIN64)
/// \brief The max log level of current thread.
MS_EXPORT extern enum MsLogLevel this_thread_max_log_level;
#else
/// \brief The max log level of current thread.
MS_EXPORT extern thread_local enum MsLogLevel this_thread_max_log_level;
#endif
class TryCatchGuard {
 public:
  TryCatchGuard() : origin_log_level_(this_thread_max_log_level) { this_thread_max_log_level = MsLogLevel::kWarning; }
  ~TryCatchGuard() { this_thread_max_log_level = origin_log_level_; }

 private:
  enum MsLogLevel origin_log_level_;
};
#define MS_LOG_TRY_CATCH_SCOPE mindspore::TryCatchGuard mindspore_log_try_catch_guard

/// \brief LogWriter defines interface to write log.
class MS_CORE_API LogWriter {
 public:
  using ExceptionHandler = void (*)(ExceptionType, const std::string &);
  using TraceProvider = std::function<void(std::ostringstream &oss, bool add_title)>;

  LogWriter(const LocationInfo &location, MsLogLevel log_level, SubModuleId submodule,
            ExceptionType excp_type = NoExceptionType)
      : location_(location), log_level_(log_level), submodule_(submodule), exception_type_(excp_type) {}
  ~LogWriter() = default;

  /// \brief Output log message from the input log stream.
  ///
  /// \param[in] stream The input log stream.
  void operator<(const LogStream &stream) const noexcept;

#if !defined(BUILD_LITE_INFERENCE) || defined(BUILD_CORE_RUNTIME)
  /// \brief Output log message from the input log stream and then throw exception.
  ///
  /// \param[in] stream The input log stream.
  void operator^(const LogStream &stream) const NO_RETURN;
#endif

  static void set_exception_handler(const ExceptionHandler &exception_handler);
  static void set_trace_provider(const TraceProvider &trace_provider);
  static TraceProvider trace_provider();

 private:
  void OutputLog(const std::ostringstream &msg) const;
  void RemoveLabelBeforeOutputLog(const std::ostringstream &msg) const;

  LocationInfo location_;
  MsLogLevel log_level_;
  SubModuleId submodule_;
  ExceptionType exception_type_;

  inline static ExceptionHandler exception_handler_ = nullptr;
  inline static TraceProvider trace_provider_ = nullptr;
};

#define MSLOG_IF(level, condition, excp_type)                                                                          \
  !(condition) ? void(0)                                                                                               \
               : mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), level, SUBMODULE_ID, \
                                      excp_type) < mindspore::LogStream()

#if !defined(BUILD_LITE_INFERENCE) || defined(BUILD_CORE_RUNTIME)
#define MSLOG_THROW(excp_type)                                                                            \
  mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), mindspore::kException, \
                       SUBMODULE_ID, excp_type) ^                                                         \
    mindspore::LogStream()
#else
#define MSLOG_THROW(excp_type)                                                                                      \
  mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), mindspore::kError, SUBMODULE_ID, \
                       excp_type) < mindspore::LogStream()
#endif

inline bool IS_OUTPUT_ON(enum MsLogLevel level) noexcept(true) {
  return (static_cast<int>(level) >= mindspore::g_ms_submodule_log_levels[SUBMODULE_ID] &&
          static_cast<int>(level) <= static_cast<int>(mindspore::this_thread_max_log_level));
}

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_DEBUG MSLOG_IF(mindspore::kDebug, IS_OUTPUT_ON(mindspore::kDebug), mindspore::NoExceptionType)
#define MS_LOG_INFO MSLOG_IF(mindspore::kInfo, IS_OUTPUT_ON(mindspore::kInfo), mindspore::NoExceptionType)
#define MS_LOG_WARNING MSLOG_IF(mindspore::kWarning, IS_OUTPUT_ON(mindspore::kWarning), mindspore::NoExceptionType)
#define MS_LOG_ERROR MSLOG_IF(mindspore::kError, IS_OUTPUT_ON(mindspore::kError), mindspore::NoExceptionType)

#define MS_LOG_EXCEPTION MSLOG_THROW(mindspore::NoExceptionType)
#define MS_EXCEPTION(type) MSLOG_THROW(type)
}  // namespace mindspore

#define MS_EXCEPTION_IF_NULL(ptr)                                  \
  do {                                                             \
    if ((ptr) == nullptr) {                                        \
      MS_LOG(EXCEPTION) << "The pointer[" << #ptr << "] is null."; \
    }                                                              \
  } while (0)

#define MS_EXCEPTION_IF_CHECK_FAIL(condition, error_info)            \
  do {                                                               \
    if (!(condition)) {                                              \
      MS_LOG(EXCEPTION) << "Failure info [" << (error_info) << "]."; \
    }                                                                \
  } while (0)

#define MS_EXCEPTION_IF_ZERO(name, value)                   \
  do {                                                      \
    if ((value) == 0) {                                     \
      MS_LOG(EXCEPTION) << "The " << (name) << " is zero."; \
    }                                                       \
  } while (0)

#define MS_ERROR_IF_NULL(ptr)                                    \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return false;                                              \
    }                                                            \
  } while (0)

#define MS_ERROR_IF_NULL_W_RET_VAL(ptr, val)                     \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return val;                                                \
    }                                                            \
  } while (0)

#define MS_ERROR_IF_NULL_WO_RET_VAL(ptr)                         \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

#define RETURN_IF_FALSE(condition) \
  do {                             \
    if (!(condition)) {            \
      return false;                \
    }                              \
  } while (false)

#define RETURN_IF_FALSE_WITH_LOG(condition, message) \
  do {                                               \
    if (!(condition)) {                              \
      MS_LOG(ERROR) << message;                      \
      return false;                                  \
    }                                                \
  } while (false)

#ifdef DEBUG
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

#endif  // MINDSPORE_CORE_UTILS_LOG_ADAPTER_H_
