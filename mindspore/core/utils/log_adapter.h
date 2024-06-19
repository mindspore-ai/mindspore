/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "mindapi/base/macros.h"
#include "utils/os.h"
#include "utils/overload.h"
#include "./securec.h"
#ifdef USE_GLOG
#define GLOG_NO_ABBREVIATED_SEVERITIES
#define google mindspore_private
#include "glog/logging.h"
#undef google
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
  ShapeError,
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
  SM_PI,                 // PIJIT
  SM_FL,                 // Federated Learning
  SM_DISTRIBUTED,        // Distributed
  SM_LITE,               // LITE
  SM_ARMOUR,             // ARMOUR
  SM_HCCL_ADPT,          // Hccl Adapter
  SM_RUNTIME_FRAMEWORK,  // Runtime framework
  SM_GE,                 // GraphEngine
  SM_API,                // MindAPI
  SM_SYMBOLIC_SHAPE,     // symbolic shape
  SM_GRAPH_KERNEL,       // graph kernel fusion
  SM_MINDIO,             // mindio tpp
  NUM_SUBMODUES,         // number of submodules
};

#ifndef SUBMODULE_ID
#define SUBMODULE_ID mindspore::SubModuleId::SM_ME
#endif

constexpr int COMPONENT_START = 10000;  // vlog start level for component
constexpr int COMPONENT_RANGE = 100;    // number for levels allocated for each component
#define NUM_ALIGN(val, base) (((val) + (base)-1) / (base) * (base))

// VLOG level definition and group
enum VLogLevel : int {
  VL_INVALID = 0,  // invalid vlog level
  VL_FLOW = 1,     // start of end to end flow related log level

  VL_CORE = COMPONENT_START + (SM_CORE - 1) * COMPONENT_RANGE,                            // 0. core
  VL_ANALYZER = COMPONENT_START + (SM_ANALYZER - 1) * COMPONENT_RANGE,                    // 1. static analyzer
  VL_COMMON = COMPONENT_START + (SM_COMMON - 1) * COMPONENT_RANGE,                        // 2. common
  VL_DEBUG = COMPONENT_START + (SM_DEBUG - 1) * COMPONENT_RANGE,                          // 3. debug
  VL_OFFLINE_DEBUG = COMPONENT_START + (SM_OFFLINE_DEBUG - 1) * COMPONENT_RANGE,          // 4. offline debug
  VL_DEVICE = COMPONENT_START + (SM_DEVICE - 1) * COMPONENT_RANGE,                        // 5. device
  VL_GE_ADPT = COMPONENT_START + (SM_GE_ADPT - 1) * COMPONENT_RANGE,                      // 6. ge adapter
  VL_IR = COMPONENT_START + (SM_IR - 1) * COMPONENT_RANGE,                                // 7. IR
  VL_KERNEL = COMPONENT_START + (SM_KERNEL - 1) * COMPONENT_RANGE,                        // 8. kernel
  VL_MD = COMPONENT_START + (SM_MD - 1) * COMPONENT_RANGE,                                // 9. MindData
  VL_ME = COMPONENT_START + (SM_ME - 1) * COMPONENT_RANGE,                                // 10. MindExpression
  VL_EXPRESS = COMPONENT_START + (SM_EXPRESS - 1) * COMPONENT_RANGE,                      // 11. EXPRESS_IR
  VL_OPTIMIZER = COMPONENT_START + (SM_OPTIMIZER - 1) * COMPONENT_RANGE,                  // 12. optimzer
  VL_PARALLEL = COMPONENT_START + (SM_PARALLEL - 1) * COMPONENT_RANGE,                    // 13. parallel
  VL_PARSER = COMPONENT_START + (SM_PARSER - 1) * COMPONENT_RANGE,                        // 14. parser
  VL_PIPELINE = COMPONENT_START + (SM_PIPELINE - 1) * COMPONENT_RANGE,                    // 15. ME pipeline
  VL_PRE_ACT = COMPONENT_START + (SM_PRE_ACT - 1) * COMPONENT_RANGE,                      // 16. pre-activate
  VL_PYNATIVE = COMPONENT_START + (SM_PYNATIVE - 1) * COMPONENT_RANGE,                    // 17. PyNative
  VL_SESSION = COMPONENT_START + (SM_SESSION - 1) * COMPONENT_RANGE,                      // 18. session
  VL_UTILS = COMPONENT_START + (SM_UTILS - 1) * COMPONENT_RANGE,                          // 19. utils
  VL_VM = COMPONENT_START + (SM_VM - 1) * COMPONENT_RANGE,                                // 20. VM
  VL_PROFILER = COMPONENT_START + (SM_PROFILER - 1) * COMPONENT_RANGE,                    // 21. profiler
  VL_PS = COMPONENT_START + (SM_PS - 1) * COMPONENT_RANGE,                                // 22. Parameter Server
  VL_PI = COMPONENT_START + (SM_PI - 1) * COMPONENT_RANGE,                                // 23. PIJIT
  VL_FL = COMPONENT_START + (SM_FL - 1) * COMPONENT_RANGE,                                // 24. Federated Learning
  VL_DISTRIBUTED = COMPONENT_START + (SM_DISTRIBUTED - 1) * COMPONENT_RANGE,              // 25. Distributed
  VL_LITE = COMPONENT_START + (SM_LITE - 1) * COMPONENT_RANGE,                            // 26. LITE
  VL_ARMOUR = COMPONENT_START + (SM_ARMOUR - 1) * COMPONENT_RANGE,                        // 27. ARMOUR
  VL_HCCL_ADPT = COMPONENT_START + (SM_HCCL_ADPT - 1) * COMPONENT_RANGE,                  // 28. Hccl Adapter
  VL_RUNTIME_FRAMEWORK = COMPONENT_START + (SM_RUNTIME_FRAMEWORK - 1) * COMPONENT_RANGE,  // 29. Runtime framework

  VL_GE = COMPONENT_START + (SM_GE - 1) * COMPONENT_RANGE,  // 30. GraphEngine
  VL_ASCEND_KERNEL_SELECT = VL_GE,                          // print ascend kernel select

  VL_API = COMPONENT_START + (SM_API - 1) * COMPONENT_RANGE,                        // 31. MindAPI
  VL_SYMBOLIC_SHAPE = COMPONENT_START + (SM_SYMBOLIC_SHAPE - 1) * COMPONENT_RANGE,  // 32. symbolic shape
  VL_GRAPH_KERNEL = COMPONENT_START + (SM_GRAPH_KERNEL - 1) * COMPONENT_RANGE,      // 33. graph kernel fusion

  VL_USER_CUSTOM = NUM_ALIGN(COMPONENT_START + (NUM_SUBMODUES - 1) * COMPONENT_RANGE,
                             COMPONENT_START),  // start of user defined vlog level
  VL_DISP_VLOG_TAGS = VL_USER_CUSTOM            // print already used vlog tags
};

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

/// \brief The variables for controlling output of vlog.
MS_EXPORT extern int g_ms_vlog_level_from;
MS_EXPORT extern int g_ms_vlog_level_to;

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

class AnfNode;
using AnfNodePtr = std::shared_ptr<AnfNode>;
/// \brief LogWriter defines interface to write log.
class MS_CORE_API LogWriter {
 public:
  using ExceptionHandler = void (*)(ExceptionType, const std::string &);
  using MessageHandler = void (*)(std::ostringstream *oss);
  using TraceProvider = std::function<void(std::ostringstream &, bool)>;
  using GetTraceStrProvider = std::string (*)(const AnfNodePtr &, bool);

  LogWriter(const LocationInfo &location, MsLogLevel log_level, SubModuleId submodule,
            ExceptionType excp_type = NoExceptionType, bool is_internal_exception = false,
            const AnfNodePtr &node = nullptr)
      : location_(location),
        log_level_(log_level),
        vlog_level_(-1),
        submodule_(submodule),
        exception_type_(excp_type),
        is_internal_exception_(is_internal_exception),
        node_(node) {}
  LogWriter(const LocationInfo &location, int vlog_level, SubModuleId submodule)
      : location_(location),
        log_level_(mindspore::kInfo),
        vlog_level_(vlog_level),
        submodule_(submodule),
        exception_type_(NoExceptionType),
        is_internal_exception_(false) {}
  ~LogWriter() = default;

  /// \brief Output log message from the input log stream.
  ///
  /// \param[in] stream The input log stream.
  void operator<(const LogStream &stream) const noexcept;

  /// \brief Output log message from the input log stream and then throw exception.
  ///
  /// \param[in] stream The input log stream.
  void operator^(const LogStream &stream) const NO_RETURN;

  /// \brief Get the function pointer of converting exception types in c++.
  ///
  /// \return A pointer of the function.
  static const ExceptionHandler &GetExceptionHandler();

  /// \brief Set the function pointer of converting exception types in c++.
  ///
  /// \param[in] A function pointer of converting exception types in c++.
  static void SetExceptionHandler(const ExceptionHandler &new_exception_handler);

  /// \brief Get the function pointer of handling message for different device.
  ///
  /// \return A pointer of the function.
  static const MessageHandler &GetMessageHandler();

  /// \brief Set the function pointer of handling message for different device.
  ///
  /// \param[in] A function pointer of handling message for different device.
  static void SetMessageHandler(const MessageHandler &new_message_handler);

  /// \brief Get the function pointer of printing trace stacks.
  ///
  /// \return A pointer of the function.
  static const TraceProvider &GetTraceProvider();

  /// \brief Set the function pointer of printing trace stacks.
  ///
  /// \param[in] A function pointer of printing trace stacks.
  static void SetTraceProvider(const TraceProvider &new_trace_provider);

  /// \brief Set the function pointer of getting node trace string.
  ///
  /// \param[in] A function pointer of getting trace string.
  static void SetGetTraceStrProvider(const LogWriter::GetTraceStrProvider &provider);

 private:
  const std::string GetNodeDebugInfoStr() const;
  void OutputLog(const std::ostringstream &msg) const;
  void RemoveLabelBeforeOutputLog(const std::ostringstream &msg) const;
  static ExceptionHandler &exception_handler();
  static MessageHandler &message_handler();
  static TraceProvider &trace_provider();
  static GetTraceStrProvider &get_trace_str_provider();

  LocationInfo location_;
  MsLogLevel log_level_;
  int vlog_level_ = -1;
  SubModuleId submodule_;
  ExceptionType exception_type_;
  bool is_internal_exception_;
  AnfNodePtr node_;
};

#define MSLOG_IF(level, condition, excp_type, node)                                                                    \
  !(condition) ? void(0)                                                                                               \
               : mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), level, SUBMODULE_ID, \
                                      excp_type, false, node) < mindspore::LogStream()

#define MSLOG_THROW(excp_type, is_internal_exception, node)                                               \
  mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), mindspore::kException, \
                       SUBMODULE_ID, excp_type, is_internal_exception, node) ^                            \
    mindspore::LogStream()

#define MATCH_LEVEL(level)                                                         \
  static_cast<int>(level) >= mindspore::g_ms_submodule_log_levels[SUBMODULE_ID] && \
    static_cast<int>(level) <= static_cast<int>(mindspore::this_thread_max_log_level)

#define IS_OUTPUT_ON(level) (MATCH_LEVEL(level))
#define IS_VLOG_ON(level) (((level) >= g_ms_vlog_level_from) && ((level) <= g_ms_vlog_level_to))

#define __MS_LOG_DEBUG(node) \
  MSLOG_IF(mindspore::kDebug, IS_OUTPUT_ON(mindspore::kDebug), mindspore::NoExceptionType, node)
#define __MS_LOG_INFO(node) MSLOG_IF(mindspore::kInfo, IS_OUTPUT_ON(mindspore::kInfo), mindspore::NoExceptionType, node)
#define __MS_LOG_WARNING(node) \
  MSLOG_IF(mindspore::kWarning, IS_OUTPUT_ON(mindspore::kWarning), mindspore::NoExceptionType, node)
#define __MS_LOG_ERROR(node) \
  MSLOG_IF(mindspore::kError, IS_OUTPUT_ON(mindspore::kError), mindspore::NoExceptionType, node)
#define MS_VLOG(level)                                                                                           \
  !(IS_VLOG_ON(level)) ? void(0)                                                                                 \
                       : mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), level, \
                                              SUBMODULE_ID) < mindspore::LogStream()
#define __MS_LOG_EXCEPTION(node) MSLOG_THROW(mindspore::NoExceptionType, false, node)
#define __MS_LOG_INTERNAL_EXCEPTION(node) MSLOG_THROW(mindspore::NoExceptionType, true, node)

#define MS_LOG(level) __MS_LOG_##level(nullptr)
#define MS_LOG_WITH_NODE(level, node) __MS_LOG_##level(node)

#define MS_LOG_EXCEPTION __MS_LOG_EXCEPTION(nullptr)
#define MS_INTERNAL_EXCEPTION(type) MSLOG_THROW(type, true, nullptr)
#define MS_EXCEPTION(type) MSLOG_THROW(type, false, nullptr)
#define MS_EXCEPTION_WITH_NODE(type, node) MSLOG_THROW(type, false, node)
}  // namespace mindspore

#define MS_EXCEPTION_IF_NULL(ptr)                                           \
  do {                                                                      \
    if ((ptr) == nullptr) {                                                 \
      MS_LOG(INTERNAL_EXCEPTION) << "The pointer[" << #ptr << "] is null."; \
    }                                                                       \
  } while (0)

#define MS_EXCEPTION_IF_CHECK_FAIL(condition, error_info)                     \
  do {                                                                        \
    if (!(condition)) {                                                       \
      MS_LOG(INTERNAL_EXCEPTION) << "Failure info [" << (error_info) << "]."; \
    }                                                                         \
  } while (0)

#define MS_EXCEPTION_IF_ZERO(name, value)                            \
  do {                                                               \
    if ((value) == 0) {                                              \
      MS_LOG(INTERNAL_EXCEPTION) << "The " << (name) << " is zero."; \
    }                                                                \
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
  } while (0)

#define RETURN_IF_FALSE_WITH_LOG(condition, message) \
  do {                                               \
    if (!(condition)) {                              \
      MS_LOG(ERROR) << message;                      \
      return false;                                  \
    }                                                \
  } while (0)

#ifdef DEBUG
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

#endif  // MINDSPORE_CORE_UTILS_LOG_ADAPTER_H_
