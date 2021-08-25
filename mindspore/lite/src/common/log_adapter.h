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

#ifndef MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_
#define MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_
namespace mindspore {
const char *const unsupport_string_tensor_log =
  "This mindspore-lite library does not support string tensors. Set environment variable MSLITE_STRING_KERNEL to on to "
  "recompile it.";
const char *const unsupport_controlflow_tensorlist_log =
  "This mindspore-lite library does not support controlflow and tensorlist op. Set environment variable "
  "MSLITE_CONTROLFLOW_TENSORLIST to on to recompile it.";
const char *const unsupport_auto_parallel_log =
  "The mindspore-lite library does not support auto parallel. Set environment variable MSLITE_AUTO_PARALLEL to on to "
  "recompile it.";
const char *const unsupport_weight_decode_log =
  "The mindspore-lite library does not support weight decode. Set environment variable MSLITE_WEIGHT_DECODE to on to "
  "recompile it.";
const char *const unsupport_custom_kernel_register_log =
  "The mindspore-lite library does not support custom kernel register. Set environment variable "
  "MSLITE_CUSTOM_KERNEL_REGISTRY to on to "
  "recompile it.";
const char *const unsupport_delegate_log =
  "The mindspore-lite library does not support delegate. Set environment variable "
  "MSLITE_DELEGATE_USE to on to "
  "recompile it.";
const char *const unsupport_fp16_log =
  "The mindspore-lite library does not support fp16. Set environment variable "
  "MSLITE_ENABLE_FP16 to on to "
  "recompile it.";
}  // namespace mindspore
#ifdef USE_GLOG
#include "utils/log_adapter.h"
#else
#include <cstdarg>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include "utils/overload.h"
// NOTICE: when relative path of 'log_adapter.h' changed, macro 'LOG_HEAR_FILE_REL_PATH' must be changed
#define LOG_HEAR_FILE_REL_PATH "mindspore/lite/src/common/log_adapter.h"

// Get start index of file relative path in __FILE__
static constexpr size_t GetRealPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(LOG_HEAR_FILE_REL_PATH) ? sizeof(__FILE__) - sizeof(LOG_HEAR_FILE_REL_PATH) : 0;
}

namespace mindspore {
#define FILE_NAME                                                                               \
  (sizeof(__FILE__) > GetRealPathPos() ? static_cast<const char *>(__FILE__) + GetRealPathPos() \
                                       : static_cast<const char *>(__FILE__))

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

enum MsLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR };

const char *EnumStrForMsLogLevel(MsLogLevel level);

class LogWriter {
 public:
  LogWriter(const LocationInfo &location, MsLogLevel log_level) : location_(location), log_level_(log_level) {}

  ~LogWriter() = default;

#ifdef _WIN32
  __declspec(dllexport) void operator<(const LogStream &stream) const noexcept;
#else
  __attribute__((visibility("default"))) void operator<(const LogStream &stream) const noexcept;
#endif

 private:
  void OutputLog(const std::ostringstream &msg) const;

  LocationInfo location_;
  MsLogLevel log_level_;
};

#define MSLOG_IF(level) \
  mindspore::LogWriter(mindspore::LocationInfo(FILE_NAME, __LINE__, __FUNCTION__), level) < mindspore::LogStream()

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_DEBUG MSLOG_IF(mindspore::DEBUG)
#define MS_LOG_INFO MSLOG_IF(mindspore::INFO)
#define MS_LOG_WARNING MSLOG_IF(mindspore::WARNING)
#define MS_LOG_ERROR MSLOG_IF(mindspore::ERROR)

}  // namespace mindspore

#ifdef Debug
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif
#endif  // USE_GLOG
#endif  // MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_
