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

#ifndef MINDSPORE_INFERENCE_LOG_H_
#define MINDSPORE_INFERENCE_LOG_H_

#include <stdarg.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <memory>
#include <iostream>
#include <chrono>
#include <vector>

#ifndef ENABLE_ACL
#include "mindspore/core/utils/log_adapter.h"
#else  // ENABLE_ACL
#include "acl/acl.h"
#endif

namespace mindspore::inference {

class LogStream {
 public:
  LogStream() { sstream_ = std::make_shared<std::stringstream>(); }
  ~LogStream() = default;

  template <typename T>
  LogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  template <typename T>
  LogStream &operator<<(const std::vector<T> &val) noexcept {
    (*sstream_) << "[";
    for (size_t i = 0; i < val.size(); i++) {
      (*this) << val[i];
      if (i + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "]";
    return *this;
  }

  LogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }

  friend class LogWriter;
  friend class Status;

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

#ifndef ENABLE_ACL
#define MSI_LOG(level) MS_LOG(level)

#define MSI_LOG_DEBUG MSI_LOG(DEBUG)
#define MSI_LOG_INFO MSI_LOG(INFO)
#define MSI_LOG_WARNING MSI_LOG(WARNING)
#define MSI_LOG_ERROR MSI_LOG(ERROR)

#define MSI_ASSERT(item) MS_ASSERT(item)

#else  // ENABLE_ACL

class LogWriter {
 public:
  LogWriter(const char *file, int line, const char *func, aclLogLevel log_level)
      : file_(file), line_(line), func_(func), log_level_(log_level) {}
  ~LogWriter() = default;

  void operator<(const LogStream &stream) const noexcept __attribute__((visibility("default"))) {
    std::ostringstream msg;
    msg << stream.sstream_->rdbuf();
    OutputLog(msg);
  }

 private:
  void OutputLog(const std::ostringstream &msg) const { aclAppLog(log_level_, func_, file_, line_, msg.str().c_str()); }

  const char *file_;
  int line_;
  const char *func_;
  aclLogLevel log_level_;
};

#define MSILOG_IF(level) inference::LogWriter(__FILE__, __LINE__, __FUNCTION__, ACL_##level) < inference::LogStream()

#define MSI_LOG(level) MSI_LOG_##level

#define MSI_LOG_DEBUG MSILOG_IF(DEBUG)
#define MSI_LOG_INFO MSILOG_IF(INFO)
#define MSI_LOG_WARNING MSILOG_IF(WARNING)
#define MSI_LOG_ERROR MSILOG_IF(ERROR)

#define MSI_ASSERT(item)

#endif  // ENABLE_ACL

#define MSI_TIME_STAMP_START(name) auto time_start_##name = std::chrono::steady_clock::now();
#define MSI_TIME_STAMP_END(name)                                                                             \
  {                                                                                                          \
    auto time_end_##name = std::chrono::steady_clock::now();                                                 \
    auto time_cost = std::chrono::duration<double, std::milli>(time_end_##name - time_start_##name).count(); \
    MSI_LOG_INFO << #name " Time Cost # " << time_cost << " ms ---------------------";                       \
  }

#define INFER_STATUS(code) inference::Status(code) < inference::LogStream()
#define ERROR_INFER_STATUS(status, type, msg) \
  MSI_LOG_ERROR << msg;                       \
  status = inference::Status(type, msg)

}  // namespace mindspore::inference

#endif  // MINDSPORE_INFERENCE_LOG_H_
