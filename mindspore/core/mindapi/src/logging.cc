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

#define MIND_LOG_NO_MS_LOG
#include "mindapi/base/logging.h"
#include "utils/log_adapter.h"

namespace mindspore::api {
static MsLogLevel ToMsLogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG:
      return MsLogLevel::DEBUG;
    case LogLevel::INFO:
      return MsLogLevel::INFO;
    case LogLevel::WARNING:
      return MsLogLevel::WARNING;
    case LogLevel::ERROR:
      return MsLogLevel::ERROR;
    case LogLevel::EXCEPTION:
      return MsLogLevel::EXCEPTION;
    default:
      return MsLogLevel::EXCEPTION;
  }
}

class LogWriterImpl {
 public:
  LogWriterImpl(LogLevel level, const char *file, int line, const char *func)
      : writer_(LocationInfo(file, line, func), ToMsLogLevel(level), SubModuleId::SM_API) {}

  ~LogWriterImpl() = default;

  void Write(const LogStream &stream) const noexcept {
    mindspore::LogStream log_stream;
    log_stream << stream.stream_.rdbuf();
    writer_ < log_stream;
  }

  void WriteAndThrow(const LogStream &stream) const __attribute__((noreturn)) {
    mindspore::LogStream log_stream;
    log_stream << stream.stream_.rdbuf();
    writer_ ^ log_stream;
  }

 private:
  mindspore::LogWriter writer_;
};

LogWriter::LogWriter(LogLevel level, const char *file, int line, const char *func)
    : impl_(std::make_unique<LogWriterImpl>(level, file, line, func)) {}

LogWriter::~LogWriter() = default;

void LogWriter::operator<(const LogStream &stream) const noexcept { impl_->Write(stream); }

void LogWriter::operator^(const LogStream &stream) const { impl_->WriteAndThrow(stream); }

bool LogWriter::IsEnabled(LogLevel level) {
  auto log_level = ToMsLogLevel(level);
  return IS_OUTPUT_ON(log_level);
}
}  // namespace mindspore::api
