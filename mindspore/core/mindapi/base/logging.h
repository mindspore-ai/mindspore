/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_LOGGING_H_
#define MINDSPORE_CORE_MINDAPI_BASE_LOGGING_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>
#include "mindapi/base/macros.h"

namespace mindspore::api {
enum class LogLevel : uint8_t { kDebug = 0, kInfo, kWarning, kError, kException };

class LogWriterImpl;

/// \brief LogStream represents a stream to write log messages.
/// This class is not expected for directly use, use MS_LOG instead.
class LogStream {
 public:
  /// \brief Write log message to this LogStream.
  ///
  /// \param[in] value The object to be written.
  template <typename T>
  LogStream &operator<<(T &&value) noexcept {
    stream_ << (std::forward<T>(value));
    return *this;
  }

 private:
  friend class LogWriterImpl;
  std::stringstream stream_;
};

/// \brief LogWriter defines interface for log message output.
/// This class is not expected for directly use, use MS_LOG instead.
class MIND_API LogWriter {
 public:
  /// \brief Create a LogWriter with the given log level, file name, line number and function name.
  ///
  /// \param[in] level The log level.
  /// \param[in] file The file name.
  /// \param[in] line The line number.
  /// \param[in] func The function name.
  LogWriter(LogLevel level, const char *file, int line, const char *func);

  /// \brief Destructor for LogWriter.
  ~LogWriter();

  /// \brief Output log message from the input log stream.
  ///
  /// \param[in] stream The input log stream.
  void operator<(const LogStream &stream) const noexcept;

  /// \brief Output log message from the input log stream and then throw exception.
  ///
  /// \param[in] stream The input log stream.
  void operator^(const LogStream &stream) const NO_RETURN;

  /// \brief Check whether the given log level is enabled or not.
  ///
  /// \return True if the log level is enabled, false otherwise.
  static bool IsEnabled(LogLevel level);

 private:
  std::unique_ptr<LogWriterImpl> impl_;
};

#define MIND_LOG_STREAM mindspore::api::LogStream()
#define MIND_LOG_WRITER mindspore::api::LogWriter
#define MIND_LOG_LEVEL(L) mindspore::api::LogLevel::L

#define MIND_LOG_THROW(L) MIND_LOG_WRITER(MIND_LOG_LEVEL(L), __FILE__, __LINE__, __FUNCTION__) ^ MIND_LOG_STREAM
#define MIND_LOG_WRITE(L) MIND_LOG_WRITER(MIND_LOG_LEVEL(L), __FILE__, __LINE__, __FUNCTION__) < MIND_LOG_STREAM
#define MIND_LOG_IF(L) \
  if (MIND_LOG_WRITER::IsEnabled(MIND_LOG_LEVEL(L))) MIND_LOG_WRITE(L)

#define MIND_LOG_DEBUG MIND_LOG_IF(kDebug)
#define MIND_LOG_INFO MIND_LOG_IF(kInfo)
#define MIND_LOG_WARNING MIND_LOG_IF(kWarning)
#define MIND_LOG_ERROR MIND_LOG_IF(kError)
#define MIND_LOG_EXCEPTION MIND_LOG_THROW(kException)
#define MIND_LOG(level) MIND_LOG_##level

#if !defined(MIND_LOG_NO_MS_LOG) && !defined(MS_LOG)
#define MS_LOG(level) MIND_LOG_##level
#endif
}  // namespace mindspore::api

#endif  // MINDSPORE_CORE_MINDAPI_BASE_LOGGING_H_
