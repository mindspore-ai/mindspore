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

#include "utils/log_adapter.h"

#include <unistd.h>
#include "pybind11/pybind11.h"

#include "debug/trace.h"

// namespace to support utils module definition
namespace mindspore {
#ifdef USE_GLOG
static std::string GetTime() {
#define BUFLEN 80
  static char buf[BUFLEN];
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  sprintf_s(buf, BUFLEN, "%d-%d-%d %d:%d:%d", now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday,
            now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, NULL);

  struct tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  // set micro-second
  buf[27] = '\0';
  int idx = 26;
  auto num = cur_time.tv_usec;
  for (int i = 5; i >= 0; i--) {
    buf[idx--] = static_cast<char>(num % 10 + '0');
    num /= 10;
    if (i % 3 == 0) {
      buf[idx--] = '.';
    }
  }
#endif
  return std::string(buf);
}

static std::string GetProcName() {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const char *appname = getprogname();
#elif defined(_GNU_SOURCE)
  const char *appname = program_invocation_name;
#else
  const char *appname = "?";
#endif
  // some times, the appname is an absolute path, its too long
  std::string app_name(appname);
  std::size_t pos = app_name.rfind("/");
  if (pos == std::string::npos) {
    return app_name;
  }
  if (pos + 1 >= app_name.size()) {
    return app_name;
  }
  return app_name.substr(pos + 1);
}

static std::string GetLogLevel(MsLogLevel level) {
#define _TO_STRING(x) #x
  static const char *const level_names[] = {
    _TO_STRING(DEBUG),
    _TO_STRING(INFO),
    _TO_STRING(WARNING),
    _TO_STRING(ERROR),
  };
#undef _TO_STRING
  if (level > ERROR) {
    level = ERROR;
  }
  return std::string(level_names[level]);
}

// convert MsLogLevel to corresponding glog level
static int GetGlogLevel(MsLogLevel level) {
  switch (level) {
    case DEBUG:
    case INFO:
      return google::GLOG_INFO;
    case WARNING:
      return google::GLOG_WARNING;
    case ERROR:
    default:
      return google::GLOG_ERROR;
  }
}
#else
// convert MsLogLevel to corresponding slog level
static int GetSlogLevel(MsLogLevel level) {
  switch (level) {
    case DEBUG:
      return DLOG_DEBUG;
    case INFO:
      return DLOG_INFO;
    case WARNING:
      return DLOG_WARN;
    case ERROR:
    default:
      return DLOG_ERROR;
  }
}
#endif

static std::string ExceptionTypeToString(ExceptionType type) {
#define _TO_STRING(x) #x
  // clang-format off
  static const char *const type_names[] = {
      _TO_STRING(NoExceptionType),
      _TO_STRING(UnknownError),
      _TO_STRING(ArgumentError),
      _TO_STRING(NotSupportError),
      _TO_STRING(NotExistsError),
      _TO_STRING(AlreadyExistsError),
      _TO_STRING(UnavailableError),
      _TO_STRING(DeviceProcessError),
      _TO_STRING(AbortedError),
      _TO_STRING(TimeOutError),
      _TO_STRING(ResourceUnavailable),
      _TO_STRING(NoPermissionError),
      _TO_STRING(ValueError),
      _TO_STRING(TypeError),
  };
  // clang-format on
#undef _TO_STRING
  if (type < UnknownError || type > TypeError) {
    type = UnknownError;
  }
  return std::string(type_names[type]);
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
#ifdef USE_GLOG
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
    << "[" << GetLogLevel(log_level_) << "] ME(" << getpid() << "," << GetProcName() << "):" << GetTime() << " "
    << "[" << location_.file_ << ":" << location_.line_ << "] " << location_.func_ << "] " << msg.str() << std::endl;
#else
  auto str_msg = msg.str();
  Dlog(static_cast<int>(ME), GetSlogLevel(log_level_), "[%s:%d] %s] %s", location_.file_, location_.line_,
       location_.func_, str_msg.c_str());
#endif
}

void LogWriter::operator<(const LogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}

void LogWriter::operator^(const LogStream &stream) const {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);

  std::ostringstream oss;
  oss << location_.file_ << ":" << location_.line_ << " " << location_.func_ << "] ";
  if (exception_type_ != NoExceptionType) {
    oss << ExceptionTypeToString(exception_type_) << " ";
  }
  oss << msg.str();

  trace::TraceGraphInfer();
  trace::GetInferStackInfo(oss);

  if (exception_type_ == ValueError) {
    throw pybind11::value_error(oss.str());
  }
  if (exception_type_ == TypeError) {
    throw pybind11::type_error(oss.str());
  }
  pybind11::pybind11_fail(oss.str());
}

static std::string GetEnv(const std::string &envvar) {
  const char *value = ::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

#ifndef USE_GLOG
// set default log warning to WARNING
int g_mslog_level = WARNING;

static void InitMsLogLevel() {
  int log_level = WARNING;  // set default log level to WARNING
  auto str_level = GetEnv("GLOG_v");
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    ch = ch - '0';  // substract ASCII code of '0', which is 48
    if (ch >= DEBUG && ch <= ERROR) {
      log_level = ch;
    }
  }
  g_mslog_level = log_level;
}

#endif

}  // namespace mindspore

extern "C" {
// shared lib init hook
void mindspore_log_init(void) {
#ifdef USE_GLOG
  // do not use glog predefined log prefix
  FLAGS_log_prefix = false;
  static bool is_glog_initialzed = false;
  if (!is_glog_initialzed) {
    google::InitGoogleLogging("mindspore");
    is_glog_initialzed = true;
  }
  // set default log level to WARNING
  if (mindspore::GetEnv("GLOG_v").empty()) {
    FLAGS_v = mindspore::WARNING;
  }
  // default print log to screen
  if (mindspore::GetEnv("GLOG_logtostderr").empty()) {
    FLAGS_logtostderr = true;
  }
#else
  mindspore::InitMsLogLevel();
#endif
}
}
