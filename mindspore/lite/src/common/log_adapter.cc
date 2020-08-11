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
#include <sys/time.h>
#include <stdio.h>
#include <cstring>
#ifdef ENABLE_ARM
#include <android/log.h>
#endif

// namespace to support utils module definitionnamespace mindspore constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
namespace mindspore {
constexpr const char *ANDROID_LOG_TAG = "MS_LITE";

int EnvToInt(const char *env) {
    if (env == nullptr)
        return -1;
    if (strcmp(env, "DEBUG") == 0)
        return 0;
    if (strcmp(env, "INFO") == 0)
        return 1;
    if (strcmp(env, "WARNING") == 0)
        return 2;
    if (strcmp(env, "ERROR") == 0)
        return 3;
    return -1;
}

bool IsPrint(int level) {
  static const char *env = std::getenv("MSLOG");
  static int ms_level = EnvToInt(env);
  if (ms_level < 0) {
    ms_level = 2;
  }
  return level >= ms_level;
}

#ifdef ENABLE_ARM
// convert MsLogLevel to corresponding android level
static int GetAndroidLogLevel(MsLogLevel level) {
  switch (level) {
     case DEBUG:
       return ANDROID_LOG_DEBUG;
     case INFO:
       return ANDROID_LOG_INFO;
     case WARNING:
       return ANDROID_LOG_WARN;
     case ERROR:
     default:
       return ANDROID_LOG_ERROR;
  }
}
#endif

const char *EnumStrForMsLogLevel(MsLogLevel level) {
  if (level == DEBUG) {
    return "DEBUG";
  } else if (level == INFO) {
    return "INFO";
  } else if (level == WARNING) {
    return "WARNING";
  } else if (level == ERROR) {
    return "ERROR";
  } else if (level == EXCEPTION) {
    return "EXCEPTION";
  } else {
    return "NO_LEVEL";
  }
}

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
      _TO_STRING(IndexError),
      _TO_STRING(ValueError),
      _TO_STRING(TypeError),
      _TO_STRING(AttributeError),
  };
  // clang-format on
#undef _TO_STRING
  if (type < UnknownError || type > AttributeError) {
    type = UnknownError;
  }
  return std::string(type_names[type]);
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
if (IsPrint(log_level_)) {
// #ifdef USE_ANDROID_LOG
#ifdef ENABLE_ARM
    __android_log_print(GetAndroidLogLevel(log_level_), ANDROID_LOG_TAG, "[%s:%d] %s] %s",  location_.file_,
           location_.line_, location_.func_, msg.str().c_str());
#else
    printf("%s [%s:%d] %s] %s\n:", EnumStrForMsLogLevel(log_level_), location_.file_, location_.line_, location_.func_,
           msg.str().c_str());
#endif
}
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
  if (exception_type_ != NoExceptionType && exception_type_ != IndexError && exception_type_ != TypeError &&
      exception_type_ != ValueError && exception_type_ != AttributeError) {
    oss << ExceptionTypeToString(exception_type_) << " ";
  }
  oss << msg.str();

  if (trace_provider_ != nullptr) {
    trace_provider_(oss);
  }

  if (exception_handler_ != nullptr) {
    exception_handler_(exception_type_, oss.str());
  }
  throw std::runtime_error(oss.str());
}
}  // namespace mindspore

