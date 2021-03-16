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

#include "src/common/log_adapter.h"
#include <cstring>
#include <cstdio>

#ifdef ENABLE_ARM
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif
#endif

// namespace to support utils module definition namespace mindspore constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
namespace mindspore {
#if defined(__ANDROID__) || defined(ANDROID)
constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
#endif

int StrToInt(const char *env) {
  if (env == nullptr) return 2;
  if (strcmp(env, "0") == 0) return 0;
  if (strcmp(env, "1") == 0) return 1;
  if (strcmp(env, "2") == 0) return 2;
  if (strcmp(env, "3") == 0) return 3;
  return 2;
}

bool IsPrint(int level) {
  static const char *env = std::getenv("GLOG_v");
  static int ms_level = StrToInt(env);
  if (level < 0) {
    level = 2;
  }
  return level >= ms_level;
}

#ifdef ENABLE_ARM
#if defined(__ANDROID__) || defined(ANDROID)
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
  } else {
    return "NO_LEVEL";
  }
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
  if (IsPrint(log_level_)) {
#ifdef ENABLE_ARM
#if defined(__ANDROID__) || defined(ANDROID)
    __android_log_print(GetAndroidLogLevel(log_level_), ANDROID_LOG_TAG, "[%s:%d] %s] %s", location_.file_,
                        location_.line_, location_.func_, msg.str().c_str());
#endif
#else
    printf("%s [%s:%d] %s] %s\n", EnumStrForMsLogLevel(log_level_), location_.file_, location_.line_, location_.func_,
           msg.str().c_str());
#endif
  }
}

void LogWriter::operator<(const LogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
}  // namespace mindspore
