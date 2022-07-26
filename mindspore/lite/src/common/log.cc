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
#include "src/common/log.h"
#include <cstring>
#include <cstdio>

#if defined(ENABLE_ARM) && (defined(__ANDROID__) || defined(ANDROID))
#include <android/log.h>
#endif

// namespace to support utils module definition namespace mindspore constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
namespace mindspore {
#if defined(ENABLE_ARM) && (defined(__ANDROID__) || defined(ANDROID))
constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
#endif

int StrToInt(const char *env) {
  if (env == nullptr) {
    return 2;
  }
  if (strcmp(env, "0") == 0) {
    return 0;
  }
  if (strcmp(env, "1") == 0) {
    return 1;
  }
  if (strcmp(env, "2") == 0) {
    return 2;
  }
  if (strcmp(env, "3") == 0) {
    return 3;
  }
  return 2;
}

bool IsPrint(int level) {
  static const char *const env = std::getenv("GLOG_v");
  static const int ms_level = StrToInt(env);
  if (level < 0) {
    level = 2;
  }
  return level >= ms_level;
}

#if defined(ENABLE_ARM) && (defined(__ANDROID__) || defined(ANDROID))
static int GetAndroidLogLevel(LiteLogLevel level) {
  switch (level) {
    case LiteLogLevel::DEBUG:
      return ANDROID_LOG_DEBUG;
    case LiteLogLevel::INFO:
      return ANDROID_LOG_INFO;
    case LiteLogLevel::WARNING:
      return ANDROID_LOG_WARN;
    case LiteLogLevel::ERROR:
    default:
      return ANDROID_LOG_ERROR;
  }
}
#endif

const char *EnumStrForMsLogLevel(LiteLogLevel level) {
  if (level == LiteLogLevel::DEBUG) {
    return "DEBUG";
  } else if (level == LiteLogLevel::INFO) {
    return "INFO";
  } else if (level == LiteLogLevel::WARNING) {
    return "WARNING";
  } else if (level == LiteLogLevel::ERROR) {
    return "ERROR";
  } else {
    return "NO_LEVEL";
  }
}

void LiteLogWriter::OutputLog(const std::ostringstream &msg) const {
  if (IsPrint(static_cast<int>(log_level_))) {
#if defined(ENABLE_ARM) && (defined(__ANDROID__) || defined(ANDROID))
    __android_log_print(GetAndroidLogLevel(log_level_), ANDROID_LOG_TAG, "[%s:%d] %s] %s", location_.file_,
                        location_.line_, location_.func_, msg.str().c_str());
#else
    printf("%s [%s:%d] %s] %s\n", EnumStrForMsLogLevel(log_level_), location_.file_, location_.line_, location_.func_,
           msg.str().c_str());
#endif
  }
}

void LiteLogWriter::operator<(const LiteLogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
}  // namespace mindspore
