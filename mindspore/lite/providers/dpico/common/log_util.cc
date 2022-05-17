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
#include "common/log_util.h"
#include <cstring>
#include <cstdio>
namespace mindspore {
int StrToInt(const char *env) {
  if (env == nullptr) {
    return static_cast<int>(mindspore::DpicoLogLevel::WARNING);
  }
  if (strcmp(env, "0") == 0) {
    return static_cast<int>(mindspore::DpicoLogLevel::DEBUG);
  }
  if (strcmp(env, "1") == 0) {
    return static_cast<int>(mindspore::DpicoLogLevel::INFO);
  }
  if (strcmp(env, "2") == 0) {
    return static_cast<int>(mindspore::DpicoLogLevel::WARNING);
  }
  if (strcmp(env, "3") == 0) {
    return static_cast<int>(mindspore::DpicoLogLevel::ERROR);
  }
  return static_cast<int>(mindspore::DpicoLogLevel::WARNING);
}

bool IsPrint(int level) {
  static const char *const env = std::getenv("GLOG_v");
  static const int ms_level = StrToInt(env);
  if (level < 0) {
    level = static_cast<int>(mindspore::DpicoLogLevel::WARNING);
  }
  return level >= ms_level;
}

const char *EnumStrForMsLogLevel(DpicoLogLevel level) {
  if (level == DpicoLogLevel::DEBUG) {
    return "DEBUG";
  } else if (level == DpicoLogLevel::INFO) {
    return "INFO";
  } else if (level == DpicoLogLevel::WARNING) {
    return "WARNING";
  } else if (level == DpicoLogLevel::ERROR) {
    return "ERROR";
  } else {
    return "NO_LEVEL";
  }
}

void DpicoLogWriter::OutputLog(const std::ostringstream &msg) const {
  if (IsPrint(static_cast<int>(log_level_))) {
    printf("%s [%s:%d] %s] %s\n", EnumStrForMsLogLevel(log_level_), location_.file_, location_.line_, location_.func_,
           msg.str().c_str());
  }
}

void DpicoLogWriter::operator<(const DpicoLogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
}  // namespace mindspore
