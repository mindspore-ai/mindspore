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

#include "common/mslog.h"
#include <iostream>
#include <cstdlib>
#include <climits>
#include <string>
#include "include/errorcode.h"

namespace mindspore {
namespace predict {
std::string GetEnv(const std::string &envvar) {
  const char *value = std::getenv(envvar.c_str());
  if (value == nullptr) {
    return std::string();
  }
  return std::string(value);
}

bool IsPrint(int level) {
  auto envString = GetEnv("MSLOG");
  static int env = static_cast<int>(std::strtol(!envString.empty() ? envString.c_str() : "3", nullptr, 0));
  if (env == INT_MIN || env == INT_MAX) {
    env = WARN;
    // enable the SP for binscope checking
    std::string errorStr = "env exceeded the value that type int is able to represent";
    MS_LOGE("%s", errorStr.c_str());
  }

  return level >= env;
}
}  // namespace predict
}  // namespace mindspore
