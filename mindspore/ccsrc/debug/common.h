/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEBUG_COMMON_H_
#define MINDSPORE_CCSRC_DEBUG_COMMON_H_

#include <string>
#include <optional>
#include "utils/contract.h"

namespace mindspore {
class Common {
 public:
  Common() = default;
  ~Common() = default;
  static std::optional<std::string> GetRealPath(const std::string &input_path);
  static std::optional<std::string> GetConfigFile(const std::string &env);
  static std::optional<std::string> GetEnvConfigFile();

 private:
  static bool CreateNotExistDirs(const std::string &path);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_COMMON_H_
