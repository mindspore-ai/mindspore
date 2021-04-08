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
static const int MAX_DIRECTORY_LENGTH = 1024;
static const int MAX_FILENAME_LENGTH = 128;
static const int MAX_OS_FILENAME_LENGTH = 255;
class Common {
 public:
  Common() = default;
  ~Common() = default;
  static std::optional<std::string> GetRealPath(const std::string &input_path);
  static std::optional<std::string> GetConfigFile(const std::string &env);
  static std::optional<std::string> GetEnvConfigFile();
  static bool IsStrLengthValid(const std::string &str, const int &length_limit, const std::string &error_message = "");
  static bool IsPathValid(const std::string &path, const int &length_limit, const std::string &error_message = "");
  static bool IsFilenameValid(const std::string &filename, const int &length_limit,
                              const std::string &error_message = "");
  static bool CreateNotExistDirs(const std::string &path);

  static std::string AddId(const std::string &filename, const std::string &suffix);

 private:
  static bool IsEveryFilenameValid(const std::string &path, const int &length_limit, const std::string &error_message);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_COMMON_H_
