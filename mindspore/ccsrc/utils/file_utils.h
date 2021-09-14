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

#ifndef MINDSPORE_CCSRC_UTILS_FILE_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_FILE_UTILS_H_

#include <string>
#include <optional>

namespace mindspore {
class FileUtils {
 public:
  FileUtils() = default;
  ~FileUtils() = default;

  static std::optional<std::string> GetRealPath(const char *path);
  static void SplitDirAndFileName(const std::string &path, std::optional<std::string> *prefix_path,
                                  std::optional<std::string> *file_name);
  static void ConcatDirAndFileName(const std::optional<std::string> *dir, const std::optional<std::string> *file_name,
                                   std::optional<std::string> *path);
  static std::optional<std::string> CreateNotExistDirs(const std::string &path,
                                                       const bool support_relative_path = false);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_FILE_UTILS_H_
