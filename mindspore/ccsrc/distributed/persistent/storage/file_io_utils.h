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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_FILE_IO_UTILS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_FILE_IO_UTILS_H_

#include <sys/stat.h>
#include <vector>
#include <string>
#include <utility>
#include "utils/os.h"
#ifdef CreateFile
#undef CreateFile
#endif

namespace mindspore {
namespace distributed {
namespace storage {
class FileIOUtils {
 public:
  // Write memory buffer to the file on overwriting mode, create a new file if the file is not exist.
  static bool Write(const std::string &file_name, const std::vector<std::pair<const void *, size_t>> &inputs);

  // Read file and load the context into memory buffer, return false if the file is not exist.
  static bool Read(const std::string &file_name, const std::vector<std::pair<void *, size_t>> &outputs);

  // Judeg whether a file exists.
  static bool IsFileOrDirExist(const std::string &file);

  // Create file.
  static void CreateFile(const std::string &file_path, mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO);

  // Create directory.
  static void CreateDir(const std::string &dir_path, mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO);

  // Create directory recursively.
  static void CreateDirRecursive(const std::string &dir_path, mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO);
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_FILE_IO_UTILS_H_
