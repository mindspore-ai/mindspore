/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_FILE_UTILS_H_
#define MINDSPORE_CORE_UTILS_FILE_UTILS_H_

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <optional>
#include "mindspore/core/utils/ms_utils.h"
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"
#include "utils/os.h"

namespace mindspore {
class MS_CORE_API FileUtils {
 public:
  FileUtils() = default;
  ~FileUtils() = default;

  static std::fstream *OpenFile(const std::string &file_path, std::ios_base::openmode open_mode);
  static bool ParserPathAndModelName(const std::string &output_path, std::string *save_path, std::string *model_name);

  static std::optional<std::string> GetRealPath(const char *path);
  static void SplitDirAndFileName(const std::string &path, std::optional<std::string> *prefix_path,
                                  std::optional<std::string> *file_name);
  static void ConcatDirAndFileName(const std::optional<std::string> *dir, const std::optional<std::string> *file_name,
                                   std::optional<std::string> *path);
  static std::optional<std::string> CreateNotExistDirs(const std::string &path,
                                                       const bool support_relative_path = false);
#if defined(_WIN32) || defined(_WIN64)
  static std::string GB2312ToUTF_8(const char *gb2312);
  static std::string UTF_8ToGB2312(const char *text);
#endif
};

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  if (access(file_name.c_str(), F_OK) == -1) {
    return;
  }
  try {
    if (chmod(common::SafeCStr(file_name), mode) != 0) {
      MS_LOG(WARNING) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
    }
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "File `" << file_name << "` change mode failed! May be not exist.";
  }
}
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_FILE_UTILS_H_
