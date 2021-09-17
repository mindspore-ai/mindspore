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

#include "utils/file_utils.h"

#include <limits.h>
#include <string.h>
#include <string>
#include <optional>
#include <memory>
#include "utils/system/file_system.h"
#include "utils/system/env.h"
#include "utils/utils.h"

namespace mindspore {
std::optional<std::string> FileUtils::GetRealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "Input path is nullptr";
    return std::nullopt;
  }

  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (strlen(path) >= PATH_MAX || _fullpath(real_path, path, PATH_MAX) == nullptr) {
    MS_LOG(ERROR) << "Get _fullpath failed";
    return std::nullopt;
  }
#else
  if (strlen(path) >= PATH_MAX || realpath(path, real_path) == nullptr) {
    MS_LOG(ERROR) << "Get realpath failed, path[" << path << "]";
    return std::nullopt;
  }
#endif
  return std::string(real_path);
}

void FileUtils::SplitDirAndFileName(const std::string &path, std::optional<std::string> *prefix_path,
                                    std::optional<std::string> *file_name) {
  auto path_split_pos = path.find_last_of('/');
  if (path_split_pos == std::string::npos) {
    path_split_pos = path.find_last_of('\\');
  }

  MS_EXCEPTION_IF_NULL(prefix_path);
  MS_EXCEPTION_IF_NULL(file_name);

  if (path_split_pos != std::string::npos) {
    *prefix_path = path.substr(0, path_split_pos);
    *file_name = path.substr(path_split_pos + 1);
  } else {
    *prefix_path = std::nullopt;
    *file_name = path;
  }
}

void FileUtils::ConcatDirAndFileName(const std::optional<std::string> *dir, const std::optional<std::string> *file_name,
                                     std::optional<std::string> *path) {
  MS_EXCEPTION_IF_NULL(dir);
  MS_EXCEPTION_IF_NULL(file_name);
  MS_EXCEPTION_IF_NULL(path);
#if defined(_WIN32) || defined(_WIN64)
  *path = dir->value() + "\\" + file_name->value();
#else
  *path = dir->value() + "/" + file_name->value();
#endif
}

std::optional<std::string> FileUtils::CreateNotExistDirs(const std::string &path, const bool support_relative_path) {
  if (path.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "The length of the path is greater than or equal to:" << PATH_MAX;
    return std::nullopt;
  }
  if (!support_relative_path) {
    auto dot_pos = path.find("..");
    if (dot_pos != std::string::npos) {
      MS_LOG(ERROR) << "Do not support relative path";
      return std::nullopt;
    }
  }

  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  char temp_path[PATH_MAX] = {0};
  for (uint32_t i = 0; i < path.length(); i++) {
    temp_path[i] = path[i];
    if (temp_path[i] == '\\' || temp_path[i] == '/') {
      if (i != 0) {
        char tmp_char = temp_path[i];
        temp_path[i] = '\0';
        std::string path_handle(temp_path);
        if (!fs->FileExist(path_handle)) {
          if (!fs->CreateDir(path_handle)) {
            MS_LOG(ERROR) << "Create " << path_handle << " dir error";
            return std::nullopt;
          }
        }
        temp_path[i] = tmp_char;
      }
    }
  }

  if (!fs->FileExist(path)) {
    if (!fs->CreateDir(path)) {
      MS_LOG(ERROR) << "Create " << path << " dir error";
      return std::nullopt;
    }
  }
  return GetRealPath(path.c_str());
}
}  // namespace mindspore
