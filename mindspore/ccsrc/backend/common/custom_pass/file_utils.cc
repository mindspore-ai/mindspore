/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/common/custom_pass/file_utils.h"

#include <sys/stat.h>
#include <string>
#include <vector>

#if !defined(_WIN32) && !defined(_WIN64)
#include <dirent.h>
#include <unistd.h>
#else
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif

namespace mindspore {
namespace opt {

bool FileUtils::Exists(const std::string &path) {
#if !defined(_WIN32) && !defined(_WIN64)
  return access(path.c_str(), F_OK) == 0;
#else
  return _access(path.c_str(), 0) == 0;
#endif
}

bool FileUtils::IsDirectory(const std::string &path) {
#if !defined(_WIN32) && !defined(_WIN64)
  struct stat path_stat;
  if (stat(path.c_str(), &path_stat) != 0) {
    return false;
  }
  return S_ISDIR(path_stat.st_mode);
#else
  DWORD attrs = GetFileAttributesA(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return false;
  }
  return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
#endif
}

std::string FileUtils::GetExtension(const std::string &path) {
  size_t dot_pos = path.rfind('.');
  if (dot_pos == std::string::npos || dot_pos == path.length() - 1) {
    return "";
  }
  return path.substr(dot_pos);
}

std::string FileUtils::GetBasename(const std::string &path) {
#if !defined(_WIN32) && !defined(_WIN64)
  size_t pos = path.rfind('/');
#else
  size_t pos = path.rfind('\\');
  if (pos == std::string::npos) {
    pos = path.rfind('/');
  }
#endif
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

std::vector<std::string> FileUtils::ListDirectory(const std::string &directory_path) {
  std::vector<std::string> files;

#if !defined(_WIN32) && !defined(_WIN64)
  DIR *dir = opendir(directory_path.c_str());
  if (!dir) {
    return files;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name != "." && name != "..") {
      std::string full_path = directory_path + "/" + name;
      files.push_back(full_path);
    }
  }
  closedir(dir);
#else
  std::string search_path = directory_path + "\\*";
  WIN32_FIND_DATAA find_data;
  HANDLE find_handle = FindFirstFileA(search_path.c_str(), &find_data);

  if (find_handle != INVALID_HANDLE_VALUE) {
    do {
      std::string name = find_data.cFileName;
      if (name != "." && name != "..") {
        std::string full_path = directory_path + "\\" + name;
        files.push_back(full_path);
      }
    } while (FindNextFileA(find_handle, &find_data));
    FindClose(find_handle);
  }
#endif

  return files;
}

bool FileUtils::IsSupportedPluginFile(const std::string &path) {
  std::string extension = GetExtension(path);
  return extension == ".so" || extension == ".dll" || extension == ".dylib";
}

}  // namespace opt
}  // namespace mindspore
