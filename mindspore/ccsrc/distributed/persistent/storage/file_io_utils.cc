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

#include "distributed/persistent/storage/file_io_utils.h"
#include <fstream>

#ifdef _MSC_VER
#include <direct.h>  // for _mkdir on windows
#endif
#include "mindspore/core/utils/file_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/os.h"

namespace mindspore {
namespace distributed {
namespace storage {
namespace {
bool CheckFStreamLength(const std::string &file_name, std::fstream &fs, size_t size) {
  size_t cur_pos = LongToSize(fs.tellp());
  (void)fs.seekp(0, std::ios::end);
  if (!fs.good() || fs.fail() || fs.bad()) {
    MS_LOG(ERROR) << "Failed to seedp file pos, file name: " << file_name;
    return false;
  }
  size_t end_pos = LongToSize(fs.tellp());
  if (end_pos - cur_pos < size) {
    MS_LOG(ERROR) << "The content length of file:" << file_name << " is less than expected size: " << size;
    return false;
  }
  (void)fs.seekp(cur_pos);
  if (!fs.good() || fs.fail() || fs.bad()) {
    MS_LOG(ERROR) << "Failed to seedp file pos, file name: " << file_name;
    return false;
  }

  return true;
}
}  // namespace

bool FileIOUtils::Write(const std::string &file_name, const std::vector<std::pair<const void *, size_t>> &inputs) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "The file name is empty";
    return false;
  }

  std::fstream fs;
  fs.open(file_name, std::ios::out | std::ios::binary);
  if (!fs.is_open() || !fs.good()) {
    MS_LOG(ERROR) << "Open file failed, file name: " << file_name;
    return false;
  }

  for (const auto &item : inputs) {
    const void *data = item.first;
    MS_ERROR_IF_NULL(data);
    size_t size = item.second;
    (void)fs.write(reinterpret_cast<const char *>(data), SizeToLong(size));
    if (!fs.good() || fs.fail() || fs.bad()) {
      fs.close();
      MS_LOG(ERROR) << "Insert data to fstream failed.";
      return false;
    }
    (void)fs.flush();
    if (!fs.good() || fs.fail() || fs.bad()) {
      fs.close();
      MS_LOG(ERROR) << "Insert data to fstream failed.";
      return false;
    }
  }

  fs.close();
  return true;
}

bool FileIOUtils::Read(const std::string &file_name, const std::vector<std::pair<void *, size_t>> &outputs) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "The file name is empty";
    return false;
  }

  std::fstream fs;
  fs.open(file_name, std::ios::in | std::ios::binary);
  if (!fs.is_open() || !fs.good()) {
    MS_LOG(ERROR) << "Open file failed, file name: " << file_name;
    return false;
  }

  for (const auto &item : outputs) {
    void *data = item.first;
    MS_ERROR_IF_NULL(data);
    size_t size = item.second;

    if (!CheckFStreamLength(file_name, fs, size)) {
      return false;
    }

    (void)fs.read(reinterpret_cast<char *>(data), SizeToLong(size));
    if (!fs.good() || fs.fail() || fs.bad()) {
      fs.close();
      MS_LOG(ERROR) << "Read data from fstream failed.";
      return false;
    }
  }
  fs.close();
  return true;
}

bool FileIOUtils::IsFileOrDirExist(const std::string &path) {
  if (path.empty()) {
    MS_LOG(EXCEPTION) << "The path name is empty";
  }

  return access(path.c_str(), F_OK) == 0;
}

void FileIOUtils::CreateFile(const std::string &file_path, mode_t mode) {
  (void)mode;
  if (IsFileOrDirExist(file_path)) {
    return;
  }

  std::ofstream output_file(file_path);
  output_file.close();
#ifndef _MSC_VER
  ChangeFileMode(file_path, mode);
#endif
}

void FileIOUtils::CreateDir(const std::string &dir_path, mode_t mode) {
  if (IsFileOrDirExist(dir_path)) {
    return;
  }

#if defined(_WIN32) || defined(_WIN64)
#ifndef _MSC_VER
  int ret = mkdir(dir_path.c_str());
#else
  int ret = _mkdir(dir_path.c_str());
#endif
#else
  int ret = mkdir(dir_path.c_str(), mode);
  if (ret == 0) {
    ChangeFileMode(dir_path, mode);
  }
#endif
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Failed to create directory " << dir_path << ". Errno = " << errno;
  }
}

void FileIOUtils::CreateDirRecursive(const std::string &dir_path, mode_t mode) {
  if (dir_path.empty()) {
    MS_LOG(EXCEPTION) << "The directory path need to be create is empty";
  }
  size_t dir_path_len = dir_path.length();
  if (dir_path_len > PATH_MAX) {
    MS_LOG(EXCEPTION) << "Directory path is too long to exceed max length limit: " << PATH_MAX
                      << ", the path: " << dir_path;
  }

  char tmp_dir_path[PATH_MAX] = {0};
  for (size_t i = 0; i < dir_path_len; ++i) {
    tmp_dir_path[i] = dir_path[i];
    if (tmp_dir_path[i] == '/' || dir_path == tmp_dir_path) {
      if (access(tmp_dir_path, F_OK) == 0) {
        continue;
      }

#if defined(_WIN32) || defined(_WIN64)
#ifndef _MSC_VER
      int32_t ret = mkdir(tmp_dir_path);
#else
      int32_t ret = _mkdir(tmp_dir_path);
#endif
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "Failed to create directory recursion: " << dir_path << ". Errno = " << errno;
      }
#else
      int32_t ret = mkdir(tmp_dir_path, mode);
      if (ret == 0) {
        ChangeFileMode(tmp_dir_path, mode);
      } else if (errno != EEXIST) {
        MS_LOG(EXCEPTION) << "Failed to create directory recursion: " << dir_path << ". Errno = " << errno;
      }
#endif
    }
  }
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
