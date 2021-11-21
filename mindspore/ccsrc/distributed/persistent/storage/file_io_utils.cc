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

#include <dirent.h>
#include <unistd.h>
#include <fstream>

#include "utils/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
namespace storage {
namespace {
bool CheckFStreamLength(const std::string &file_name, std::fstream &fs, size_t size) {
  size_t cur_pos = fs.tellp();
  fs.seekp(0, std::ios::end);
  if (!fs.good() || fs.fail() || fs.bad()) {
    MS_LOG(ERROR) << "Failed to seedp file pos, file name: " << file_name;
    return false;
  }
  size_t end_pos = fs.tellp();
  if (end_pos - cur_pos < size) {
    MS_LOG(ERROR) << "The content length of file:" << file_name << " is less than expected size: " << size;
    return false;
  }
  fs.seekp(cur_pos);
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
    fs.write(reinterpret_cast<const char *>(data), size);
    if (!fs.good() || fs.fail() || fs.bad()) {
      fs.close();
      MS_LOG(ERROR) << "Insert data to fstream failed.";
      return false;
    }
    fs.flush();
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

    fs.read(reinterpret_cast<char *>(data), size);
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
  if (IsFileOrDirExist(file_path)) {
    return;
  }

  std::ofstream output_file(file_path);
  output_file.close();
  ChangeFileMode(file_path, mode);
}

void FileIOUtils::CreateDir(const std::string &dir_path, mode_t mode) {
  if (IsFileOrDirExist(dir_path)) {
    return;
  }

#if defined(_WIN32) || defined(_WIN64)
  int ret = mkdir(dir_path.c_str());
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
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
