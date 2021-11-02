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

bool FileIOUtils::IsFileExist(const std::string &file) {
  std::ifstream fs(file.c_str());
  bool file_exist = fs.good();
  fs.close();
  return file_exist;
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
