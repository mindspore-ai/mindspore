/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/common/shard_utils.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "./securec.h"

#ifdef _MSC_VER
#define stat _stat64  //  for file size exceeds (1<<31)-1 bytes
#endif

namespace mindspore {
namespace mindrecord {
// split a string using a character
std::vector<std::string> StringSplit(const std::string &field, char separator) {
  std::vector<std::string> res;
  uint64_t s_pos = 0;
  while (s_pos < field.length()) {
    size_t e_pos = field.find_first_of(separator, s_pos);
    if (e_pos != std::string::npos) {
      res.push_back(field.substr(s_pos, e_pos - s_pos));
    } else {
      res.push_back(field.substr(s_pos, field.length() - s_pos));
      break;
    }
    s_pos = e_pos + 1;
  }
  return res;
}

bool ValidateFieldName(const std::string &str) {
  auto it = str.cbegin();
  if (it == str.cend()) {
    return false;
  }
  for (; it != str.cend(); ++it) {
    if (*it == '_' || ((*it >= '0') && (*it <= '9')) || ((*it >= 'A') && (*it <= 'Z')) ||
        ((*it >= 'a') && (*it <= 'z'))) {
      continue;
    }
    return false;
  }
  return true;
}

Status GetFileName(const std::string &path, std::shared_ptr<std::string> *fn_ptr) {
  RETURN_UNEXPECTED_IF_NULL_MR(fn_ptr);

  std::optional<std::string> prefix_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(path, &prefix_path, &file_name);
  if (!file_name.has_value()) {
    RETURN_STATUS_UNEXPECTED_MR(
      "Invalid file, failed to get the filename of mindrecord file. Please check file path: " + path);
  }
  *fn_ptr = std::make_shared<std::string>(file_name.value());

  return Status::OK();
}

Status GetParentDir(const std::string &path, std::shared_ptr<std::string> *pd_ptr) {
  RETURN_UNEXPECTED_IF_NULL_MR(pd_ptr);

  std::optional<std::string> prefix_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(path, &prefix_path, &file_name);
  if (!prefix_path.has_value()) {
    prefix_path = ".";
  }

  auto realpath = FileUtils::GetRealPath(prefix_path.value().c_str());
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    realpath.has_value(), "Invalid file, failed to get the parent dir of mindrecord file. Please check file: " + path);

  *pd_ptr = std::make_shared<std::string>(realpath.value() + kPathSeparator);
  return Status::OK();
}

bool CheckIsValidUtf8(const std::string &str) {
  int n = 0;
  int ix = str.length();
  for (int i = 0; i < ix; ++i) {
    uint8_t c = static_cast<unsigned char>(str[i]);
    if (c <= 0x7f) {
      n = 0;
    } else if ((c & 0xE0) == 0xC0) {
      n = 1;
    } else if (c == 0xed && i < (ix - 1) && (static_cast<unsigned char>(str[i + 1]) & 0xa0) == 0xa0) {
      return false;
    } else if ((c & 0xF0) == 0xE0) {
      n = 2;
    } else if ((c & 0xF8) == 0xF0) {
      n = 3;
    } else {
      return false;
    }
    for (int j = 0; j < n && i < ix; ++j) {
      if ((++i == ix) || ((static_cast<unsigned char>(str[i]) & 0xC0) != 0x80)) {
        return false;
      }
    }
  }
  return true;
}

Status CheckFile(const std::string &path) {
  struct stat s;
#if defined(_WIN32) || defined(_WIN64)
  if (stat(FileUtils::UTF_8ToGB2312(path.data()).data(), &s) == 0) {
#else
  if (stat(common::SafeCStr(path), &s) == 0) {
#endif
    if (S_ISDIR(s.st_mode)) {
      RETURN_STATUS_UNEXPECTED_MR("Invalid file, " + path + " is not a mindrecord file, but got directory.");
    }
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED_MR(
    "Invalid file, mindrecord file: " + path +
    " can not be found. Please check whether the mindrecord file exists and do not rename the mindrecord file.");
}

Status GetDiskSize(const std::string &str_dir, const DiskSizeType &disk_type, std::shared_ptr<uint64_t> *size_ptr) {
  RETURN_UNEXPECTED_IF_NULL_MR(size_ptr);
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
  *size_ptr = std::make_shared<uint64_t>(100);
  return Status::OK();
#else
  uint64_t ll_count = 0;
  struct statfs64 disk_info;
  if (statfs64(common::SafeCStr(str_dir), &disk_info) == -1) {
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to get free disk size.");
  }

  switch (disk_type) {
    case kTotalSize:
      ll_count = disk_info.f_bsize * disk_info.f_blocks;
      ll_count = ll_count >> 20;
      break;
    case kFreeSize:
      ll_count = disk_info.f_bsize * disk_info.f_bavail;
      ll_count = ll_count >> 20;
      break;
    default:
      ll_count = 0;
      break;
  }
  *size_ptr = std::make_shared<uint64_t>(ll_count);
  return Status::OK();
#endif
}

uint32_t GetMaxThreadNum() {
  // define the number of thread
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) {
    thread_num = kMaxConsumerCount;
  }
  return thread_num;
}

Status GetDatasetFiles(const std::string &path, const json &addresses, std::shared_ptr<std::vector<std::string>> *ds) {
  RETURN_UNEXPECTED_IF_NULL_MR(ds);
  std::shared_ptr<std::string> parent_dir;
  RETURN_IF_NOT_OK_MR(GetParentDir(path, &parent_dir));
  for (const auto &p : addresses) {
    std::string abs_path = *parent_dir + std::string(p);
    (*ds)->emplace_back(abs_path);
  }
  return Status::OK();
}

std::mt19937 GetRandomDevice() {
#if defined(_WIN32) || defined(_WIN64)
  unsigned int number;
  rand_s(&number);
  std::mt19937 random_device{static_cast<uint32_t>(number)};
#else
  int i = 0;
  while (i < 5) {
    try {
      std::mt19937 random_device{std::random_device("/dev/urandom")()};
      return random_device;
    } catch (const std::exception &e) {
      MS_LOG(WARNING) << "Get std::random_device failed, retry: " << i << ", error: " << e.what();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      i++;
    }
  }
  std::mt19937 random_device{std::random_device("/dev/urandom")()};
#endif
  return random_device;
}
}  // namespace mindrecord
}  // namespace mindspore
