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

#include "mindrecord/include/common/shard_utils.h"
#include "common/utils.h"
#include "./securec.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::DEBUG;
using mindspore::MsLogLevel::ERROR;

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
  return std::move(res);
}

bool ValidateFieldName(const std::string &str) {
  std::string::const_iterator it = str.begin();
  if (it == str.end()) {
    return false;
  }
  for (; it != str.end(); ++it) {
    if (*it == '_' || ((*it >= '0') && (*it <= '9')) || ((*it >= 'A') && (*it <= 'Z')) ||
        ((*it >= 'a') && (*it <= 'z'))) {
      continue;
    }
    return false;
  }
  return true;
}

std::pair<MSRStatus, std::string> GetFileName(const std::string &path) {
  char real_path[PATH_MAX] = {0};
  char buf[PATH_MAX] = {0};
  if (strncpy_s(buf, PATH_MAX, common::SafeCStr(path), path.length()) != EOK) {
    MS_LOG(ERROR) << "Securec func [strncpy_s] failed, path: " << path;
    return {FAILED, ""};
  }
  char tmp[PATH_MAX] = {0};
  if (realpath(dirname(&(buf[0])), tmp) == nullptr) {
    MS_LOG(ERROR) << "Invalid file path, path: " << buf;
    return {FAILED, ""};
  }
  if (realpath(common::SafeCStr(path), real_path) == nullptr) {
    MS_LOG(DEBUG) << "Path: " << path << "check successfully";
  }
  std::string s = real_path;
  char sep = '/';
  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos) {
    if (i + 1 < s.size()) {
      return {SUCCESS, s.substr(i + 1)};
    }
  }
  return {SUCCESS, s};
}

std::pair<MSRStatus, std::string> GetParentDir(const std::string &path) {
  char real_path[PATH_MAX] = {0};
  char buf[PATH_MAX] = {0};
  if (strncpy_s(buf, PATH_MAX, common::SafeCStr(path), path.length()) != EOK) {
    MS_LOG(ERROR) << "Securec func [strncpy_s] failed, path: " << path;
    return {FAILED, ""};
  }
  char tmp[PATH_MAX] = {0};
  if (realpath(dirname(&(buf[0])), tmp) == nullptr) {
    MS_LOG(ERROR) << "Invalid file path, path: " << buf;
    return {FAILED, ""};
  }
  if (realpath(common::SafeCStr(path), real_path) == nullptr) {
    MS_LOG(DEBUG) << "Path: " << path << "check successfully";
  }
  std::string s = real_path;
  if (s.rfind('/') + 1 <= s.size()) {
    return {SUCCESS, s.substr(0, s.rfind('/') + 1)};
  }
  return {SUCCESS, "/"};
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

bool IsLegalFile(const std::string &path) {
  struct stat s;
  if (stat(common::SafeCStr(path), &s) == 0) {
    if (s.st_mode & S_IFDIR) {
      return false;
    }
    return true;
  }
  return false;
}

std::pair<MSRStatus, uint64_t> GetDiskSize(const std::string &str_dir, const DiskSizeType &disk_type) {
  uint64_t ll_count = 0;
  struct statfs disk_info;
  if (statfs(common::SafeCStr(str_dir), &disk_info) == -1) {
    MS_LOG(ERROR) << "Get disk size error";
    return {FAILED, 0};
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

  return {SUCCESS, ll_count};
}

uint32_t GetMaxThreadNum() {
  // define the number of thread
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) {
    thread_num = kMaxConsumerCount;
  }
  return thread_num;
}
}  // namespace mindrecord
}  // namespace mindspore
