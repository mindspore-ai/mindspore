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

#include "utils/system/file_system.h"
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <deque>

namespace mindspore {
namespace system {

#if defined(SYSTEM_ENV_POSIX)
// Implement the Posix file systen
WriteFilePtr PosixFileSystem::CreateWriteFile(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Create write file failed because the file name is null.";
    return nullptr;
  }
  auto fp = std::make_shared<PosixWriteFile>(file_name);
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Create write file(" << file_name << ") failed.";
    return nullptr;
  }
  bool result = fp->Open();
  if (!result) {
    MS_LOG(ERROR) << "Open the write file(" << file_name << ") failed.";
    return nullptr;
  }
  return fp;
}

bool PosixFileSystem::FileExist(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = access(file_name.c_str(), F_OK);
  if (result != 0) {
    MS_LOG(INFO) << "The file(" << file_name << ") not exist.";
    return false;
  }
  return true;
}

bool PosixFileSystem::DeleteFile(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = unlink(file_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the file(" << file_name << ") is falire, error(" << errno << ").";
    return false;
  }
  return true;
}

static const int DEFAULT_MKDIR_MODE = 0700;
bool PosixFileSystem::CreateDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = mkdir(dir_name.c_str(), DEFAULT_MKDIR_MODE);
  if (result != 0) {
    MS_LOG(ERROR) << "Create the dir(" << dir_name << ") is falire, error(" << errno << ").";
    return false;
  }
  return true;
}

bool PosixFileSystem::DeleteDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = rmdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the dir(" << dir_name << ") is falire, error(" << errno << ").";
    return false;
  }
  return true;
}
#endif

}  // namespace system
}  // namespace mindspore
