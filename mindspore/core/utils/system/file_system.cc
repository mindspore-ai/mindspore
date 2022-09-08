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

#if defined(SYSTEM_ENV_POSIX)
#include <sys/stat.h>
#include <unistd.h>
#elif defined(SYSTEM_ENV_WINDOWS)
#include <direct.h>
#endif

namespace mindspore {
namespace system {
#if defined(SYSTEM_ENV_POSIX)
// Implement the Posix file system
WriteFilePtr PosixFileSystem::CreateWriteFile(const string &file_name, const char *mode) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Create write file failed because the file name is null.";
    return nullptr;
  }
  auto fp = std::make_shared<PosixWriteFile>(file_name);
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Create write file(" << file_name << ") failed.";
    return nullptr;
  }
  bool result = fp->Open(mode);
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
    MS_LOG(DEBUG) << "The file(" << file_name << ") not exist.";
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
    MS_LOG(ERROR) << "Delete the file(" << file_name << ") failed." << ErrnoToString(errno);
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
    if (errno != EEXIST) {
      MS_LOG(ERROR) << "Create the dir(" << dir_name << ") failed." << ErrnoToString(errno);
      return false;
    }
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
    MS_LOG(ERROR) << "Delete the dir(" << dir_name << ") failed." << ErrnoToString(errno);
    return false;
  }
  return true;
}
#endif

#if defined(SYSTEM_ENV_WINDOWS)

// Implement the Windows file system

WriteFilePtr WinFileSystem::CreateWriteFile(const string &file_name, const char *mode) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Create write file failed because the file name is null.";
    return nullptr;
  }
  auto fp = std::make_shared<WinWriteFile>(file_name);
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Create write file(" << file_name << ") failed.";
    return nullptr;
  }
  bool result = fp->Open(mode);
  if (!result) {
    MS_LOG(ERROR) << "Open the write file(" << file_name << ") failed.";
    return nullptr;
  }
  return fp;
}

bool WinFileSystem::FileExist(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = access(file_name.c_str(), F_OK);
  if (result != 0) {
    MS_LOG(DEBUG) << "The file(" << file_name << ") not exist.";
    return false;
  }
  return true;
}

bool WinFileSystem::CreateDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = mkdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Create the dir(" << dir_name << ") is failed, error(" << result << ").";
    return false;
  }
  return true;
}

bool WinFileSystem::DeleteDir(const string &dir_name) {
  if (dir_name.empty()) {
    MS_LOG(WARNING) << "The directory name is null.";
    return false;
  }
  auto result = rmdir(dir_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the dir(" << dir_name << ") is failed, error(" << result << ").";
    return false;
  }
  return true;
}

bool WinFileSystem::DeleteFile(const string &file_name) {
  if (file_name.empty()) {
    MS_LOG(WARNING) << "The file name is null.";
    return false;
  }
  auto result = unlink(file_name.c_str());
  if (result != 0) {
    MS_LOG(ERROR) << "Delete the file(" << file_name << ") is failed." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::Open(const char *mode) {
  if (file_ != nullptr) {
    MS_LOG(WARNING) << "The File(" << file_name_ << ") already open.";
    return true;
  }
  // check the path
  if (file_name_.c_str() == nullptr) {
    MS_LOG(EXCEPTION) << "The file path is null.";
  }
  if (file_name_.size() >= PATH_MAX) {
    MS_LOG(EXCEPTION) << "The file name is too long, file name is " << file_name_ << ".";
  }

  // open the file
  file_ = fopen(file_name_.c_str(), mode);
  if (file_ == nullptr) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::Write(const std::string &data) {
  MS_LOG(DEBUG) << "Write data(" << data.size() << ") to file(" << this->file_name_ << ").";
  size_t r = fwrite(data.data(), 1, data.size(), file_);
  if (r != data.size()) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    return false;
  }
  return true;
}

bool WinWriteFile::PWrite(const void *buf, size_t nbytes, size_t offset) { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::PRead(void *buf, size_t nbytes, size_t offset) { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::Trunc(size_t length) { MS_LOG(EXCEPTION) << "Not Implement"; }

size_t WinWriteFile::Size() { MS_LOG(EXCEPTION) << "Not Implement"; }

bool WinWriteFile::Close() {
  if (file_ == nullptr) {
    MS_LOG(WARNING) << "File(" << file_name_ << ") already close.";
    return true;
  }
  bool result = true;
  if (fclose(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR." << ErrnoToString(errno);
    result = false;
  }
  file_ = nullptr;
  return result;
}

bool WinWriteFile::Flush() {
  if (fflush(file_) != 0) {
    MS_LOG(ERROR) << "File(" << file_name_ << ") IO ERROR: " << EBADF << ".";
    return false;
  }
  return true;
}

bool WinWriteFile::Sync() { return Flush(); }

WinWriteFile::~WinWriteFile() {
  try {
    if (file_ != nullptr) {
      (void)fclose(file_);
      file_ = nullptr;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when closing file.";
  } catch (...) {
    MS_LOG(ERROR) << "Non standard exception when closing file.";
  }
}
#endif
}  // namespace system
}  // namespace mindspore
