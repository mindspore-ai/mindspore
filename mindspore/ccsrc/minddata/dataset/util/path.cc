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
#include "minddata/dataset/util/path.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <new>
#include <sstream>

#ifdef _MSC_VER
#include <direct.h>  // for _mkdir
#endif

#include "./securec.h"
#ifndef BUILD_LITE
#include "mindspore/core/utils/file_utils.h"
#else
#include "mindspore/lite/src/common/file_utils.h"
#endif
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
#if defined(_WIN32) || defined(_WIN64)
char Path::separator_ = '\\';
#else
char Path::separator_ = '/';
#endif

Path::Path(const std::string &s) {
#if defined(_WIN32) || defined(_WIN64)
  path_ = FileUtils::UTF_8ToGB2312(s.data());
#else
  path_ = s;
#endif
}

Path::Path(const char *p) {
#if defined(_WIN32) || defined(_WIN64)
  path_ = FileUtils::UTF_8ToGB2312(p);
#else
  path_ = p;
#endif
}

Path::Path(const Path &p) : path_(p.path_) {}

Path &Path::operator=(const Path &p) {
  if (&p != this) {
    this->path_ = p.path_;
  }
  return *this;
}

Path &Path::operator=(Path &&p) noexcept {
  if (&p != this) {
    this->path_ = std::move(p.path_);
  }
  return *this;
}

Path::Path(Path &&p) noexcept { this->path_ = std::move(p.path_); }

Path Path::operator+(const Path &p) {
  std::string q = path_ + p.ToString();
  return Path(q);
}

Path Path::operator+(const std::string &p) {
  std::string q = path_ + p;
  return Path(q);
}

Path Path::operator+(const char *p) {
  std::string q = path_ + p;
  return Path(q);
}

Path &Path::operator+=(const Path &rhs) {
  path_ += rhs.ToString();
  return *this;
}

Path &Path::operator+=(const std::string &p) {
#if defined(_WIN32) || defined(_WIN64)
  path_ += FileUtils::UTF_8ToGB2312(p.data());
#else
  path_ += p;
#endif
  return *this;
}

Path &Path::operator+=(const char *p) {
#if defined(_WIN32) || defined(_WIN64)
  path_ += FileUtils::UTF_8ToGB2312(p);
#else
  path_ += p;
#endif
  return *this;
}

Path Path::operator/(const Path &p) {
  std::string q = path_ + separator_ + p.ToString();
  return Path(q);
}

Path Path::operator/(const std::string &p) {
  std::string q = path_ + separator_ + p;
  return Path(q);
}

Path Path::operator/(const char *p) {
  std::string q = path_ + separator_ + p;
  return Path(q);
}

std::string Path::Extension() const {
  std::size_t found = path_.find_last_of('.');
  if (found != std::string::npos) {
    return path_.substr(found);
  } else {
    return std::string("");
  }
}

bool Path::Exists() {
  struct stat sb;
  int rc = stat(common::SafeCStr(path_), &sb);
  if (rc == -1 && errno != ENOENT) {
    MS_LOG(WARNING) << "Unable to query the status of " << path_ << ". Errno = " << errno << ".";
  }
  return (rc == 0);
}

bool Path::IsDirectory() {
  struct stat sb;
  int rc = stat(common::SafeCStr(path_), &sb);
  if (rc == 0) {
    return S_ISDIR(sb.st_mode);
  } else {
    return false;
  }
}

Status Path::CreateDirectory(bool is_common_dir) {
  if (!Exists()) {
#if defined(_WIN32) || defined(_WIN64)
#ifndef _MSC_VER
    int rc = mkdir(common::SafeCStr(path_));
#else
    int rc = _mkdir(common::SafeCStr(path_));
#endif
#else
    int rc = mkdir(common::SafeCStr(path_), S_IRUSR | S_IWUSR | S_IXUSR);
    if (rc == 0 && is_common_dir) {
      rc = chmod(common::SafeCStr(path_), S_IRWXU | S_IRWXG | S_IRWXO);
    }
#endif
    if (rc) {
      std::ostringstream oss;
      oss << "Unable to create directory " << path_ << ". Errno = " << errno;
      RETURN_STATUS_UNEXPECTED(oss.str());
    }
    return Status::OK();
  } else {
    if (IsDirectory()) {
      return Status::OK();
    } else {
      std::ostringstream oss;
      oss << "Unable to create directory " << path_ << ". It exists but is not a directory";
      RETURN_STATUS_UNEXPECTED(oss.str());
    }
  }
}

std::string Path::ParentPath() {
  std::string r("");
  std::size_t found = path_.find_last_of(separator_);
  if (found != std::string::npos) {
    if (found == 0) {
      r += separator_;
    } else {
      r = std::string(path_.substr(0, found));
    }
  }
  return r;
}

Status Path::CreateDirectories(bool is_common_dir) {
  if (IsDirectory()) {
    MS_LOG(DEBUG) << "Directory " << ToString() << " already exists.";
    return Status::OK();
  } else {
    MS_LOG(DEBUG) << "Creating directory " << ToString() << ".";
    std::string parent = ParentPath();
    if (!parent.empty()) {
      if (Path(parent).CreateDirectories(is_common_dir)) {
        return CreateDirectory(is_common_dir);
      }
    } else {
      return CreateDirectory(is_common_dir);
    }
  }
  return Status::OK();
}

Status Path::CreateCommonDirectories() { return CreateDirectories(true); }

Status Path::Remove() {
  if (Exists()) {
    if (IsDirectory()) {
#ifndef _MSC_VER
      errno_t err = rmdir(common::SafeCStr(path_));
#else
      errno_t err = _rmdir(common::SafeCStr(path_));
#endif
      if (err == -1) {
        std::ostringstream oss;
        oss << "Unable to delete directory " << path_ << ". Errno = " << errno;
        RETURN_STATUS_UNEXPECTED(oss.str());
      }
    } else {
      errno_t err = unlink(common::SafeCStr(path_));
      if (err == -1) {
        std::ostringstream oss;
        oss << "Unable to delete file " << path_ << ". Errno = " << errno;
        RETURN_STATUS_UNEXPECTED(oss.str());
      }
    }
  }
  return Status::OK();
}

Status Path::CreateFile(int *file_descriptor) { return OpenFile(file_descriptor, true); }

Status Path::OpenFile(int *file_descriptor, bool create) {
  int fd;
  if (file_descriptor == nullptr) {
    RETURN_STATUS_UNEXPECTED("null pointer");
  }
  if (IsDirectory()) {
    std::ostringstream oss;
    oss << "Unable to create file " << path_ << " which is a directory.";
    RETURN_STATUS_UNEXPECTED(oss.str());
  }
  // Convert to canonical form.
  if (strlen(common::SafeCStr(path_)) >= PATH_MAX) {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
  char canonical_path[PATH_MAX] = {0x00};
#if defined(_WIN32) || defined(_WIN64)
  auto err = _fullpath(canonical_path, common::SafeCStr(path_), PATH_MAX);
#else
  auto err = realpath(common::SafeCStr(path_), canonical_path);
#endif
  if (err == nullptr) {
    if (errno == ENOENT && create) {
      // File doesn't exist and we are to create it. Let's break it down.
      auto file_part = Basename();
      auto parent_part = ParentPath();
#if defined(_WIN32) || defined(_WIN64)
      auto parent_err = _fullpath(canonical_path, common::SafeCStr(parent_part), PATH_MAX);
#else
      auto parent_err = realpath(common::SafeCStr(parent_part), canonical_path);
#endif
      if (parent_err == nullptr) {
        RETURN_STATUS_UNEXPECTED(strerror(errno));
      }
      auto cur_inx = strlen(canonical_path);
      if (cur_inx + file_part.length() >= PATH_MAX) {
        RETURN_STATUS_UNEXPECTED(strerror(errno));
      }
      canonical_path[cur_inx++] = separator_;
      if (strncpy_s(canonical_path + cur_inx, PATH_MAX - cur_inx, common::SafeCStr(file_part), file_part.length()) !=
          EOK) {
        RETURN_STATUS_UNEXPECTED(strerror(errno));
      }
    } else {
      RETURN_STATUS_UNEXPECTED(strerror(errno));
    }
  }
  if (create) {
    fd = open(canonical_path, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP);
  } else {
    fd = open(canonical_path, O_RDWR);
  }
  if (fd == -1) {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
  *file_descriptor = fd;
  return Status::OK();
}

Status Path::CloseFile(int fd) const {
  if (close(fd) < 0) {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
  return Status::OK();
}

Status Path::TruncateFile(int fd) const {
#ifdef _MSC_VER
  int rc = _chsize(fd, 0);
#else
  int rc = ftruncate(fd, 0);
#endif
  if (rc == 0) {
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
}

std::string Path::Basename() {
  std::size_t found = path_.find_last_of(separator_);
  if (found != std::string::npos) {
    return path_.substr(found + 1);
  } else {
    return path_;
  }
}

std::shared_ptr<Path::DirIterator> Path::DirIterator::OpenDirectory(Path *f) {
  auto it = new (std::nothrow) DirIterator(f);

  if (it == nullptr) {
    return nullptr;
  }

  if (it->dp_) {
    return std::shared_ptr<DirIterator>(it);
  } else {
    delete it;
    return nullptr;
  }
}

Path::DirIterator::~DirIterator() {
  if (dp_) {
    (void)closedir(dp_);
  }
  dp_ = nullptr;
  dir_ = nullptr;
  entry_ = nullptr;
}

Path::DirIterator::DirIterator(Path *f) : dir_(f), dp_(nullptr), entry_(nullptr) {
  MS_LOG(DEBUG) << "Open directory " << f->ToString() << ".";
  dp_ = opendir(f->ToString().c_str());
}

bool Path::DirIterator::HasNext() {
  do {
    entry_ = readdir(dp_);
    if (entry_) {
      if (strcmp(entry_->d_name, ".") == 0 || strcmp(entry_->d_name, "..") == 0) {
        continue;
      }
    }
    break;
  } while (true);
  return (entry_ != nullptr);
}

Path Path::DirIterator::Next() { return (*(this->dir_) / Path(entry_->d_name)); }

Status Path::RealPath(const std::string &path, std::string &realpath_str) {
  char real_path[PATH_MAX] = {0};
  // input_path is only file_name
#if defined(_WIN32) || defined(_WIN64)
  CHECK_FAIL_RETURN_UNEXPECTED(path.length() < PATH_MAX,
                               "The length of path: " + path + " exceeds limit: " + std::to_string(PATH_MAX));
  auto ret = _fullpath(real_path, common::SafeCStr(path), PATH_MAX);
  CHECK_FAIL_RETURN_UNEXPECTED(ret != nullptr, "The file " + path + " does not exist.");
#else
  CHECK_FAIL_RETURN_UNEXPECTED(path.length() < NAME_MAX,
                               "The length of path: " + path + " exceeds limit: " + std::to_string(NAME_MAX));
  auto ret = realpath(common::SafeCStr(path), real_path);
  CHECK_FAIL_RETURN_UNEXPECTED(ret != nullptr, "The file " + path + " does not exist.");
#endif
  realpath_str = std::string(real_path);
  return Status::OK();
}

std::ostream &operator<<(std::ostream &os, const Path &s) {
  os << s.path_;
  return os;
}
}  // namespace dataset
}  // namespace mindspore
