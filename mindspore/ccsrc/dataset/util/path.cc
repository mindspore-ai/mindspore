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
#include "dataset/util/path.h"

#include <sys/stat.h>
#include <new>
#include <sstream>
#include <utility>

#include "common/utils.h"
#include "dataset/util/de_error.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
#ifdef _WIN32
char Path::separator_ = '\\';
#else
char Path::separator_ = '/';
#endif

Path::Path(const std::string &s) : path_(s) {}

Path::Path(const char *p) : path_(p) {}

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
  std::string q = path_ + p.toString();
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
  path_ += rhs.toString();
  return *this;
}

Path &Path::operator+=(const std::string &p) {
  path_ += p;
  return *this;
}

Path &Path::operator+=(const char *p) {
  path_ += p;
  return *this;
}

Path Path::operator/(const Path &p) {
  std::string q = path_ + separator_ + p.toString();
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

Status Path::CreateDirectory() {
  if (!Exists()) {
#if defined(_WIN32) || defined(_WIN64)
    int rc = mkdir(common::SafeCStr(path_));
#else
    int rc = mkdir(common::SafeCStr(path_), 0700);
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

Status Path::CreateDirectories() {
  if (IsDirectory()) {
    MS_LOG(DEBUG) << "Directory " << toString() << " already exists.";
    return Status::OK();
  } else {
    MS_LOG(DEBUG) << "Creating directory " << toString() << ".";
    std::string parent = ParentPath();
    if (!parent.empty()) {
      if (Path(parent).CreateDirectories()) {
        return CreateDirectory();
      }
    } else {
      return CreateDirectory();
    }
  }
  return Status::OK();
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
  MS_LOG(DEBUG) << "Open directory " << f->toString() << ".";
  dp_ = opendir(common::SafeCStr(f->toString()));
}

bool Path::DirIterator::hasNext() {
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

Path Path::DirIterator::next() { return (*(this->dir_) / Path(entry_->d_name)); }
}  // namespace dataset
}  // namespace mindspore
