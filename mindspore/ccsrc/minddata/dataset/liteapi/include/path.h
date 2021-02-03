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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_PATH_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_PATH_H_

#include <dirent.h>
#include <memory>
#include <string>

#include "include/status.h"

namespace mindspore {
namespace dataset {
class Path {
 public:
  class DirIterator {
   public:
    static std::shared_ptr<DirIterator> OpenDirectory(Path *f);

    ~DirIterator();

    bool hasNext();

    Path next();

   private:
    explicit DirIterator(Path *f);

    Path *dir_;
    DIR *dp_;
    struct dirent *entry_;
  };

  explicit Path(const std::string &);

  explicit Path(const char *);

  ~Path() = default;

  Path(const Path &);

  Path &operator=(const Path &);

  Path(Path &&) noexcept;

  Path &operator=(Path &&) noexcept;

  std::string toString() const { return path_; }

  Path operator+(const Path &);

  Path operator+(const std::string &);

  Path operator+(const char *);

  Path &operator+=(const Path &rhs);

  Path &operator+=(const std::string &);

  Path &operator+=(const char *);

  Path operator/(const Path &);

  Path operator/(const std::string &);

  Path operator/(const char *);

  bool operator==(const Path &rhs) const { return (path_ == rhs.path_); }

  bool operator!=(const Path &rhs) const { return (path_ != rhs.path_); }

  bool operator<(const Path &rhs) const { return (path_ < rhs.path_); }

  bool operator>(const Path &rhs) const { return (path_ > rhs.path_); }

  bool operator<=(const Path &rhs) const { return (path_ <= rhs.path_); }

  bool operator>=(const Path &rhs) const { return (path_ >= rhs.path_); }

  bool Exists();

  bool IsDirectory();

  Status CreateDirectory();

  Status CreateDirectories();

  std::string Extension() const;

  std::string ParentPath();

  Status Remove();

  Status CreateFile(int *fd);

  Status OpenFile(int *fd, bool create = false);

  Status CloseFile(int fd) const;

  Status TruncateFile(int fd) const;

  std::string Basename();

  friend std::ostream &operator<<(std::ostream &os, const Path &s);

 private:
  static char separator_;
  std::string path_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_PATH_H_
