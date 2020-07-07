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
#ifndef DATASET_UTIL_STORAGE_CONTAINER_H_
#define DATASET_UTIL_STORAGE_CONTAINER_H_

#include <limits.h>
#include <unistd.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "dataset/util/system_pool.h"
#include "dataset/util/buddy.h"
#include "dataset/util/path.h"
#include "dataset/util/slice.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class StorageManager;

class StorageContainer {
 public:
  friend class StorageManager;

  ~StorageContainer() noexcept;

  StorageContainer(const StorageContainer &) = delete;

  StorageContainer &operator=(const StorageContainer &) = delete;

  friend std::ostream &operator<<(std::ostream &os, const StorageContainer &s);

  Status Open() noexcept;

  Status Close() noexcept;

  Status Insert(const std::vector<ReadableSlice> &buf, off64_t *offset) noexcept;

  Status Write(const ReadableSlice &dest, off64_t offset) const noexcept;

  Status Read(WritableSlice *dest, off64_t offset) const noexcept;

  Status Truncate() const noexcept;

  bool IsOpen() const { return is_open_; }

  static Status CreateStorageContainer(std::shared_ptr<StorageContainer> *out_sc, const std::string &path);

 private:
  mutable std::mutex mutex_;
  Path cont_;
  int fd_;
  bool is_open_;
  std::unique_ptr<BuddySpace> bs_;

  // Use the default value of BuddySpace
  // which can map upto 4G of space.
  explicit StorageContainer(const std::string &path) : cont_(path), fd_(-1), is_open_(false), bs_(nullptr) {}

  Status Create();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_STORAGE_CONTAINER_H_
