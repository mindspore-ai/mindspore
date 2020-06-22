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
#ifndef DATASET_UTIL_STORAGE_MANAGER_H_
#define DATASET_UTIL_STORAGE_MANAGER_H_

#include <unistd.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dataset/util/allocator.h"
#include "dataset/util/auto_index.h"
#include "dataset/util/lock.h"
#include "dataset/util/memory_pool.h"
#include "dataset/util/path.h"
#include "dataset/util/service.h"
#include "dataset/util/slice.h"
#include "dataset/util/storage_container.h"

using ListOfContainers = std::vector<std::shared_ptr<mindspore::dataset::StorageContainer>>;
namespace mindspore {
namespace dataset {
class StorageManager : public Service {
 public:
  using storage_index = AutoIndexObj<std::pair<int, std::pair<off_t, size_t>>>;
  using key_type = storage_index::key_type;
  using value_type = storage_index::value_type;

  explicit StorageManager(const Path &);

  ~StorageManager() override;

  StorageManager(const StorageManager &) = delete;

  StorageManager &operator=(const StorageManager &) = delete;

  Status Write(key_type *out_key, const std::vector<ReadableSlice> &buf);

  Status Read(key_type key, WritableSlice *dest, size_t *bytesRead) const;

  Status DoServiceStart() override;

  Status DoServiceStop() noexcept override;

  friend std::ostream &operator<<(std::ostream &os, const StorageManager &s);

 private:
  Path root_;
  ListOfContainers containers_;
  int file_id_;
  RWLock rw_lock_;
  storage_index index_;

  std::string GetBaseName(const std::string &prefix, int32_t file_id);

  std::string ConstructFileName(const std::string &prefix, int32_t file_id, const std::string &suffix);

  Status AddOneContainer();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_STORAGE_MANAGER_H_
