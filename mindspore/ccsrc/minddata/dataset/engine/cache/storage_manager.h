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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STORAGE_MANAGER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STORAGE_MANAGER_H_

#include <unistd.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/cache/storage_container.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/slice.h"

using ListOfContainers = std::vector<std::shared_ptr<mindspore::dataset::StorageContainer>>;
namespace mindspore {
namespace dataset {
class StorageManager : public Service {
 public:
  // Use these traits for the B+ tree inside the StorageManager
  struct StorageBPlusTreeTraits {
    // This determines the limit of number of keys in a node.
    using slot_type = uint16_t;
    // Number of slots in each leaf of the tree.
    static constexpr slot_type kLeafSlots = 512;
    // Number of slots in each inner node of the tree
    static constexpr slot_type kInnerSlots = 256;
  };
  using value_type = std::pair<int, std::pair<off_t, size_t>>;
  using storage_index = AutoIndexObj<value_type, std::allocator<value_type>, StorageBPlusTreeTraits>;
  using key_type = storage_index::key_type;
  constexpr static int32_t kMaxNumContainers = 1000;

  explicit StorageManager(const Path &);

  StorageManager(const Path &root, int pool_size);

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
  std::vector<int> writable_containers_pool_;
  int pool_size_;

  std::string GetBaseName(const std::string &prefix, int32_t file_id);

  std::string ConstructFileName(const std::string &prefix, int32_t file_id, const std::string &suffix);

  /// \brief Add a new storage container
  /// The newly-created container is going to be added into a pool of writable containers.
  /// \param replaced_container_pos If provided, will use the newly created container to replace the corresponding old
  /// container in the pool. If not provided, will just append the newly created container to the end of the pool.
  /// \return Status object
  Status AddOneContainer(int replaced_container_pos = -1);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STORAGE_MANAGER_H_
