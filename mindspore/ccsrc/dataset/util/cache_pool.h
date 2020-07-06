/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef DATASET_UTIL_CACHE_POOL_H_
#define DATASET_UTIL_CACHE_POOL_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "dataset/util/allocator.h"
#include "dataset/util/service.h"
#include "dataset/util/slice.h"
#include "dataset/util/storage_manager.h"
#include "dataset/util/auto_index.h"

namespace mindspore {
namespace dataset {
/// \brief A CachePool provides service for backup/restore a buffer. A buffer can be represented in a form of vector of
/// ReadableSlice where all memory blocks will be copied to one contiguous block which can be in memory or spilled to
/// disk (if a disk directory is provided). Every buffer insert will return a generated key which can be used to
/// restore the buffer.
/// \see ReadableSlice
class CachePool : public Service {
 public:
  using base_type = uint8_t;
  using pointer = base_type *;
  using const_pointer = const base_type *;
  using reference = base_type &;
  using const_reference = const base_type &;
  using value_allocator = Allocator<base_type>;

  // An internal class to locate the whereabouts of a backed up buffer which can be either in
  class DataLocator {
   public:
    DataLocator() : ptr(nullptr), sz(0), storage_key(0) {}
    ~DataLocator() = default;
    DataLocator(const DataLocator &other) = default;
    DataLocator &operator=(const DataLocator &other) = default;
    DataLocator(DataLocator &&other) noexcept {
      ptr = other.ptr;
      sz = other.sz;
      storage_key = other.storage_key;
      other.ptr = nullptr;
      other.sz = 0;
      other.storage_key = 0;
    }
    DataLocator &operator=(DataLocator &&other) noexcept {
      if (&other != this) {
        ptr = other.ptr;
        sz = other.sz;
        storage_key = other.storage_key;
        other.ptr = nullptr;
        other.sz = 0;
        other.storage_key = 0;
      }
      return *this;
    }
    pointer ptr;
    size_t sz;
    StorageManager::key_type storage_key;
  };

  using data_index = AutoIndexObj<DataLocator>;
  using key_type = data_index::key_type;
  using bl_alloc_type = typename value_allocator::template rebind<DataLocator>::other;

  /// \brief Simple statistics returned from CachePool like how many elements are cached in memory and
  /// how many elements are spilled to disk.
  struct CacheStat {
    int64_t num_mem_cached;
    int64_t num_disk_cached;
  };

  /// \brief Constructor
  /// \param alloc Allocator to allocate memory from
  /// \param root Optional disk folder to spill
  explicit CachePool(const value_allocator &alloc, const std::string &root = "");

  CachePool(const CachePool &) = delete;
  CachePool(CachePool &&) = delete;
  CachePool &operator=(const CachePool &) = delete;
  CachePool &operator=(CachePool &&) = delete;
  ~CachePool() noexcept;

  Status DoServiceStart() override;
  Status DoServiceStop() override;

  Path GetSpillPath() const;

  /// \brief Insert a sequence of ReadableSlice objects into the pool.
  /// All memory blocks will be consolidated into one contiguous block and be cached in either memory or on disk.
  /// \param[in] buf A sequence of ReadableSlice objects.
  /// \param[out] key Generated key
  /// \return Error code
  Status Insert(const std::vector<ReadableSlice> &buf, key_type *key);
  /// \brief Restore a cached buffer (from memory or disk)
  /// \param[in] key A previous key returned from Insert
  /// \param[out] dest The cached buffer will be copied to this destination represented by a WritableSlice
  /// \param[out] bytesRead Optional. Number of bytes read.
  /// \return Error code
  Status Read(key_type key, WritableSlice *dest, size_t *bytesRead = nullptr) const;

  Status Spill(DataLocator *dl);

  Status Locate(DataLocator *dl);

  size_t GetSize(key_type key) const;

  /// \brief Get statistics.
  /// \return CacheStat object
  CacheStat GetStat() const;

  const value_allocator &get_allocator() const;

  std::string MyName() const { return subfolder_; }

 private:
  value_allocator alloc_;
  Path root_;
  const std::string subfolder_;
  std::shared_ptr<StorageManager> sm_;
  std::shared_ptr<data_index> tree_;
};
}  // namespace dataset
}  // namespace mindspore
#endif
