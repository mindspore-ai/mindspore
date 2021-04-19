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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_CACHE_POOL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_CACHE_POOL_H_

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_numa.h"
#include "minddata/dataset/engine/cache/storage_manager.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/slice.h"
#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/btree.h"

namespace mindspore {
namespace dataset {
/// \brief A CachePool provides service for backup/restore a buffer. A buffer can be represented in a form of vector of
/// ReadableSlice where all memory blocks will be copied to one contiguous block which can be in memory or spilled to
/// disk (if a disk directory is provided). User must provide a key to insert the buffer.
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
    DataLocator() : ptr(nullptr), sz(0), node_id(0), node_hit(false), storage_key(0) {}
    ~DataLocator() = default;
    DataLocator(const DataLocator &other) = default;
    DataLocator &operator=(const DataLocator &other) = default;
    DataLocator(DataLocator &&other) noexcept {
      ptr = other.ptr;
      sz = other.sz;
      node_id = other.node_id;
      node_hit = other.node_hit;
      storage_key = other.storage_key;
      other.ptr = nullptr;
      other.sz = 0;
      other.storage_key = 0;
    }
    DataLocator &operator=(DataLocator &&other) noexcept {
      if (&other != this) {
        ptr = other.ptr;
        sz = other.sz;
        node_id = other.node_id;
        node_hit = other.node_hit;
        storage_key = other.storage_key;
        other.ptr = nullptr;
        other.sz = 0;
        other.storage_key = 0;
      }
      return *this;
    }
    pointer ptr;
    size_t sz;
    numa_id_t node_id;  // where the numa node the memory is allocated to
    bool node_hit;      // we can allocate to the preferred node
    StorageManager::key_type storage_key;
  };

  using data_index = BPlusTree<int64_t, DataLocator>;
  using key_type = data_index::key_type;
  using bl_alloc_type = typename value_allocator::template rebind<DataLocator>::other;

  /// \brief Simple statistics returned from CachePool like how many elements are cached in memory and
  /// how many elements are spilled to disk.
  struct CacheStat {
    key_type min_key;
    key_type max_key;
    int64_t num_mem_cached;
    int64_t num_disk_cached;
    int64_t average_cache_sz;
    int64_t num_numa_hit;
    std::vector<key_type> gap;
  };

  /// \brief Constructor
  /// \param alloc Allocator to allocate memory from
  /// \param root Optional disk folder to spill
  explicit CachePool(std::shared_ptr<NumaMemoryPool> mp, const std::string &root = "");

  CachePool(const CachePool &) = delete;
  CachePool(CachePool &&) = delete;
  CachePool &operator=(const CachePool &) = delete;
  CachePool &operator=(CachePool &&) = delete;
  ~CachePool() noexcept override;

  Status DoServiceStart() override;
  Status DoServiceStop() override;

  Path GetSpillPath() const;

  /// \brief Insert a sequence of ReadableSlice objects into the pool.
  /// All memory blocks will be consolidated into one contiguous block and be cached in either memory or on disk.
  /// \param[in] key User supplied key
  /// \param[in] buf A sequence of ReadableSlice objects.
  /// \param[in] writeToDiskDirectly If true, no spill to disk if spill is enabled, or return no memory
  /// \return Error code
  Status Insert(CachePool::key_type key, const std::vector<ReadableSlice> &buf);

  /// \brief Restore a cached buffer (from memory or disk)
  /// \param[in] key A previous key returned from Insert
  /// \param[out] dest The cached buffer will be copied to this destination represented by a WritableSlice
  /// \param[out] bytesRead Optional. Number of bytes read.
  /// \return Error code
  Status Read(key_type key, WritableSlice *dest, size_t *bytesRead = nullptr) const;

  /// \brief Serialize a DataLocator
  Status GetDataLocator(key_type, const std::shared_ptr<flatbuffers::FlatBufferBuilder> &,
                        flatbuffers::Offset<DataLocatorMsg> *) const;

  /// \brief Get statistics.
  /// \return CacheStat object
  CacheStat GetStat(bool GetMissingKeys = false) const;

  std::string MyName() const { return subfolder_; }

  /// \brief Toggle locking
  /// \note Once locking is off. It is user's responsibility to ensure concurrency
  void SetLocking(bool on_off) { tree_->SetLocking(on_off); }

 private:
  std::shared_ptr<NumaMemoryPool> mp_;
  Path root_;
  const std::string subfolder_;
  std::shared_ptr<StorageManager> sm_;
  std::shared_ptr<data_index> tree_;
  std::atomic<uint64_t> soft_mem_limit_;  // the available memory in the machine
  std::atomic<uint64_t> temp_mem_usage_;  // temporary count on the amount of memory usage by cache every 100Mb (because
                                          // we will adjust soft_mem_limit_ every 100Mb based on this parameter)
  uint64_t min_avail_mem_;                // lower bound of the available memory
  const int kMemoryCapAdjustInterval = 104857600;
};
}  // namespace dataset
}  // namespace mindspore
#endif
