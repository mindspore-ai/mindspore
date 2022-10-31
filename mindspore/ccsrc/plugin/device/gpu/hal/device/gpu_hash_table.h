/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_

#include <cuda.h>
#if CUDA_VERSION > 11000
#include <curand_kernel.h>
#include <cuda/std/atomic>

#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include "runtime/device/hash_table.h"
#include "plugin/device/gpu/hal/device/gpu_allocator.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

namespace mindspore {
namespace device {
namespace gpu {
constexpr static int kMaxThreadsPerBlockRandomGen = 2048;
constexpr static size_t kInitialCapacity = 100000;
constexpr static uint32_t kTileSize = 4;
constexpr static uint32_t kBlockSize = 128;
constexpr static size_t kImportTensorNum = 3;
constexpr static char kNormalDistribution[] = "normal";
constexpr static char kZerosDistribution[] = "zeros";
constexpr static char kOnesDistribution[] = "ones";

template <typename Key, typename Value, typename Allocator>
class CudaDynamicMap;
// A hash table base on GPU.
template <typename Key, typename Value, typename Allocator = GPUAllocator<char>>
class GPUHashTable : public HashTable<Key, Value> {
 public:
  using Status = typename HashTable<Key, Value>::Status;

  GPUHashTable(int32_t value_dim, const std::string &initializer, const Allocator &alloc = Allocator());
  GPUHashTable(int32_t value_dim, const Value &default_value, const Allocator &alloc = Allocator());
  ~GPUHashTable();

  // The Allocator type used allocate gpu memory for 'Key' type.
  using KeyAllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<Key>;
  // The Allocator type used allocate gpu memory for 'Value' type.
  using ValueAllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<Value>;
  // The Allocator type used allocate gpu memory for 'index' type(int).
  using IndexAllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<int>;
  // The general Allocator type used allocate gpu memory.
  using CharAllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<char>;

  // Find elements with specific keys, if a key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map. The initializer can be 'normal', 'zero' or 'one', and also
  // could be a specific 'Value' type scalar.
  bool Find(const Key *keys, size_t key_num, Value *outputs, void *stream) override;

  // Insert elements with specific keys. If key exists, update the value of the key.
  bool Insert(const Key *keys, size_t key_num, const Value *value, void *stream) override;

  // Erase elements with specific keys.
  bool Erase(const Key *keys, size_t key_num, void *stream) override;

  // Reserves space for at least the specified number of elements.
  bool Reserve(size_t new_capacity, void *stream) override;

  // Export all keys and values in hash map, the order of each element of keys and values is consistent.
  // Note: Even if the elements of the hash map are unchanged, the order of the key-value pair returned by the function
  // may be different each time it is called, because there may be multi-threaded concurrent exports inside the
  // function.
  bool GetKeysAndValues(Key *keys, Value *values, void *stream) override;

  // Import keys, values into the device hash map.
  bool Import(const DataLenPair &input_data) override;

  // Export all keys, values and status to host side.
  bool Export(const DataLenPair &keys, const DataLenPair &values, const DataLenPair &status) override;

  // Get the number of elements that can be held in currently allocated storage.
  size_t capacity() const override { return capacity_; }

  // Get the number of elements.
  size_t size() const override { return size_; }

  // Clear all elements of hash table.
  bool Clear();

 private:
  // Find elements with specific keys, if the key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map.The initializer can be 'normal', 'zeros' or 'ones'.
  bool Find(const Key *keys, size_t key_num, const std::string &initializer, Value *outputs, void *stream);

  // Find elements with specific keys, if the key does not exist, initialize the value for the key by 'default_value'
  // and insert the key-value pair into map.
  bool Find(const Key *keys, size_t key_num, const Value &default_value, Value *outputs, void *stream);

  // Get all indices in blocks according to the key.
  bool GetIndicesByKeys(const Key *key, size_t key_num, bool insert_miss_key, int32_t *indices, cudaStream_t stream);

  // Insert default value according to initializer, initializer can be 'normal', 'zeros' or 'ones'.
  bool InsertDefaultValueByInitializer(size_t key_num, const std::string &initializer, const int *indices,
                                       cudaStream_t stream);

  // initialize normal distribution random generator states on GPU.
  bool InitNormalDistRandomGenerator(cudaStream_t stream);

  // Initialize the hash map: create cuda dynamic map and initialize the atomic counter.
  void Initialize(const Allocator &alloc);
  // Finalize the hash map: destroy cuda dynamic map and free the GPU memory for the atomic counter and blocks.
  void Finalize();

  // Allocate GPU memory use char_alloc_.
  template <typename T>
  void AllocateMemory(size_t size, T **ptr);
  // Free GPU memory use char_alloc_.
  void FreeMemory(void *ptr);

  // Reset the buffer that record block and idle status.
  bool ResetBlockAndIdleFlag(cudaStream_t stream);

  // Record all block memory that contain all values.
  std::vector<Value *> blocks_;
  // Record all first address of blocks, the buffer is on device memory.
  Value **blocks_ptr_{nullptr};

  // Record whether every slot is occupied or not for all block.
  std::vector<bool *> idle_flags_;
  // Record whether every slot is occupied or not for all block, the buffer is on device memory.
  bool **idle_flags_ptr_{nullptr};

  // The sentinel record the latest used location in blocks.
  cuda::atomic<std::size_t, cuda::thread_scope_device> *current_index_{nullptr};

  // The counter record the idle slot number, if the contents of a slot are erased, the slot is marked with the idle
  // status.
  cuda::atomic<int32_t, cuda::thread_scope_device> *erased_counter_{nullptr};
  // The buffer keep all the idle slot position(offset index to the beginning of block).
  int32_t *erased_slot_{nullptr};

  // The value dimension for each key.
  size_t value_dim_;

  // The initializer used to initialize the values for missing keys, the initializer could be 'normal', 'zero' or 'one'.
  std::string initializer_;
  // The default value used to initialize the values for missing keys.
  Value default_value_;

  // The common allocator used to alloacte gpu memory.
  CharAllocatorType char_alloc_;

  // The dynamic hash table on GPU.
  std::unique_ptr<CudaDynamicMap<Key, int32_t, Allocator>> cuda_dynamic_map_{nullptr};

  // Record the number of elements in the map.
  size_t size_{0};

  // Record the number of elements that can be held in currently allocated storage
  size_t capacity_{0};

  // The number of elements of one block.
  size_t elements_per_block_{kInitialCapacity};

  // Record the number of successfully inserted keys.
  cuda::atomic<std::size_t, cuda::thread_scope_device> *insert_success_number_{nullptr};

  // The flag record whether normal distribution random generator state is initialized.
  std::atomic_bool random_gen_init_{false};
  // The random generator states used to generate normal distribution.
  curandStatePhilox4_32_10_t *random_gen_state_{nullptr};

  // The block size used to launch cuda kernel for inserting normal distribution random values.
  int random_gen_threads_per_block_{GET_THREADS};
  // The grid size used to launch cuda kernel for inserting normal distribution random values.
  int random_gen_block_count_{(kMaxThreadsPerBlockRandomGen - 1) / random_gen_threads_per_block_ + 1};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_
