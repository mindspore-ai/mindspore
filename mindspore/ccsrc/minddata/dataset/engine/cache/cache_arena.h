/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_

#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <utility>
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_ipc.h"
namespace mindspore {
namespace dataset {
/// This is like a CircularPool but each arena is in shared memory and
/// possibly bind to a numa socket.
class CachedSharedMemory {
 public:
  // Disable copy and assignment constructor
  CachedSharedMemory(const CachedSharedMemory &) = delete;
  CachedSharedMemory &operator=(const CachedSharedMemory &) = delete;
  ~CachedSharedMemory();

  /// \brief Create an Arena in shared memory
  /// \param[out] p_ba Pointer to a unique_ptr
  /// \param shmkey Shared memory key
  /// \param val_in_GB size of shared memory in gigabyte
  /// \return Status object
  static Status CreateArena(std::unique_ptr<CachedSharedMemory> *out, int32_t port, size_t val_in_GB);

  /// \brief This returns where we attach to the shared memory.
  /// Some gRPC requests will ask for a shared memory block, and
  /// we can't return the absolute address as this makes no sense
  /// in the client. So instead we will return an address relative
  /// to the base address of the shared memory where we attach to.
  /// \return Base address of the shared memory.
  const void *SharedMemoryBaseAddr() const { return shm_.SharedMemoryBaseAddr(); }
  void *SharedMemoryBaseAddr() { return shm_.SharedMemoryBaseAddr(); }

  /// \brief Get the shared memory key of the shared memory
  SharedMemory::shm_key_t GetKey() const { return shm_.GetKey(); }

  /// \brief Allocate shared memory for a given pipeline
  Status AllocateSharedMemory(int32_t client_id, size_t sz, void **p);

  /// \brief Deallocate shared memory for a given pipeline
  void DeallocateSharedMemory(int32_t client_id, void *p);

 private:
  int32_t shared_memory_sz_in_gb_;
  int32_t port_;
  SharedMemory shm_;
  std::vector<std::unique_ptr<ArenaImpl>> shm_pool_;
  std::unique_ptr<std::mutex[]> mux_;
  int32_t num_numa_nodes_;
  int64_t sub_pool_sz_;
  /// Private constructor. Not to be called directly.
  CachedSharedMemory(int32_t port, size_t val_in_GB);
  Status Init();
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_
