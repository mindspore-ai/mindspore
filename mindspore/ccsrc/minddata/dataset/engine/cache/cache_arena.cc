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
#include "minddata/dataset/engine/cache/cache_arena.h"
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/path.h"
namespace mindspore {
namespace dataset {
CachedSharedMemory::CachedSharedMemory(int32_t port, size_t val_in_GB)
    : shared_memory_sz_in_gb_(val_in_GB), port_(port), num_numa_nodes_(-1), sub_pool_sz_(-1) {
  // We create the shared memory and we will destroy it. All other client just detach only.
  shm_.RemoveResourcesOnExit();
}
CachedSharedMemory::~CachedSharedMemory() = default;

Status CachedSharedMemory::Init() {
  CacheServer &cs = CacheServer::GetInstance();
  num_numa_nodes_ = cs.GetNumaNodeCount();
  // Generate the ftok using a combination of port.
  SharedMemory::shm_key_t shm_key;
  RETURN_IF_NOT_OK(PortToFtok(port_, &shm_key));
  shm_.SetPublicKey(shm_key);
  // Value is in GB. Convert into bytes.
  int64_t shm_mem_sz = shared_memory_sz_in_gb_ * 1073741824L;
  RETURN_IF_NOT_OK(shm_.Create(shm_mem_sz));
  MS_LOG(INFO) << "Creation of shared memory successful. Shared memory key " << shm_.GetKey();
  // Interleave the memory.
  cs.GetHWControl()->InterleaveMemory(shm_.SharedMemoryBaseAddr(), shm_mem_sz);
  // We will create a number of sub pool out of shared memory to reduce latch contention
  int32_t num_of_pools = num_numa_nodes_;
  if (num_numa_nodes_ == 1) {
    num_of_pools = shared_memory_sz_in_gb_ * 2;
  }
  sub_pool_sz_ = shm_mem_sz / num_of_pools;
  // If each subpool is too small, readjust the number of pools
  constexpr int64 min_subpool_sz = 512 * 1048576L;
  if (sub_pool_sz_ < min_subpool_sz) {
    sub_pool_sz_ = min_subpool_sz;
    num_of_pools = shm_mem_sz / min_subpool_sz;
  }
  shm_pool_.reserve(num_of_pools);
  for (auto i = 0; i < num_of_pools; ++i) {
    void *ptr = static_cast<char *>(shm_.SharedMemoryBaseAddr()) + i * sub_pool_sz_;
    shm_pool_.push_back(std::make_unique<ArenaImpl>(ptr, sub_pool_sz_));
  }
  mux_ = std::make_unique<std::mutex[]>(num_of_pools);
  return Status::OK();
}

Status CachedSharedMemory::CreateArena(std::unique_ptr<CachedSharedMemory> *out, int32_t port, size_t val_in_GB) {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto mem_pool = std::unique_ptr<CachedSharedMemory>(new CachedSharedMemory(port, val_in_GB));
  RETURN_IF_NOT_OK(mem_pool->Init());
  *out = std::move(mem_pool);
  return Status::OK();
}

Status CachedSharedMemory::AllocateSharedMemory(int32_t client_id, size_t sz, void **p) {
  Status rc;
  RETURN_UNEXPECTED_IF_NULL(p);
  auto begin_slot = client_id % shm_pool_.size();
  auto slot = begin_slot;
  do {
    std::unique_lock<std::mutex> lock(mux_[slot]);
    rc = shm_pool_[slot]->Allocate(sz, p);
    if (rc == StatusCode::kMDOutOfMemory) {
      slot = (slot + 1) % shm_pool_.size();
    }
  } while (rc.IsError() && slot != begin_slot);
  if (rc.IsError()) {
    return rc;
  }
  return Status::OK();
}

void CachedSharedMemory::DeallocateSharedMemory(int32_t client_id, void *p) {
  auto begin_slot = client_id % shm_pool_.size();
  auto slot = begin_slot;
  auto start_addr = static_cast<char *>(SharedMemoryBaseAddr());
  bool found = false;
  do {
    auto ptr = start_addr + slot * sub_pool_sz_;
    if (ptr <= p && p < (ptr + sub_pool_sz_)) {
      std::unique_lock<std::mutex> lock(mux_[slot]);
      shm_pool_[slot]->Deallocate(p);
      found = true;
      break;
    } else {
      slot = (slot + 1) % shm_pool_.size();
    }
  } while (slot != begin_slot);
  if (!found) {
    MS_LOG(ERROR) << "Programming error. Can't find the arena the pointer " << p << " comes from";
  }
}
}  // namespace dataset
}  // namespace mindspore
