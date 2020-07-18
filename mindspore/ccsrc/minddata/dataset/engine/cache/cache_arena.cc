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
#include "minddata/dataset/util/path.h"
namespace mindspore {
namespace dataset {
CachedSharedMemoryArena::CachedSharedMemoryArena(int32_t port, size_t val_in_GB)
    : Arena::Arena(val_in_GB * 1024), port_(port), shmid_(-1) {}

CachedSharedMemoryArena::~CachedSharedMemoryArena() {
#if CACHE_LOCAL_CLIENT
  if (this->ptr_ != nullptr && this->ptr_ != reinterpret_cast<void *>(-1)) {
    shmdt(this->ptr_);
  }
  this->ptr_ = nullptr;
  if (shmid_ != -1) {
    shmctl(shmid_, IPC_RMID, nullptr);
    // Also remove the path we use to generate ftok.
    Path p(PortToUnixSocketPath(port_));
    (void)p.Remove();
  }
#endif
}

Status CachedSharedMemoryArena::CreateArena(std::unique_ptr<CachedSharedMemoryArena> *out, int32_t port,
                                            size_t val_in_GB) {
  RETURN_UNEXPECTED_IF_NULL(out);
#if CACHE_LOCAL_CLIENT
  auto ba = new (std::nothrow) CachedSharedMemoryArena(port, val_in_GB);
  if (ba == nullptr) {
    return Status(StatusCode::kOutOfMemory);
  }
  // Transfer the ownership of this pointer. Any future error in the processing we will have
  // the destructor of *out to deal.
  (*out).reset(ba);
  // Generate the ftok using a combination of port.
  int err;
  auto shm_key = PortToFtok(port, &err);
  if (shm_key == (key_t)-1) {
    std::string errMsg = "Ftok failed with errno " + std::to_string(err);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  auto access_mode = S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP;
  ba->shmid_ = shmget(shm_key, ba->size_in_bytes_, IPC_CREAT | IPC_EXCL | access_mode);
  if (ba->shmid_) {
    ba->ptr_ = shmat(ba->shmid_, nullptr, 0);
    if (ba->ptr_ == reinterpret_cast<void *>(-1)) {
      RETURN_STATUS_UNEXPECTED("Shared memory attach failed. Errno " + std::to_string(errno));
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Shared memory creation failed. Errno " + std::to_string(errno));
  }
  uint64_t num_blks = ba->size_in_bytes_ / ARENA_BLK_SZ;
  MS_LOG(DEBUG) << "Size of memory pool is " << num_blks << ", number of blocks of size is " << ARENA_BLK_SZ << ".";
  ba->tr_.Insert(0, num_blks);
#endif
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
