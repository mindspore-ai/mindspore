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
CachedSharedMemoryArena::CachedSharedMemoryArena(int32_t port, size_t val_in_GB) : val_in_GB_(val_in_GB), port_(port) {
  // We create the shared memory and we will destroy it. All other client just detach only.
  shm_.RemoveResourcesOnExit();
}
CachedSharedMemoryArena::~CachedSharedMemoryArena() {}

Status CachedSharedMemoryArena::CreateArena(std::unique_ptr<CachedSharedMemoryArena> *out, int32_t port,
                                            size_t val_in_GB) {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto ba = new (std::nothrow) CachedSharedMemoryArena(port, val_in_GB);
  if (ba == nullptr) {
    return Status(StatusCode::kOutOfMemory);
  }
  // Transfer the ownership of this pointer. Any future error in the processing we will have
  // the destructor of *out to deal.
  (*out).reset(ba);
  // Generate the ftok using a combination of port.
  SharedMemory::shm_key_t shm_key;
  RETURN_IF_NOT_OK(PortToFtok(port, &shm_key));
  ba->shm_.SetPublicKey(shm_key);
  // Value is in GB. Convert into bytes.
  int64_t sz = val_in_GB * 1073741824L;
  RETURN_IF_NOT_OK(ba->shm_.Create(sz));
  ba->impl_ = std::make_unique<ArenaImpl>(ba->shm_.SharedMemoryBaseAddr(), sz);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
