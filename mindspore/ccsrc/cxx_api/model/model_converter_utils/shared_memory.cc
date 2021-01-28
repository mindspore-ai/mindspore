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
#include "cxx_api/model/model_converter_utils/shared_memory.h"
#include <sys/shm.h>
#include <sys/stat.h>
#include <string>
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
Status SharedMemory::Create(uint64_t memory_size) {
  auto access_mode = S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP;
  shm_id_ = shmget(IPC_PRIVATE, memory_size, IPC_CREAT | IPC_EXCL | access_mode);
  if (shm_id_ == -1) {
    MS_LOG_ERROR << "Shared memory creation failed. Errno " + std::to_string(errno);
    return kMCFailed;
  }
  MS_LOG_INFO << "shmget success, shm id " << shm_id_;
  return kSuccess;
}

Status SharedMemory::Attach() {
  void *shmat_addr = shmat(shm_id_, nullptr, 0);
  if (shmat_addr == reinterpret_cast<void *>(-1)) {
    MS_LOG_ERROR << "Shared memory attach failed. Errno " + std::to_string(errno);
    return kMCFailed;
  }
  shmat_addr_ = reinterpret_cast<uint8_t *>(shmat_addr);
  return kSuccess;
}

void SharedMemory::Detach() {
  if (shmat_addr_) {
    auto err = shmdt(shmat_addr_);
    if (err == -1) {
      MS_LOG_ERROR << "Shared memory detach failed. Errno " + std::to_string(errno);
      return;
    }
  }
  shmat_addr_ = nullptr;
}

void SharedMemory::Destroy() {
  // Remove the shared memory and never mind about the return code.
  auto err = shmctl(shm_id_, IPC_RMID, nullptr);
  if (err == -1) {
    std::string errMsg = "Unable to remove shared memory with id " + std::to_string(shm_id_);
    errMsg += ". Errno :" + std::to_string(errno);
    errMsg += "\nPlesae remove it manually using ipcrm -m command";
    MS_LOG_ERROR << errMsg;
  }
}
}  // namespace mindspore
