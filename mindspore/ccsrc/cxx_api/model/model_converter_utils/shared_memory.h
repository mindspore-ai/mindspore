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

#ifndef MINDSPORE_CCSRC_CXXAPI_SHARED_MEMORY_H
#define MINDSPORE_CCSRC_CXXAPI_SHARED_MEMORY_H
#include <iostream>
#include "include/api/status.h"

namespace mindspore {
class SharedMemory {
 public:
  Status Create(uint64_t memory_size);
  Status Attach();
  void Detach();
  void Destroy();
  uint8_t *GetSharedMemoryAddr() { return shmat_addr_; }

 private:
  int shm_id_ = -1;
  uint8_t *shmat_addr_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXXAPI_SHARED_MEMORY_H
