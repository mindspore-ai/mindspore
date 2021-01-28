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
#include "minddata/dataset/util/memory_pool.h"
#include "./securec.h"

namespace mindspore {
namespace dataset {
Status DeMalloc(std::size_t s, void **p, bool init_to_zero = false) {
  if (p == nullptr) {
    RETURN_STATUS_UNEXPECTED("p is null");
  }
  void *q = ::malloc(s);
  if (q == nullptr) {
    return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  } else {
    *p = q;
    if (init_to_zero) {
      (void)memset_s(q, s, 0, s);
    }
    return Status::OK();
  }
}
}  // namespace dataset
}  // namespace mindspore

void *operator new(std::size_t s, mindspore::Status *rc, std::shared_ptr<mindspore::dataset::MemoryPool> b) {
  void *ptr = nullptr;
  *rc = b->Allocate(s, &ptr);
  return ptr;
}

void *operator new[](std::size_t s, mindspore::Status *rc, std::shared_ptr<mindspore::dataset::MemoryPool> b) {
  void *ptr = nullptr;
  *rc = b->Allocate(s, &ptr);
  return ptr;
}

void operator delete(void *p, std::shared_ptr<mindspore::dataset::MemoryPool> b) {
  if (p != nullptr) b->Deallocate(p);
}

void operator delete[](void *p, std::shared_ptr<mindspore::dataset::MemoryPool> b) {
  if (p != nullptr) b->Deallocate(p);
}
