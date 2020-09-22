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

#ifndef MINDSPORE_LITE_INTERNAL_SRC_ALLOCATOR_H_
#define MINDSPORE_LITE_INTERNAL_SRC_ALLOCATOR_H_

#include <stddef.h>
#include <pthread.h>
#include "internal/include/string.h"

namespace mindspore::lite {
constexpr int kBlockRange = 9;

typedef struct AllocatorContext {
  bool lock_flag_;
} AllocatorContext;

typedef struct MemNode {
  MemNode *pre_ = nullptr;
  MemNode *next_ = nullptr;
  size_t size_ = 0;
} MemNode;


class Allocator {
 public:
  Allocator();
  ~Allocator();
  void SetContext(const AllocatorContext &ctx);
  void *Malloc(size_t size);
  void Free(void *ptr);
  void Clear();
  size_t GetTotalSize();

 private:
  void Lock();
  void UnLock();

  bool lock_flag_ = false;
  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
  MemNode *large_mem_list_ = nullptr;
  MemNode *allocated_list_[kBlockRange];
  MemNode *free_list_[kBlockRange];
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_INTERNAL_SRC_ALLOCATOR_H_
