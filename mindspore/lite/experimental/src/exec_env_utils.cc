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

#include "experimental/src/exec_env_utils.h"

namespace mindspore::lite::experimental {
void *DefaultAllocatorMalloc(void *allocator, size_t sz) {
  if (allocator == nullptr || sz == 0) {
    MS_LOG(ERROR) << "in param invalid";
    return nullptr;
  }
  auto default_allocator = static_cast<mindspore::DefaultAllocator *>(allocator);
  return default_allocator->Malloc(sz);
}

void DefaultAllocatorFree(void *allocator, void *ptr) {
  if (allocator == nullptr || ptr == nullptr) {
    MS_LOG(ERROR) << "in param invalid";
    return;
  }
  auto default_allocator = static_cast<mindspore::DefaultAllocator *>(allocator);
  return default_allocator->Free(ptr);
}

int DefaultThreadPoolParallelLunch(void *threadPool, void *task, void *param, int taskNr) {
  using TaskFunc = int (*)(void *param, int task_id, float l, float r);
  TaskFunc task_func = (TaskFunc)task;

  ThreadPool *pool = static_cast<ThreadPool *>(threadPool);
  if (pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return RET_NULL_PTR;
  }
  return pool->ParallelLaunch(task_func, param, taskNr);
}
}  // namespace mindspore::lite::experimental
