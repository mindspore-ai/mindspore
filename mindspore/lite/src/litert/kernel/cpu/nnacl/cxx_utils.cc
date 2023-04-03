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

#include "nnacl/cxx_utils.h"
#include "src/litert/pack_weight_manager.h"
#include "thread/threadpool.h"
#include "src/litert/inner_allocator.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "include/errorcode.h"

namespace mindspore::nnacl {
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
    return lite::RET_NULL_PTR;
  }
  return pool->ParallelLaunch(task_func, param, taskNr);
}

void *DefaultGetSharingPackData(void *manager, const void *tensor_data, const size_t size, bool *is_packed) {
  if (manager == nullptr) {
    MS_LOG(ERROR) << "in param invalid";
    return nullptr;
  }
  auto weight_manager = static_cast<mindspore::lite::PackWeightManager *>(manager);
  return weight_manager->GetPackData(tensor_data, size, is_packed);
}

void DefaultFreeSharingPackData(void *manager, void *tensor_data) {
  if (manager == nullptr) {
    MS_LOG(ERROR) << "in param invalid";
    return;
  }
  auto weight_manager = static_cast<mindspore::lite::PackWeightManager *>(manager);
  return weight_manager->Free(tensor_data);
}
}  // namespace mindspore::nnacl
