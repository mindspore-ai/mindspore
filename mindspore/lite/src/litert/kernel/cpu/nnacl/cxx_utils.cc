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
#include "src/litert/thread_cost_model.h"
#include "thread/threadpool.h"
#include "src/litert/inner_allocator.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "include/errorcode.h"

namespace mindspore::nnacl {
void *DefaultAllocatorMalloc(void *allocator, size_t sz) {
  if (allocator == nullptr) {
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

int DefaultUpdateThreadNumPass(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                               int64_t unit_num, int thread_num) {
#ifdef DYNAMIC_THREAD_DISTRIBUTE
  int update_thread = lite::UpdateThreadNum(kernel_type, per_unit_load_num, per_unit_store_num, unit_num, thread_num);
#else
  int update_thread = thread_num > 0 ? thread_num : 1;
#endif
  return update_thread;
}
}  // namespace mindspore::nnacl
