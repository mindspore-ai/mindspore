/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "thread/core_affinity.h"
#include "thread/threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

static mindspore::ThreadPool *g_pool = nullptr;

int CreateThreadPool(int thread_num) {
  g_pool = mindspore::ThreadPool::CreateThreadPool(thread_num);
  if (g_pool == nullptr) {
    return mindspore::THREAD_ERROR;
  }
  return mindspore::THREAD_OK;
}

int SetCoreAffinity(int bind_mode) {
  if (g_pool == nullptr) {
    return mindspore::THREAD_ERROR;
  }
  return g_pool->SetCpuAffinity(static_cast<mindspore::BindMode>(bind_mode));
}

int GetCurrentThreadNum() {
  if (g_pool == nullptr) {
    return 0;
  }
  return g_pool->thread_num();
}

int ParallelLaunch(int (*func)(void *, int, float, float), void *content, int task_num) {
  if (g_pool == nullptr) {
    return mindspore::THREAD_ERROR;
  }
  return g_pool->ParallelLaunch(func, content, task_num);
}

void ClearThreadPool() {
  delete g_pool;
  g_pool = nullptr;
}

#ifdef __cplusplus
}
#endif
