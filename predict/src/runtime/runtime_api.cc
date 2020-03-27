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

#include "src/runtime/runtime_api.h"
#include <mutex>
#include <string>
#include "src/runtime/workspace_pool.h"
#include "src/runtime/thread_pool.h"
#include "common/mslog.h"

static std::mutex gWorkspaceMutex;
#ifdef __cplusplus
extern "C" {
#endif
void LiteAPISetLastError(const char *msg) { MS_LOGE("The lite api set last error is %s.", msg); }

void *LiteBackendAllocWorkspace(int deviceType, int deviceId, uint64_t size, int dtypeCode, int dtypeBits) {
  std::lock_guard<std::mutex> lock(gWorkspaceMutex);
  auto p = mindspore::predict::WorkspacePool::GetInstance();
  if (p == nullptr) {
    MS_LOGE("get ThreadPool install failed");
    return nullptr;
  }
  return p->AllocWorkSpaceMem(size);
}

int LiteBackendFreeWorkspace(int deviceType, int deviceId, void *ptr) {
  std::lock_guard<std::mutex> lock(gWorkspaceMutex);
  auto p = mindspore::predict::WorkspacePool::GetInstance();
  if (p == nullptr) {
    MS_LOGE("get ThreadPool install failed");
    return -1;
  }
  p->FreeWorkSpaceMem(ptr);
  return 0;
}

void ConfigThreadPool(int mode, int nthreads) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOGE("get ThreadPool install failed");
    return;
  }
  p->ConfigThreadPool(mode, nthreads);
}

int LiteBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOGE("get ThreadPool install failed");
    return -1;
  }
  if (!p->LaunchThreadPoolTask()) {
    MS_LOGE("get ThreadPool or thread bind failed");
    return -1;
  }
  if (!p->AddTask(flambda, cdata, num_task)) {
    MS_LOGE("AddTask failed");
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
}
#endif
