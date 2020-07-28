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

#include <mutex>
#include <string>
#include "src/runtime/runtime_api.h"
#include "src/runtime/workspace_pool.h"
#include "src/runtime/thread_pool.h"
#include "utils/log_adapter.h"

static std::mutex gWorkspaceMutex;
#ifdef __cplusplus
extern "C" {
#endif
void LiteAPISetLastError(const char *msg) {
  MS_LOG(ERROR) << "The lite api set last error is " << msg;
}

void *LiteBackendAllocWorkspace(int deviceType,
                                int deviceId,
                                uint64_t size,
                                int dtypeCode,
                                int dtypeBits) {
  std::lock_guard<std::mutex> lock(gWorkspaceMutex);
  auto p = mindspore::predict::WorkspacePool::GetInstance();
  if (p == nullptr) {
    MS_LOG(ERROR) << "Get thread pool instance failed";
    return nullptr;
  }
  return p->AllocWorkSpaceMem(size);
}

int LiteBackendFreeWorkspace(int deviceType, int deviceId, void *ptr) {
  std::lock_guard<std::mutex> lock(gWorkspaceMutex);
  auto p = mindspore::predict::WorkspacePool::GetInstance();
  if (p == nullptr) {
    return -1;
  }
  p->FreeWorkSpaceMem(ptr);
  return 0;
}

void SetMaxWokerNum(int num) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOG(ERROR) << "Get thread pool instance failed";
    return;
  }
  if (num < 0) {
    LiteAPISetLastError("The number of work thread is less than 0");
    return;
  }
  p->ConfigMaxThreadNum(num);
}

void ConfigThreadPool(int mode, int nthreads) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOG(ERROR) << "Get thread pool instance failed";
    return;
  }
  p->ConfigThreadPool(mode, nthreads);
}

int LiteBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOG(ERROR) << "Get thread pool instance failed";
    return -1;
  }
  if (!p->LaunchWork(flambda, cdata, num_task)) {
    MS_LOG(ERROR) << "launch thread pool work failed";
    return -1;
  }
  return 0;
}

void DoAllThreadBind(bool ifBind, int mode) {
  auto p = mindspore::predict::ThreadPool::GetInstance();
  if (p == nullptr) {
    MS_LOG(ERROR) << "Get thread pool instance failed";
    return;
  }
  if (!p->BindAllThreads(ifBind, mode)) {
    MS_LOG(ERROR) << "do thread cpu bind failed";
  }
}

#ifdef __cplusplus
}
#endif

