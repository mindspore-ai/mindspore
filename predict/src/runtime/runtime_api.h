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
#ifndef PREDICT_SRC_RUNTIME_RUNTIME_API_H_
#define PREDICT_SRC_RUNTIME_RUNTIME_API_H_
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *sync_handle;
  int32_t num_task;
} TVMParallelGroupEnv;
typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv *penv, void *cdata);
void LiteAPISetLastError(const char *msg);
void *LiteBackendAllocWorkspace(int deviceType, int deviceId, uint64_t size, int dtypeCode, int dtypeBits);
int LiteBackendFreeWorkspace(int deviceType, int deviceId, void *ptr);
void ConfigThreadPool(int mode, int nthreads);
int LiteBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task);
int LiteBackendRegisterSystemLibSymbol(const char *name, void *ptr);

#ifdef __cplusplus
}
#endif
#endif  // PREDICT_SRC_RUNTIME_RUNTIME_API_H_
