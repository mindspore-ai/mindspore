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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_API_H_
#define MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_API_H_
#include <memory>

#ifndef INTERNAL_API_DLL
#ifdef _WIN32
#ifdef LITE_EXPORTS
#define INTERNAL_API_DLL __declspec(dllexport)
#else
#define INTERNAL_API_DLL __declspec(dllimport)
#endif
#else
#define INTERNAL_API_DLL __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *sync_handle;
  int32_t num_task;
} LiteParallelGroupEnv;
typedef int (*FTVMParallelLambda)(int task_id, LiteParallelGroupEnv *penv, void *cdata);
INTERNAL_API_DLL void LiteAPISetLastError(const char *msg);
INTERNAL_API_DLL void *LiteBackendAllocWorkspace(int deviceType, int deviceId, uint64_t size, int dtypeCode,
                                                 int dtypeBits);
INTERNAL_API_DLL int LiteBackendFreeWorkspace(int deviceType, int deviceId, void *ptr);
INTERNAL_API_DLL void SetMaxWokerNum(int num);
INTERNAL_API_DLL void ConfigThreadPool(int mode, int nthreads);
INTERNAL_API_DLL inline void CfgThreadPool(int nthread) { ConfigThreadPool(-1, nthread); }
INTERNAL_API_DLL int LiteBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task);
INTERNAL_API_DLL int LiteBackendRegisterSystemLibSymbol(const char *name, void *ptr);
INTERNAL_API_DLL void DoAllThreadBind(bool ifBind, int mode);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_API_H_

