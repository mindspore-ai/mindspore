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
#include "src/runtime/thread_pool.h"

#endif

INTERNAL_API_DLL void LiteAPISetLastError(const char *msg);
INTERNAL_API_DLL void *LiteBackendAllocWorkspace(int deviceType, int deviceId, uint64_t size, int dtypeCode,
                                                 int dtypeBits);
INTERNAL_API_DLL int LiteBackendFreeWorkspace(int deviceType, int deviceId, void *ptr);
INTERNAL_API_DLL int LiteBackendRegisterSystemLibSymbol(const char *name, void *ptr);
#ifdef __cplusplus
}
#endif
#endif  // PREDICT_SRC_RUNTIME_RUNTIME_API_H_
