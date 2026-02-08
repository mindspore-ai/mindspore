/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_OPS_UTILS_DL_UTILS_H
#define MINDSPORE_OPS_OPS_UTILS_DL_UTILS_H

#include <string>

#if defined(_WIN32)
#include <windows.h>
#define DL_OPEN(path)                                                                                   \
  [](const std::string &p) -> void * {                                                                  \
    SetLastError(0);                                                                                    \
    return reinterpret_cast<void *>(LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)); \
  }(path)

#define DL_SYM(handle, name)                                                     \
  [](void *h, const char *n) -> void * {                                         \
    SetLastError(0);                                                             \
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(h), n)); \
  }(handle, name)

#define DL_CLOSE(handle) FreeLibrary((HMODULE)handle)

#define DL_ERROR()                                  \
  []() -> const char * {                            \
    static std::string errMsg;                      \
    DWORD errCode = GetLastError();                 \
    if (errCode == 0) return nullptr;               \
    errMsg = "WinError " + std::to_string(errCode); \
    return errMsg.c_str();                          \
  }()
#elif !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#define DL_OPEN(path) dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)
#define DL_SYM(handle, name) dlsym(handle, name)
#define DL_CLOSE(handle) dlclose(handle)
#define DL_ERROR() dlerror()
#endif

#endif  // MINDSPORE_OPS_OPS_UTILS_DL_UTILS_H
