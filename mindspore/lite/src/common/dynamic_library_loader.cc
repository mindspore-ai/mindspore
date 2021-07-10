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

#include "src/common/dynamic_library_loader.h"
#include <string.h>
#include <climits>
#include "include/errorcode.h"
#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#define LOG_ERROR(content) \
  { printf("[ERROR] %s|%d|%s: " #content "\r\n", __FILE__, __LINE__, __func__); }

namespace mindspore {
namespace lite {
int DynamicLibraryLoader::Open(const char *lib_path) {
  if (handler_ != nullptr) {
    return RET_ERROR;
  }
  if ((strlen(lib_path)) >= PATH_MAX) {
    LOG_ERROR("path is too long");
    return RET_ERROR;
  }
  char resolved_path[PATH_MAX];

#ifndef _WIN32
  char *real_path = realpath(lib_path, resolved_path);
#else
  char *real_path = _fullpath(resolved_path, lib_path, 1024);
#endif

  if (real_path == nullptr) {
    LOG_ERROR("path not exist");
    return RET_ERROR;
  }

#ifndef _WIN32
  handler_ = dlopen(lib_path, RTLD_LAZY);
#else
  handler_ = LoadLibrary(lib_path);
#endif

  if (handler_ == nullptr) {
    LOG_ERROR("handler is nullptr.");
    return RET_ERROR;
  }
  return RET_OK;
}

void *DynamicLibraryLoader::GetFunc(const char *func_name) {
#ifndef _WIN32
  return dlsym(handler_, func_name);
#else
  auto func = GetProcAddress(reinterpret_cast<HINSTANCE__ *>(handler_), func_name);
  return reinterpret_cast<void *>(func);
#endif
}

int DynamicLibraryLoader::Close() {
  if (handler_ == nullptr) {
    return RET_OK;
  }
#ifndef _WIN32
  auto close_res = dlclose(handler_);
  if (close_res != 0) {
    LOG_ERROR("can not close handler");
    return RET_ERROR;
  }
#else
  auto close_res = FreeLibrary(reinterpret_cast<HINSTANCE__ *>(handler_));
  if (close_res == 0) {
    LOG_ERROR("can not close handler");
    return RET_ERROR;
  }
#endif
  handler_ = nullptr;
  return RET_OK;
}

DynamicLibraryLoader::~DynamicLibraryLoader() {
  if (handler_ != nullptr) {
    Close();
  }
}

}  // namespace lite
}  // namespace mindspore
