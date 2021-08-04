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
#include <climits>
#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#undef ERROR
#undef SM_DEBUG
#endif
#include "include/errorcode.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
int DynamicLibraryLoader::Open(const std::string &lib_path) {
  if (handler_ != nullptr) {
    return RET_ERROR;
  }
  std::string real_path = RealPath(lib_path.c_str());

#ifndef _WIN32
  handler_ = dlopen(real_path.c_str(), RTLD_LAZY);
#else
  handler_ = LoadLibrary(real_path.c_str());
#endif
  if (handler_ == nullptr) {
    MS_LOG(ERROR) << "handler is nullptr.";
    return RET_ERROR;
  }
  return RET_OK;
}

void *DynamicLibraryLoader::GetFunc(const std::string &func_name) {
#ifndef _WIN32
  return dlsym(handler_, func_name.c_str());
#else
  auto func = GetProcAddress(reinterpret_cast<HINSTANCE__ *>(handler_), func_name.c_str());
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
    MS_LOG(ERROR) << "can not close handler";
    return RET_ERROR;
  }
#else
  auto close_res = FreeLibrary(reinterpret_cast<HINSTANCE__ *>(handler_));
  if (close_res == 0) {
    MS_LOG(ERROR) << "can not close handler";
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
