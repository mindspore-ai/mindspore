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

#include "minddata/dataset/plugin/shared_lib_util.h"
#ifdef __linux__
#include <dlfcn.h>
#endif
namespace mindspore {
namespace dataset {
#ifdef __linux__
void *SharedLibUtil::Load(const std::string &name) { return dlopen(name.c_str(), RTLD_LAZY); }
void *SharedLibUtil::FindSym(void *handle, const std::string &name) { return dlsym(handle, name.c_str()); }
int32_t SharedLibUtil::Close(void *handle) { return dlclose(handle); }
std::string SharedLibUtil::ErrMsg() {
  char *err_msg = dlerror();
  return err_msg != nullptr ? std::string(err_msg) : "dlerror() returned a nullptr";
}
#else  // MindData currently doesn't support loading shared library on platform that doesn't support dlopen
void *SharedLibUtil::Load(const std::string &name) { return nullptr; }
void *SharedLibUtil::FindSym(void *handle, const std::string &name) { return nullptr; }
int32_t SharedLibUtil::Close(void *handle) { return -1; }
std::string SharedLibUtil::ErrMsg() { return std::string("Plugin on non-Linux platform is not yet supported."); }
#endif
}  // namespace dataset
}  // namespace mindspore
