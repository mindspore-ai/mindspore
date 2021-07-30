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

#ifndef MINDSPORE_CCSRC_UTILS_DLOPEN_MACRO_H
#define MINDSPORE_CCSRC_UTILS_DLOPEN_MACRO_H
#include <dlfcn.h>
#include <string>
#include <functional>
#include "utils/log_adapter.h"

#define PLUGIN_METHOD(name, return_type, params...)                        \
  extern "C" {                                                             \
  __attribute__((visibility("default"))) return_type Plugin##name(params); \
  }                                                                        \
  constexpr const char *k##name##Name = "Plugin" #name;                    \
  using name##FunObj = std::function<return_type(params)>;                 \
  using name##FunPtr = return_type (*)(params);

#define ORIGIN_METHOD(name, return_type, params...)        \
  extern "C" {                                             \
  return_type name(params);                                \
  }                                                        \
  constexpr const char *k##name##Name = #name;             \
  using name##FunObj = std::function<return_type(params)>; \
  using name##FunPtr = return_type (*)(params);

inline static std::string GetDlErrorMsg() {
  const char *result = dlerror();
  return (result == nullptr) ? "Unknown" : result;
}

template <class T>
static T DlsymWithCast(void *handle, const char *symbol_name) {
  T symbol = reinterpret_cast<T>(dlsym(handle, symbol_name));
  if (symbol == nullptr) {
    MS_LOG(EXCEPTION) << "Dlsym symbol " << symbol_name << " failed, result = " << GetDlErrorMsg();
  }
  return symbol;
}

#define DlsymFuncObj(func_name, plugin_handle) DlsymWithCast<func_name##FunPtr>(plugin_handle, k##func_name##Name);
#endif  // MINDSPORE_CCSRC_UTILS_DLOPEN_MACRO_H
