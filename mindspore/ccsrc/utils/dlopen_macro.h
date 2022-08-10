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

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#undef ERROR
#undef SM_DEBUG
#undef Yield
#endif
#include <string>
#include <functional>
#include "utils/log_adapter.h"

#ifndef _WIN32
#define PORTABLE_EXPORT __attribute__((visibility("default")))
#else
#define PORTABLE_EXPORT __declspec(dllexport)
#endif

#define PLUGIN_METHOD(name, return_type, ...)                   \
  extern "C" {                                                  \
  PORTABLE_EXPORT return_type Plugin##name(__VA_ARGS__);        \
  }                                                             \
  constexpr const char *k##name##Name = "Plugin" #name;         \
  using name##FunObj = std::function<return_type(__VA_ARGS__)>; \
  using name##FunPtr = return_type (*)(__VA_ARGS__);

#define ORIGIN_METHOD(name, return_type, ...)                   \
  extern "C" {                                                  \
  return_type name(__VA_ARGS__);                                \
  }                                                             \
  constexpr const char *k##name##Name = #name;                  \
  using name##FunObj = std::function<return_type(__VA_ARGS__)>; \
  using name##FunPtr = return_type (*)(__VA_ARGS__);

inline static std::string GetDlErrorMsg() {
#ifndef _WIN32
  const char *result = dlerror();
  return (result == nullptr) ? "Unknown" : result;
#else
  return std::to_string(GetLastError());
#endif
}

template <class T>
static T DlsymWithCast(void *handle, const char *symbol_name) {
#ifndef _WIN32
  T symbol = reinterpret_cast<T>(reinterpret_cast<intptr_t>(dlsym(handle, symbol_name)));
#else
  T symbol = reinterpret_cast<T>(GetProcAddress(reinterpret_cast<HINSTANCE__ *>(handle), symbol_name));
#endif
  if (symbol == nullptr) {
    MS_LOG(EXCEPTION) << "Dynamically load symbol " << symbol_name << " failed, result = " << GetDlErrorMsg();
  }
  return symbol;
}

#define DlsymFuncObj(func_name, plugin_handle) DlsymWithCast<func_name##FunPtr>(plugin_handle, k##func_name##Name);
#endif  // MINDSPORE_CCSRC_UTILS_DLOPEN_MACRO_H
