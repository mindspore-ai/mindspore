/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
#include <string>
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "utils/log_adapter.h"
#include "acl/acl.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#include "include/utils/callback.h"

template <typename Function, typename... Args>
auto RunAscendApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    if (ret != ACL_SUCCESS) {
      static auto fail_cb =
        GET_COMMON_CALLBACK(RunFailCallback, void, const char *, int, const char *, const std::string &, bool);
      if (fail_cb != nullptr) {
        fail_cb(file, line, call_f, func_name, true);
      }
    }
    return ret;
  } else {
    return f(args...);
  }
#else
  return f(args...);
#endif
}

template <typename Function>
auto RunAscendApi(Function f, const char *file, int line, const char *call_f, const char *func_name) {
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f)>, int>) {
    auto ret = f();
    if (ret != ACL_SUCCESS) {
      static auto fail_cb =
        GET_COMMON_CALLBACK(RunFailCallback, void, const char *, int, const char *, const std::string &, bool);
      if (fail_cb != nullptr) {
        fail_cb(file, line, call_f, func_name, true);
      }
    }
    return ret;
  } else {
    return f();
  }
#else
  return f();
#endif
}

template <typename Function>
bool HasAscendApi(Function f) {
  return f != nullptr;
}

namespace mindspore::device::ascend {

#define CALL_ASCEND_API(func_name, ...) \
  RunAscendApi(mindspore::device::ascend::func_name##_, FILE_NAME, __LINE__, __FUNCTION__, #func_name, ##__VA_ARGS__)

#define HAS_ASCEND_API(func_name) HasAscendApi(mindspore::device::ascend::func_name##_)

std::string GetAscendPath();
void *GetLibHandler(const std::string &lib_path, bool if_global = false);
void LoadAscendApiSymbols();
void LoadSimulationApiSymbols();
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
