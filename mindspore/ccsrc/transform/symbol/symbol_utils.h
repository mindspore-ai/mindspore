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
#include "utils/log_adapter.h"

template <typename Function, typename... Args>
auto RunAscendApi(Function f, const char *func_name, Args... args) {
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
  return f(args...);
}

template <typename Function>
auto RunAscendApi(Function f, const char *func_name) {
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
  return f();
}

namespace mindspore {
namespace transform {

#define CALL_ASCEND_API(func_name, ...) RunAscendApi(mindspore::transform::func_name##_, #func_name, __VA_ARGS__)
#define CALL_ASCEND_API2(func_name) RunAscendApi(mindspore::transform::func_name##_, #func_name)
void *GetLibHandler(const std::string &lib_path);
void LoadAscendApiSymbols();
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
