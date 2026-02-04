/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

// NOTICE: This header file should only be included once in the whole project.
// We change the cpp file to header file, to avoid MSVC compiler problem.
#ifndef MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_
#define MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_

#include "utils/log_adapter.h"
#include <string>
#include "include/utils/python_utils.h"

namespace mindspore {
class PyExceptionInitializer {
 public:
  PyExceptionInitializer();

  ~PyExceptionInitializer() = default;

 private:
  static void HandleExceptionPy(ExceptionType exception_type, const std::string &str);
  static void HandleExceptionRethrow(const std::function<void(void)> &main_func,
                                     const std::function<void(void)> &already_set_error_handler,
                                     const std::function<void(void)> &other_error_handler,
                                     const std::function<void(void)> &default_error_handler,
                                     const DebugInfoPtr &debug_info, bool force_rethrow);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_
