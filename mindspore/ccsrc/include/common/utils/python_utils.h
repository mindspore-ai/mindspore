/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_

#include <functional>

#include "include/common/visible.h"
#include "utils/trace_base.h"

namespace mindspore {
COMMON_EXPORT void HandleExceptionRethrow(const std::function<void(void)> &main_func,
                                          const std::function<void(void)> &already_set_error_handler,
                                          const std::function<void(void)> &other_error_handler,
                                          const std::function<void(void)> &default_error_handler,
                                          const DebugInfoPtr &debug_info = nullptr);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_
