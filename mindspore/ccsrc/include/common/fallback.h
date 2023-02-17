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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_FALLBACK_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_FALLBACK_H_
#include <string>

#include "include/common/visible.h"
#include "pybind11/pybind11.h"
#include "ir/value.h"
namespace py = pybind11;

namespace mindspore {
COMMON_EXPORT bool HasPyExecuteOutput();
COMMON_EXPORT py::object PopPyExecuteOutput();
COMMON_EXPORT void PushPyExecuteOutput(const py::object &output);
// To be removed map function when PyExecute ops can be auto monad
COMMON_EXPORT bool HasPyExecuteOutput(const ValuePtr &key);
COMMON_EXPORT void PushPyExecuteOutput(const ValuePtr &key, const py::object &output);
COMMON_EXPORT py::object PopPyExecuteOutput(const ValuePtr &key);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_FALLBACK_H_
