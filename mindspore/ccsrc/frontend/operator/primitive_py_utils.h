/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_PRIMITIVE_PY_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_PRIMITIVE_PY_UTILS_H_

#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "base/base_ref.h"
#include "include/utils/utils.h"

namespace py = pybind11;

namespace mindspore {
namespace prim {
py::function GetTaylorRuleFunctionByObj(const py::object &obj);

py::function GetTaylorRuleFunction(const std::string &name);

py::function GetVmapRuleFunctionByObj(const py::object &obj, int axis_size);

py::function GetVmapRuleFunction(const std::string &name, int axis_size);
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_PRIMITIVE_PY_UTILS_H_
