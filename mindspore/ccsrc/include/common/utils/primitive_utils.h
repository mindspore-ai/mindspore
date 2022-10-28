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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMITIVE_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMITIVE_UTILS_H_

#include <string>
#include "pybind11/pybind11.h"
#include "base/base_ref.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"

namespace py = pybind11;

namespace mindspore {
COMMON_EXPORT py::function GetBpropFunctionByObj(const py::object &obj, bool get_closure = false);

COMMON_EXPORT py::function GetBpropFunction(const std::string &name);

COMMON_EXPORT py::function GetTaylorRuleFunctionByObj(const py::object &obj);

COMMON_EXPORT py::function GetTaylorRuleFunction(const std::string &name);

COMMON_EXPORT py::function GetComputeFunction(const std::string &name);

COMMON_EXPORT BaseRef RunComputeFunction(const PrimitivePtr &prim, const VectorRef &args);

COMMON_EXPORT py::function GetComputeFunctionWithoutPyObj(const std::string &name);

COMMON_EXPORT BaseRef RunComputeFunctionWithoutPyObj(const PrimitivePtr &prim, const VectorRef &args);

COMMON_EXPORT py::tuple ConvertDatatoPyTuple(const VectorRef &args);

COMMON_EXPORT py::function GetVmapRuleFunctionByObj(const py::object &obj, int axis_size);

COMMON_EXPORT py::function GetVmapRuleFunction(const std::string &name, int axis_size);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMITIVE_UTILS_H_
