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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_OPERATOR_PRIMITIVE_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_OPERATOR_PRIMITIVE_UTILS_H_

#include <set>
#include <string>
#include <vector>
#include "ops/op_def.h"
#include "pybind11/pybind11.h"
#include "base/base_ref.h"
#include "include/utils/utils.h"

namespace py = pybind11;

namespace mindspore {
COMMON_EXPORT py::function GetBpropFunctionByObj(const py::object &obj, bool get_closure = false);

COMMON_EXPORT py::function GetBpropFunction(const std::string &name);

COMMON_EXPORT py::function GetComputeFunction(const std::string &name);

COMMON_EXPORT BaseRef RunComputeFunction(const PrimitivePtr &prim, const VectorRef &args);

COMMON_EXPORT py::function GetComputeFunctionWithoutPyObj(const std::string &name);

COMMON_EXPORT BaseRef RunComputeFunctionWithoutPyObj(const PrimitivePtr &prim, const VectorRef &args);

COMMON_EXPORT py::tuple ConvertDatatoPyTuple(const VectorRef &args);

namespace prim {
COMMON_EXPORT std::map<std::string, std::vector<std::string>> GetFunctionalSignatureMap(bool is_method);

COMMON_EXPORT std::string ErrorMessageForConvertRefDtype(const ValuePtr &func, const std::string &ref_type,
                                                         const std::string &target_type, size_t index);

COMMON_EXPORT std::stringstream BuildApiInputInfo(const std::string &function_name,
                                                  const std::vector<std::string> &arg_info_list);

COMMON_EXPORT std::string BuildFunctionalErrorMsg(const std::string &function_name,
                                                  const std::vector<std::string> &arg_info_list, bool is_method);

COMMON_EXPORT std::string OpDTypeToString(ops::OP_DTYPE dtype);

COMMON_EXPORT std::string BuildOpErrorMsg(const ops::OpDefPtr &op_def, const std::vector<std::string> &op_type_list);

COMMON_EXPORT std::string BuildOpInputsErrorMsg(const ops::OpDefPtr &op_def, const std::string &arg_name,
                                                const TypePtr &arg_type);
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_OPERATOR_PRIMITIVE_UTILS_H_
