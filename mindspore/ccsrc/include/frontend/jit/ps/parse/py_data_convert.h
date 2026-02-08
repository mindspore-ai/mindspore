/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PARSE_PY_DATA_CONVERT_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PARSE_PY_DATA_CONVERT_H_

#include "pybind11/pybind11.h"
#include "ir/dtype/type.h"
#include "include/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace parse {
// Convert python object to ValuePtr.
FRONTEND_EXPORT bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature = false,
                                 const TypePtr &dtype = nullptr, bool forbid_reuse = false);

namespace data_converter {
FRONTEND_EXPORT ValuePtr PyDataToValue(const py::object &obj);
FRONTEND_EXPORT ValuePtr PyObjToValue(const py::object &obj, bool stub = false);
FRONTEND_EXPORT void ClearObjectCache();
}  // namespace data_converter
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PARSE_PY_DATA_CONVERT_H_
