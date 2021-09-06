/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_PY_H_
#define MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_PY_H_

#include <memory>

#include "pybind11/pybind11.h"
#include "utils/convert_utils_base.h"
#include "utils/any.h"
#include "base/base_ref.h"
#include "base/base.h"
#include "ir/anf.h"

namespace py = pybind11;

namespace mindspore {
py::object AnyToPyData(const Any &value);
py::object BaseRefToPyData(const BaseRef &value);
py::object ValueToPyData(const ValuePtr &value);

bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                       const std::shared_ptr<py::object> &ret_val);

AbstractBasePtr MakePyInferRes2Abstract(const py::object &shape_obj, const py::object &type_obj,
                                        const py::object &output = py::none());
void SetValueRange(const AbstractBasePtr &tensor, const py::object &output);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_PY_H_
