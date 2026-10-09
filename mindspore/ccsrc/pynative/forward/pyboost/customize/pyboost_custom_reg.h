/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_CUSTOMIZE_PYBOOST_CUSTOM_REG_H_
#define MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_CUSTOMIZE_PYBOOST_CUSTOM_REG_H_

#include <string>
#include "pybind11/pybind11.h"
#include "ir/primitive.h"
#include "include/utils/visible.h"

namespace mindspore::pynative {
namespace py = pybind11;

py::object PyboostCustomExtBase(const PrimitivePtr &prim, const py::list &args);
py::object Pyboost_CustomExt(const py::args &args);
py::object Pyboost_CellBackwardHook(const py::args &args);
py::object PyboostDetach(const py::object &input);
py::tuple ValidateAllToAllSingleAndBuildCommArgs(const py::object &output, const py::object &input,
                                                 const py::object &output_split_sizes,
                                                 const py::object &input_split_sizes, const py::object &group,
                                                 const py::object &async_op);

PYNATIVE_EXPORT void RegPyBoostCustomizeModule(py::module *m);
}  // namespace mindspore::pynative

#endif  // MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_CUSTOMIZE_PYBOOST_CUSTOM_REG_H_
