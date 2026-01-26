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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_PY_INFER_CONVERT_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_PY_INFER_CONVERT_H_

#include "pybind11/pybind11.h"
#include "abstract/abstract_value.h"
#include "include/utils/visible.h"
#include "include/frontend/operator/primitive_py.h"

namespace py = pybind11;

namespace mindspore {
namespace abstract {
/// \brief Prepare Python inputs from abstract arguments for inference value functions
/// \param[in] args List of abstract base pointers needed to be converted.
/// \param[in] create_empty_tensor If true, create empty tensor when argument is a tensor with anyvalue (default: false)
/// \return Python tuple containing converted arguments, each converted to dictionary format via ConvertAbstractToPython
FRONTEND_EXPORT py::tuple PreparePyInputs(const AbstractBasePtrList &args, bool create_empty_tensor = false);
FRONTEND_EXPORT AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output);
}  // namespace abstract
}  // namespace mindspore
#endif  //  MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_PY_INFER_CONVERT_H_
