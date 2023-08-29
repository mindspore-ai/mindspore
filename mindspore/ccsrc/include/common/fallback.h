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

#include "include/common/visible.h"
#include "pybind11/pybind11.h"
#include "ir/value.h"
#include "abstract/abstract_value.h"
namespace py = pybind11;

namespace mindspore {
namespace fallback {
COMMON_EXPORT bool HasPyExecuteOutput();
COMMON_EXPORT py::object PopPyExecuteOutput();
COMMON_EXPORT void PushPyExecuteOutput(const py::object &output);
COMMON_EXPORT int GetJitSyntaxLevel();
COMMON_EXPORT bool CheckListValid(const py::list &obj, bool to_raw_memory);
COMMON_EXPORT bool CheckSequenceToMemory(const py::sequence &obj);
COMMON_EXPORT abstract::AbstractSequencePtr GenerateAbstractSequence(const BaseShapePtr &base_shape,
                                                                     const TypePtr &type, bool is_dyn_shape);
}  // namespace fallback
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_FALLBACK_H_
