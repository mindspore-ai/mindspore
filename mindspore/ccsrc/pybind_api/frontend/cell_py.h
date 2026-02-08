/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_CELL_PY_H_
#define MINDSPORE_CCSRC_PYBIND_API_IR_CELL_PY_H_

#include <memory>
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "ir/cell.h"

namespace py = pybind11;
// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of Mindsporeession project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// Cell python wrapper and adapter class.
class CellPy {
 public:
  static void AddAttr(CellPtr cell, const std::string &name, const py::object &obj);
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_CELL_PY_H_
