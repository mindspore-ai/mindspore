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

#include "pybind_api/ir/cell_py.h"
#include <string>

#include "include/common/pybind_api/api_register.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/data_converter.h"

namespace mindspore {
void CellPy::AddAttr(CellPtr cell, const std::string &name, const py::object &obj) {
  ValuePtr converted_ret = nullptr;
  MS_EXCEPTION_IF_NULL(cell);
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "Call '_add_attr' to add attribute to Cell failed,"
                      << " not support py::module to be attribute value; Cell name: " << cell->name()
                      << ", attribute name: " << name << ", attribute value: " << py::str(obj) << "'.";
  }
  bool converted = parse::ConvertData(obj, &converted_ret, true);
  if (!converted) {
    MS_LOG(DEBUG) << "Attribute convert error with type: " << std::string(py::str(obj));
  } else {
    MS_LOG(DEBUG) << cell->ToString() << " add attr " << name << converted_ret->ToString();
    cell->AddAttr(name, converted_ret);
  }
}
// Define python 'Cell' class.
void RegCell(const py::module *m) {
  (void)py::enum_<MixedPrecisionType>(*m, "MixedPrecisionType", py::arithmetic())
    .value("NOTSET", MixedPrecisionType::kNotSet)
    .value("FP16", MixedPrecisionType::kFP16)
    .value("FP32", MixedPrecisionType::kFP32);
  (void)py::class_<Cell, std::shared_ptr<Cell>>(*m, "Cell_")
    .def(py::init<std::string &>())
    .def("__str__", &Cell::ToString)
    .def("_add_attr", &CellPy::AddAttr, "Add Cell attr.")
    .def("_del_attr", &Cell::DelAttr, "Delete Cell attr.")
    .def("set_mixed_precision_type", &Cell::SetMixedPrecisionType, "Set mixed precision type.")
    .def("get_mixed_precision_type", &Cell::GetMixedPrecisionType, "Get mixed precision type.")
    .def(
      "construct", []() { MS_LOG(EXCEPTION) << "we should define `construct` for all `cell`."; },
      "construct")
    .def(py::pickle(
      [](const Cell &cell) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::str(cell.name()));
      },
      [](const py::tuple &tup) {  // __setstate__
        if (tup.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        Cell data(tup[0].cast<std::string>());
        return data;
      }));
}
}  // namespace mindspore
