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

#include <cstddef>

#include "pybind_api/frontend/frontend_api.h"

#include "include/pynative/forward/pyboost/converter.h"

namespace mindspore {
namespace {
enum class CellToOverloadIndex {
  kToDeviceOverload,
  kToDtypeOverload,
  kToOtherOverload,
};

enum class CellToArgIndex {
  kFirstArgIndex,
  kSecondArgIndex,
  kThirdArgIndex,
};

constexpr size_t ToArgIndex(CellToArgIndex index) { return static_cast<size_t>(index); }
}  // namespace

static py::object CellParseToArgs(const py::args &args, const py::kwargs &kwargs) {
  static mindspore::pynative::PythonArgParser parser(
    {"ToDevice(str device=None, type dtype=None, bool non_blocking=False)",
     "ToDtype(type dtype=None, bool non_blocking=False)", "ToOther(tensor other, bool non_blocking=False)"},
    "Cell.to");

  auto parse_args = parser.Parse(args.ptr(), kwargs.ptr(), false);

  py::object device = py::none();
  py::object dtype = py::none();
  py::object non_blocking = py::bool_(false);

  switch (static_cast<CellToOverloadIndex>(parse_args.GetOvertLoadIndex())) {
    case CellToOverloadIndex::kToDeviceOverload:
      device = py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kFirstArgIndex)]);
      dtype = py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kSecondArgIndex)]);
      non_blocking =
        py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kThirdArgIndex)]);
      break;
    case CellToOverloadIndex::kToDtypeOverload:
      dtype = py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kFirstArgIndex)]);
      non_blocking =
        py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kSecondArgIndex)]);
      break;
    case CellToOverloadIndex::kToOtherOverload: {
      py::object other =
        py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kFirstArgIndex)]);
      device = other.attr("device");
      dtype = other.attr("dtype");
      non_blocking =
        py::reinterpret_borrow<py::object>(parse_args.arg_list_[ToArgIndex(CellToArgIndex::kSecondArgIndex)]);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Unexpected overload index from PythonArgParser.";
  }

  return py::make_tuple(device, dtype, non_blocking);
}

void RegCellUtilsApi(py::module *m) {
  (void)m->def("_cell_parse_to_args", &CellParseToArgs,
               "Parse Cell.to arguments into (device, dtype, non_blocking) tuple. "
               "Reuses PythonArgParser for consistent error messages with Tensor.to.");
}

}  // namespace mindspore
