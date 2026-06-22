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

#include "pybind_api/frontend/frontend_api.h"

#include "include/pynative/forward/pyboost/converter.h"

namespace mindspore {
namespace {
enum CellToOverloadIndex {
  kToDeviceOverload,
  kToDtypeOverload,
  kToOtherOverload,
};

enum CellToArgIndex {
  kFirstArgIndex,
  kSecondArgIndex,
  kThirdArgIndex,
};
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

  switch (parse_args.GetOvertLoadIndex()) {
    case kToDeviceOverload:
      device = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kFirstArgIndex]);
      dtype = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kSecondArgIndex]);
      non_blocking = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kThirdArgIndex]);
      break;
    case kToDtypeOverload:
      dtype = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kFirstArgIndex]);
      non_blocking = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kSecondArgIndex]);
      break;
    case kToOtherOverload: {
      py::object other = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kFirstArgIndex]);
      device = other.attr("device");
      dtype = other.attr("dtype");
      non_blocking = py::reinterpret_borrow<py::object>(parse_args.arg_list_[kSecondArgIndex]);
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
