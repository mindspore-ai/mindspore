/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/common/utils/hook.h"
#include <string>
#include "include/common/utils/convert_utils_py.h"
#include "pybind11/pytypes.h"

namespace mindspore {
namespace {
py::tuple GetPythonArg(const ValuePtr &grad) {
  // Get _c_expression tensor
  py::tuple py_args(kIndex1);
  py_args[0] = ValueToPyData(grad);
  // Get python tensor
  py::tuple converted_args(kIndex1);
  ConvertCTensorToPyTensor(py_args, &converted_args);
  return converted_args;
}

ValuePtrList GetCValue(const py::object &output) {
  // Convert pyobject output to c++ tensor.
  ValuePtrList output_tensors;
  ConvertPyObjectToTensor(output, &output_tensors);
  return output_tensors;
}

void RunHook(std::map<uint64_t, py::function> *hook_map, py::tuple *arg) {
  MS_EXCEPTION_IF_NULL(hook_map);
  MS_EXCEPTION_IF_NULL(arg);
  for (auto it = hook_map->begin(); it != hook_map->end();) {
    if (it->second.ptr() == nullptr) {
      MS_LOG(DEBUG) << "Hook id " << it->first << " have been delete by python";
      hook_map->erase(it++);
    } else {
      MS_LOG(DEBUG) << "Run hook id " << it->first << " and its value " << ConvertPyObjToString(it->second);
      // Flatten input
      auto res = (it->second)(*(*arg));
      if (py::isinstance<py::none>(res)) {
        MS_EXCEPTION(ValueError) << "Get None result for hook call";
      }
      if (MS_UNLIKELY(py::isinstance<py::tuple>(res) || py::isinstance<py::list>(res))) {
        auto tuple = py::cast<py::tuple>(res);
        if (tuple.size() != arg->size()) {
          MS_LOG(EXCEPTION) << "Hook input size " << arg->size() << " is not equal to hook output size "
                            << tuple.size();
        }
        *arg = res;
      } else {
        // Default
        if (arg->size() != kIndex1) {
          MS_LOG(EXCEPTION) << "Hook output size " << arg->size() << "is not equal to default input size 1";
        }
        (*arg)[kIndex0] = res;
      }
      ++it;
    }
  }
}
}  // namespace

TensorBackwardHook::TensorBackwardHook(uint64_t tensor_id, const py::function &obj) {
  (void)hook_map_.emplace(tensor_id, obj);
}

TensorBackwardHook::~TensorBackwardHook() {
  py::gil_scoped_acquire acquire_gil;
  hook_map_.clear();
}

ValuePtr TensorBackwardHook::operator()(const ValuePtr &grad) {
  py::gil_scoped_acquire acquire_gil;
  auto py_args = GetPythonArg(grad);
  RunHook(&hook_map_, &py_args);
  auto c_output_values = GetCValue(py_args);
  return c_output_values[kIndex0];
}
}  // namespace mindspore
