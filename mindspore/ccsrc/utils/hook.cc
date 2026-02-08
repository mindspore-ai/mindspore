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

#include "include/utils/hook.h"
#include <string>
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "pybind11/pytypes.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace {
py::object RunHook(uint64_t tensor_id, const py::function &hook, const py::object &arg) {
  if (hook.ptr() == nullptr) {
    MS_LOG(DEBUG) << "Hook for tensor id " << tensor_id << " have been deleted by python";
    return arg;
  }

  MS_LOG(DEBUG) << "Run hook for tensor id: " << tensor_id << ", hook: " << ConvertPyObjToString(hook);
  auto res = hook(arg);
  if (py::isinstance<py::none>(res)) {
    return arg;
  }

  if (MS_UNLIKELY(py::isinstance<py::tuple>(res) || py::isinstance<py::list>(res))) {
    MS_LOG(EXCEPTION) << "Tensor hook should be return None or a single value";
  }
  return res;
}
}  // namespace

TensorBackwardHook::TensorBackwardHook(uint64_t tensor_id, const py::function &obj)
    : tensor_id_(tensor_id), hook_(obj) {}

TensorBackwardHook::~TensorBackwardHook() { py::gil_scoped_acquire acquire_gil; }

ValuePtr TensorBackwardHook::operator()(const ValuePtr &grad) {
  py::gil_scoped_acquire acquire_gil;
  const auto py_arg = CValueToPybindObj(grad);
  const auto ret = RunHook(tensor_id_, hook_, py_arg);
  ValuePtr val;
  if (tensor::IsTensorPy(ret)) {
    val = tensor::ConvertToTensor(ret);
  } else {
    MS_LOG(EXCEPTION) << "Tensor hook should be return Tensor, but get type: "
                      << py::str(ret.get_type().attr("__name__")).cast<std::string>() << ".";
  }
  return val;
}
}  // namespace mindspore
