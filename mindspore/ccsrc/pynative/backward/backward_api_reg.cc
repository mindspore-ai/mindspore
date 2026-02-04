/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/utils/tensor_py.h"
#include "mindspore/ccsrc/pynative/backward/op_grad/func_grad.h"
#include "mindspore/ccsrc/pybind_api/pynative/pynative_api.h"

namespace mindspore {
namespace pynative {
namespace autograd {
std::vector<ValuePtr> ConvertPyTupleToTensorList(const py::tuple &tuple_args) {
  std::vector<ValuePtr> tensors;
  tensors.reserve(tuple_args.size());
  for (size_t i = 0; i < tuple_args.size(); ++i) {
    py::object obj = tuple_args[i];
    if (!tensor::IsTensorPy(obj)) {
      MS_LOG(EXCEPTION) << "Elements of tuple should be tensor, but got" << py::str(obj);
    }
    auto tensor = tensor::ConvertToTensor(obj);
    (void)tensors.emplace_back(tensor);
  }
  return tensors;
}

py::object RunBackward(const py::object &tensors, const py::object &grad_tensors, bool keep_graph, bool high_order,
                       const py::object &inputs, bool allow_unreachable, bool accumulate_grad) {
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();
  if (!py::isinstance<py::tuple>(tensors)) {
    MS_LOG(EXCEPTION) << "Output tensors should be tuple! but got " << py::str(tensors);
  }
  ValuePtr output = std::make_shared<ValueTuple>(ConvertPyTupleToTensorList(tensors));
  ValuePtr sens_gradients = nullptr;
  if (py::isinstance<py::tuple>(grad_tensors)) {
    sens_gradients = std::make_shared<ValueTuple>(ConvertPyTupleToTensorList(grad_tensors));
  }
  if (!py::isinstance<py::tuple>(inputs) && !py::isinstance<py::none>(inputs)) {
    MS_LOG(EXCEPTION) << "input tensors should be tuple or none! but got " << py::str(inputs);
  }
  ValuePtrList input_tensors;
  if (py::isinstance<py::tuple>(inputs)) {
    auto tuple_inputs = py::cast<py::tuple>(inputs);
    if (!accumulate_grad && tuple_inputs.empty()) {
      MS_LOG(EXCEPTION) << "Please set inputs for grad interface, grad requires non-empty inputs.";
    }
    input_tensors = ConvertPyTupleToTensorList(tuple_inputs);
    if (accumulate_grad) {
      for (const auto &input_tensor : input_tensors) {
        auto tensor = input_tensor->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        tensor->retain_grad();
      }
    }
  }
  auto engine = std::make_shared<autograd::AutoDiff>(output, keep_graph, high_order, false);
  autograd::AutoDiffGuard auto_diff_guard(engine);
  auto grads = engine->RunBackward(input_tensors, sens_gradients, accumulate_grad);
  engine->RunFinalCallback();
  engine->Clear();
  if (accumulate_grad) {
    return py::none();
  }
  auto tuple_grads = grads->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_grads);
  py::tuple py_grads(tuple_grads->size());
  for (size_t i = 0; i < tuple_grads->size(); ++i) {
    if (tuple_grads->value()[i]->isa<None>() && allow_unreachable) {
      py_grads[i] = py::none();
      continue;
    }
    auto tensor = tuple_grads->value()[i]->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(EXCEPTION)
        << "One of the input Tensor's grad is None. Set allow_unused=True if this behavior meets expectations.";
    }
    py_grads[i] = tensor::PackTensorToPyObject(tensor);
  }
  return std::move(py_grads);
}

void RegBackwardFunction(py::module *m) {
  (void)m->def("run_backward", &RunBackward, py::arg("tensors"), py::arg("grad_tensors"), py::arg("keep_graph"),
               py::arg("create_graph"), py::arg("inputs"), py::arg("allow_unreachable") = True, py::kw_only(),
               py::arg("accumulate_grad") = True, "run backward function");
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
