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
#include "include/utils/pynative/variable.h"
#include "include/utils/pyobj_manager.h"
#include "pynative/backward/backward_node_py.h"

namespace mindspore {
namespace pynative {
namespace autograd {

// Check if object is GradientEdge
bool IsGradientEdge(const py::handle &obj) {
  if (!obj.check()) {
    return false;
  }
  PyObject *raw_obj = obj.ptr();
  if (PyObject_IsInstance(raw_obj, PyObjManager::Get().GetGradientEdgeClass())) {
    return true;
  }
  return false;
}

// Convert Python tuple (tensors or GradientEdge) to EdgeList
EdgeList ConvertPyTupleToEdgeList(const py::tuple &tuple_args, bool need_retain_grad = false) {
  EdgeList edges;
  edges.reserve(tuple_args.size());
  for (size_t i = 0; i < tuple_args.size(); ++i) {
    py::object obj = tuple_args[i];
    BackwardNodePtr grad_node = nullptr;
    size_t output_index = 0;
    if (IsGradientEdge(obj)) {
      py::tuple gradient_edge = py::cast<py::tuple>(obj);
      py::object grad_node_obj = gradient_edge[0];
      if (IsFunctionBasePy(grad_node_obj)) {
        grad_node = reinterpret_cast<FunctionBase *>(grad_node_obj.ptr())->weak_grad_node.lock();
        MS_EXCEPTION_IF_NULL(grad_node);
      } else if (IsBackwardNodePy(grad_node_obj)) {
        grad_node = reinterpret_cast<BackwardNodePy *>(grad_node_obj.ptr())->cdata;
        MS_EXCEPTION_IF_NULL(grad_node);
      } else {
        MS_LOG(EXCEPTION) << "Element 0 of tuple should be grad node, but got " << py::str(grad_node_obj);
      }
      if (!py::isinstance<py::int_>(gradient_edge[1])) {
        MS_LOG(EXCEPTION) << "Element 1 of tuple should be int type, but got " << py::str(gradient_edge[1]);
      }
      auto raw_index = py::cast<int64_t>(gradient_edge[1]);
      MS_EXCEPTION_IF_CHECK_FAIL(raw_index >= 0, "output_index must be non-negative");
      output_index = static_cast<size_t>(raw_index);
    } else if (tensor::IsTensorPy(obj)) {
      auto tensor = tensor::ConvertToTensor(obj);
      MS_EXCEPTION_IF_NULL(tensor);
      if (need_retain_grad) {
        tensor->retain_grad();
      }
      grad_node = impl::GetUnsafeGradNodeImpl(tensor);
      const auto &grad_meta = tensor->auto_grad_meta_data();
      if (grad_meta != nullptr) {
        output_index = grad_meta->output_index();
      }
    } else {
      MS_LOG(EXCEPTION) << "Elements of tuple should be tensor or GradientEdge, but got " << py::str(obj);
    }
    (void)edges.emplace_back(grad_node, output_index);
  }
  return edges;
}

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

  // Convert tensors to EdgeList (supports both tensor tuples and GradientEdge tuples)
  auto tuple_tensors = py::cast<py::tuple>(tensors);
  EdgeList output_edges = ConvertPyTupleToEdgeList(tuple_tensors);

  // Convert grad_tensors to ValuePtr (sens)
  ValuePtrList sens_gradients;
  if (py::isinstance<py::tuple>(grad_tensors)) {
    sens_gradients = ConvertPyTupleToTensorList(py::cast<py::tuple>(grad_tensors));
  } else if (!py::isinstance<py::none>(grad_tensors)) {
    MS_LOG(EXCEPTION) << "grad_tensors should be tuple or none! but got " << py::str(grad_tensors);
  }

  // Convert inputs to EdgeList (supports both tensor tuples and GradientEdge tuples)
  EdgeList input_edges;
  if (py::isinstance<py::tuple>(inputs)) {
    auto tuple_inputs = py::cast<py::tuple>(inputs);
    if (!accumulate_grad && tuple_inputs.empty()) {
      MS_LOG(EXCEPTION) << "Please set inputs for grad interface, grad requires non-empty inputs.";
    }
    input_edges = ConvertPyTupleToEdgeList(tuple_inputs, accumulate_grad);
  } else if (!py::isinstance<py::none>(inputs)) {
    MS_LOG(EXCEPTION) << "input tensors should be tuple or none! but got " << py::str(inputs);
  }

  auto engine = std::make_shared<autograd::AutoDiff>(keep_graph, high_order, false);
  autograd::AutoDiffGuard auto_diff_guard(engine);
  auto grads = engine->RunBackward(output_edges, input_edges, sens_gradients, accumulate_grad);
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
