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
#include "pipeline/jit/pi/auto_grad/function_node.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include "ops/sequence_ops.h"
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
void FunctionNode::RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs) {
  auto func_node = std::make_shared<FunctionNode>(out);
  std::for_each(inputs.begin(), inputs.end(), [&func_node, prim, inputs](const auto &obj) {
    auto input = py::cast<py::object>(obj);
    if (!IsStubTensor(input)) {
      return;
    }
    auto requires_grad = input.attr("requires_grad");
    if (py::isinstance<py::none>(requires_grad) || PyObject_Not(requires_grad.ptr())) {
      return;
    }
    py::object grad_fn = input.attr("grad_fn");
    if (py::isinstance<py::none>(grad_fn)) {
      grad_fn = py::cast(std::make_shared<FunctionNode>(input));
    }
    auto edge = grad_fn.cast<grad::FunctionNodePtr>();
    edge->GenBropFunction(prim, inputs);
    func_node->AddNextEdge(edge);
  });
  if (func_node->GetNextEdges().empty()) {
    return;
  }
  py::setattr(out, "grad_fn", py::cast(func_node));
  py::setattr(out, "requires_grad", py::bool_(True));
}

void InitGradIfNeed(const FunctionNodePtr &func_node, bool zero_like) {
  if (func_node->GetGrad() != nullptr) {
    return;
  }
  auto output = func_node->GetOutput();
  MS_EXCEPTION_IF_CHECK_FAIL(output->isa<tensor::Tensor>(), "Expected a tensor.");
  auto tensor = output->cast<tensor::TensorPtr>();
  tensor = TensorConstructUtils::CreateZerosTensor(tensor->Dtype(), tensor->shape());
  char *data = reinterpret_cast<char *>(tensor->data_c());
  std::fill(data, data + tensor->data().nbytes(), (zero_like ? 0 : 1));
  func_node->SetGrad(tensor);
}

/// \brief Generate the bprop function.
void FunctionNode::GenBropFunction(const py::object &prim, const py::tuple &inputs) {
  InputList values;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(values), [](const auto &input) {
    return pynative::PyNativeAlgo::DataConvert::PyObjToValue(py::cast<py::object>(input));
  });
  FunctionContext::SetInputs(values);
  auto executor = GradExecutor::GetInstance();
  auto prim_value = pynative::PyNativeAlgo::DataConvert::PyObjToValue(prim);
  grad_fn_ = executor->GetBpropGraph(NewValueNode(prim_value), GetInputs(), GetOutput(), GetOutput());
  size_t index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), tensor_));
  MS_EXCEPTION_IF_CHECK_FAIL(index < inputs.size(), "The index of tensor is invalid.");
  auto output = grad_fn_->output();
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    grad_fn_->set_output(output->cast<CNodePtr>()->input(index + 1));
  }
  if (!edges_.empty()) {
    return;
  }
  auto acc = grad::GradExecutor::GetInstance()->GetAccumulateGraph(tensor_);
  auto param = grad_fn_->add_parameter();
  param->set_abstract(acc->output()->abstract());
  auto ret = grad_fn_->NewCNode({NewValueNode(acc), param, grad_fn_->output()});
  grad_fn_->set_output(ret);
  grad_fn_ = executor->PrimBpropGraphPass(grad_fn_);
}

void FunctionNode::Apply(const py::object &grad) {
  if (py::isinstance<py::none>(grad)) {
    InitGradIfNeed(shared_from_base<FunctionNode>(), edges_.empty());
  } else {
    SetGrad(pynative::PyNativeAlgo::DataConvert::PyObjToValue(grad));
  }
  auto grad_value = GetGrad();
  auto executor = GradExecutor::GetInstance();
  InputList inputs(GetInputs());
  if (edges_.empty()) {
    inputs.push_back(grad_value);
  }
  if (grad_fn_ != nullptr) {
    grad_value = executor->RunGraph(grad_fn_, inputs);
    SetGrad(grad_value);
  }
  py::setattr(tensor_, "grad", pynative::PyNativeAlgo::DataConvert::ValueToPyObj(grad_value));

  auto output = GetOutput();
  for (auto &edge : edges_) {
    auto func_node = edge->GetFunction();
    func_node->AddInput(output);
    func_node->AddInput(grad_value);
    func_node->Apply();
  }
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
