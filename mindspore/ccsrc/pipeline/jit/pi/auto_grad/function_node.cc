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
#include <utility>
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "ops/sequence_ops.h"
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
bool IsRequiresGradient(const py::object &input) {
  if (!IsStubTensor(input)) {
    return false;
  }
  auto requires_grad = input.attr("requires_grad");
  return !py::isinstance<py::none>(requires_grad) && PyObject_IsTrue(requires_grad.ptr());
}

FunctionNodePtr GetOrCreateFunctionNode(const py::object &obj, const py::object &prim, const py::object &out) {
  MS_EXCEPTION_IF_CHECK_FAIL(IsStubTensor(obj), "Must be a stub tensor.");
  py::object grad_fn = obj.attr("grad_fn");
  if (py::isinstance<py::none>(grad_fn)) {
    return std::move(std::make_shared<FunctionNode>(obj, prim, out));
  }
  auto func_node = grad_fn.cast<grad::FunctionNodePtr>();
  func_node->SetFunction(Convert::PyObjToValue(prim));
  func_node->SetOutput(Convert::PyObjToValue(out));
  return func_node;
}

void PostBpropFunctionToEdges(const py::object &tensor) {
  if (!IsStubTensor(tensor)) {
    return;
  }
  py::object grad_fn = tensor.attr("grad_fn");
  if (py::isinstance<py::none>(grad_fn)) {
    return;
  }
  auto func_node = grad_fn.cast<grad::FunctionNodePtr>();
  auto edges = func_node->GetNextEdges();
  std::for_each(edges.begin(), edges.end(), [](const EdgePtr &edge) { edge->GetFunction()->GenerateBropFunction(); });
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

void FunctionNode::RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs) {
  auto record_task = std::make_shared<RecordTask>(
    [](const py::object &prim, const py::object &out, const py::list &inputs) {
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      auto func_node = std::make_shared<FunctionNode>(out);
      func_node->InitDataField(prim, inputs);
    },
    prim, out, inputs);
  GradExecutor::GetInstance()->DispatchRecordTask(record_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetRecordTaskQueue()->Wait();
  }
  PostBpropFunctionToEdges(out);
}

void FunctionNode::InitDataField(const py::object &prim, const py::list &inputs) {
  std::for_each(inputs.begin(), inputs.end(), [this, &prim, &inputs](const auto &obj) {
    auto input = py::cast<py::object>(obj);
    if (!IsRequiresGradient(input)) {
      return;
    }
    auto edge = GetOrCreateFunctionNode(input, prim, tensor_);
    edge->SetInputs(inputs);
    AddNextEdge(edge);
  });
  if (edges_.empty()) {
    return;
  }
  py::setattr(tensor_, "grad_fn", py::cast(shared_from_base<FunctionNode>()));
  py::setattr(tensor_, "requires_grad", py::bool_(True));
}

void FunctionNode::SetInputs(const py::list &inputs) {
  RemoveInputs();
  std::for_each(inputs.begin(), inputs.end(),
                [this](const auto &input) { AddInput(Convert::PyObjToValue(py::cast<py::object>(input))); });
  index_ = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), tensor_));
}

/// \brief Generate the bprop function.
void FunctionNode::GenerateBropFunction() {
  auto generate_task = std::make_shared<RunBpropTask>([this]() {
    auto inputs = GetInputs();
    auto output = GetOutput();
    auto executor = GradExecutor::GetInstance();
    // gil for PyObject accessing
    py::gil_scoped_acquire gil_acquire;
    grad_fn_ = BasicClone(executor->GetBpropGraph(NewValueNode(GetFunction()), inputs, output, output));
    MS_EXCEPTION_IF_CHECK_FAIL(index_ < inputs.size(), "The index of tensor is invalid.");
    auto grad_output = grad_fn_->output();
    if (IsPrimitiveCNode(grad_output, prim::kPrimMakeTuple)) {
      grad_fn_->set_manager(Manage(grad_fn_, true));
      grad_fn_->set_output(grad_output->cast<CNodePtr>()->input(index_ + 1));
    }
    if (!edges_.empty()) {
      return;
    }
    auto acc = executor->GetAccumulateGraph(output);
    auto param = grad_fn_->add_parameter();
    param->set_abstract(acc->output()->abstract());
    auto ret = grad_fn_->NewCNode({NewValueNode(acc), param, grad_fn_->output()});
    grad_fn_->set_output(ret);
    grad_fn_ = executor->PrimBpropGraphPass(grad_fn_);
  });
  GradExecutor::GetInstance()->DispatchGenerateTask(generate_task);
}

void FunctionNode::Apply(const py::object &grad) {
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetGenerateTaskQueue()->Wait();
  }
  if (py::isinstance<py::none>(grad)) {
    InitGradIfNeed(shared_from_base<FunctionNode>(), edges_.empty());
  } else {
    SetGrad(Convert::PyObjToValue(grad));
  }
  ApplyInner(GetGrad());
}

void FunctionNode::ApplyInner(const ValuePtr &dout) {
  auto run_task = std::make_shared<RunBpropTask>([this, &dout]() {
    if (grad_fn_ != nullptr) {
      if (!edges_.empty()) {
        SetGrad(GradExecutor::GetInstance()->RunGraph(grad_fn_, GetInputs(), GetOutput(), dout));
      } else {
        InitGradIfNeed(shared_from_base<FunctionNode>(), true);
        InputList inputs = {GetOutput(), dout, GetGrad()};
        inputs.insert(inputs.begin(), GetInputs().begin(), GetInputs().end());
        SetGrad(GradExecutor::GetInstance()->RunGraph(grad_fn_, inputs));
      }
    }
    {
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      py::setattr(tensor_, "grad", Convert::ValueToPyObj(GetGrad()));
    }

    for (auto &edge : edges_) {
      // gil for PyEval_SaveThread
      py::gil_scoped_acquire gil_acquire;
      auto func_node = edge->GetFunction();
      func_node->ApplyInner(GetGrad());
    }
  });

  GradExecutor::GetInstance()->DispatchRunTask(run_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetRunTaskQueue()->Wait();
  }
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
