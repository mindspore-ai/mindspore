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
#include <memory>
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace jit {
namespace grad {
void FunctionNode::AddInput(const py::object &input) {
  FunctionContext::AddInput(pynative::PyNativeAlgo::DataConvert::PyObjToValue(input));
}

void FunctionNode::SetOutput(const py::object &output) {
  FunctionContext::SetOutput(pynative::PyNativeAlgo::DataConvert::PyObjToValue(output));
}

/// \brief Generate the bprop function.
void FunctionNode::GenBropFunction() {
  auto executor = GradExecutor::GetInstance();
  if (IsStubTensor(fn_)) {
    grad_fn_ = grad::GradExecutor::GetInstance()->GetAccumulateGraph(fn_);
  } else {
    auto prim = pynative::PyNativeAlgo::DataConvert::PyObjToValue(fn_);
    grad_fn_ = executor->GetBpropGraph(NewValueNode(prim), GetInputs(), GetOutput(), GetOutput());
  }
}

void InitGradIfNeed(const FunctionNodePtr &func_node, bool ones_like = false) {
  if (func_node->GetGrad() != nullptr) {
    return;
  }
  auto output = func_node->GetOutput();
  MS_EXCEPTION_IF_CHECK_FAIL(output->isa<tensor::Tensor>(), "Expected a tensor.");
  auto tensor = output->cast<tensor::TensorPtr>();
  tensor = TensorConstructUtils::CreateZerosTensor(tensor->Dtype(), tensor->shape());
  char *data = reinterpret_cast<char *>(tensor->data_c());
  std::fill(data, data + tensor->data().nbytes(), (ones_like ? 1 : 0));
  func_node->SetGrad(tensor);
}

void FunctionNode::Apply(const py::object &grad) {
  if (!py::isinstance<py::none>(grad)) {
    SetGrad(pynative::PyNativeAlgo::DataConvert::PyObjToValue(grad));
  } else {
    InitGradIfNeed(shared_from_base<FunctionNode>(), true);
  }
  auto executor = GradExecutor::GetInstance();
  if (IsStubTensor(fn_)) {
    auto grad_value = executor->RunGraph(grad_fn_, GetInputs());
    SetGrad(grad_value);
    auto grad = pynative::PyNativeAlgo::DataConvert::ValueToPyObj(grad_value);
    py::setattr(fn_, "grad", grad);
  } else {
    auto grad_value = executor->RunGraph(grad_fn_, GetInputs(), GetOutput(), GetGrad());
    for (auto &edge : edges_) {
      auto func_node = edge->GetFunction();
      if (grad_value->isa<ValueTuple>()) {
        auto values = grad_value->cast<ValueTuplePtr>()->value();
        if (IsStubTensor(func_node->GetFunction())) {
          InitGradIfNeed(func_node);
          func_node->SetInputs({func_node->GetGrad(), values[edge->GetIndex()]});
        } else {
          func_node->SetGrad(values[edge->GetIndex()]);
        }
      } else {
        if (IsStubTensor(func_node->GetFunction())) {
          InitGradIfNeed(func_node);
          func_node->SetInputs({func_node->GetGrad(), grad_value});
        } else {
          edge->GetFunction()->SetGrad(grad_value);
        }
      }
      edge->GetFunction()->Apply();
    }
  }
}
}  // namespace grad
}  // namespace jit
}  // namespace mindspore
