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
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include "ops/sequence_ops.h"
#include "pipeline/jit/pi/auto_grad/edge.h"
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include "pipeline/jit/pi/auto_grad/native_backward_function.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
void FunctionNode::CleanResource() {
  if (HasGradFunc(tensor_)) {
    py::setattr(tensor_, "grad_fn", py::none());
  }
  tensor_ = py::none();
  backward_func_ = nullptr;
  edges_.clear();
  dependences_.clear();
}

void PostBpropFunctionToEdges(const py::object &tensor) {
  py::object grad_fn = python_adapter::GetPyObjAttr(tensor, "grad_fn");
  if (py::isinstance<py::none>(grad_fn)) {
    return;
  }
  auto func_node = grad_fn.cast<grad::FunctionNodePtr>();
  func_node->GenerateBropFunction();
  auto edges = func_node->GetNextEdges();
  std::for_each(edges.begin(), edges.end(), [](const EdgePtr &edge) { edge->GetNode()->GenerateBropFunction(); });
}

ValuePtrList ConvertTupleToValueList(const py::list &inputs) {
  ValuePtrList value_list;
  for (const auto input : inputs) {
    ValuePtr value = Convert::PyObjToValue(py::cast<py::object>(input));
    if (value->template isa<None>()) {
      return value_list;
    }
    value_list.push_back(value);
  }
  return value_list;
}

ValuePtr ConvertArgByCastDtype(const py::object &arg, const ops::OpInputArg &op_arg) {
  parse::OpDefConvertFunc convert_func = parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
  MS_EXCEPTION_IF_NULL(convert_func);
  ValuePtr value = convert_func(arg);
  if (value != nullptr) {
    return value;
  }
  for (auto cast_dtype : op_arg.cast_dtype_) {
    convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
    MS_EXCEPTION_IF_NULL(convert_func);
    value = convert_func(arg);
    if (value != nullptr) {
      return value;
    }
  }
  if (!py::isinstance<py::none>(arg) && value == nullptr) {
    value = Convert::PyObjToValue(arg);
  }
  return value;
}

ValuePtrList ParseInputsByOpDef(const PrimitivePyPtr &prim, const ops::OpDefPtr &op_def, const py::list &inputs) {
  ValuePtrList input_values;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() <= op_def->args_.size(), "The arguments is not match defined.");
  size_t index = 0;
  std::for_each(op_def->args_.begin(), op_def->args_.end(), [&prim, &inputs, &input_values, &index](const auto &arg) {
    if (!arg.as_init_arg_) {
      input_values.push_back(ConvertArgByCastDtype(inputs[index], arg));
    } else {
      auto value = py::getattr(prim->GetPyObj(), common::SafeCStr(arg.arg_name_));
      if (!py::isinstance<py::none>(value)) {
        input_values.push_back(ConvertArgByCastDtype(value, arg));
      }
    }
    index++;
  });
  return input_values;
}

ValuePtrList ParseInputs(const PrimitivePyPtr &prim, const py::list &inputs) {
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def == nullptr) {
    return ConvertTupleToValueList(inputs);
  }
  return ParseInputsByOpDef(prim, op_def, inputs);
}

bool FunctionNode::IsRequiresGradient(const py::handle &obj) {
  if (!HasAttrReqGrad(obj)) {
    return false;
  }
  auto requires_grad = obj.attr("requires_grad");
  return py::isinstance<py::bool_>(requires_grad) && py::bool_(requires_grad);
}

bool FunctionNode::HasGradFunc(const py::handle &obj) {
  auto grad_fn = python_adapter::GetPyObjAttr(py::cast<py::object>(obj), "grad_fn");
  return py::isinstance<grad::FunctionNode>(grad_fn);
}

FunctionNodePtr GetOrCreateFunctionNode(const py::object &tensor, const py::object &prim, const py::object &out,
                                        const py::list &inputs) {
  py::object grad_fn = python_adapter::GetPyObjAttr(tensor, "grad_fn");
  if (py::isinstance<grad::FunctionNode>(grad_fn)) {
    return grad_fn.cast<grad::FunctionNodePtr>();
  }
  auto func_node = FunctionNode::CreateFunctionNode(tensor, prim, out, inputs);
  if (!func_node->GetNextEdges().empty()) {
    py::setattr(func_node->GetTensor(), "grad_fn", py::cast(func_node));
    py::setattr(func_node->GetTensor(), "requires_grad", py::bool_(True));
  }
  return func_node;
}

FunctionNodePtr FunctionNode::CreateFunctionNode(const py::object &tensor, const py::object &prim,
                                                 const py::object &out, const py::list &inputs) {
  auto func_node = std::make_shared<FunctionNode>(tensor, prim, out);
  MS_LOG_DEBUG << "Create a function node(" << func_node.get() << ") for " << tensor.ptr();
  func_node->SetInputs(inputs);
  std::for_each(inputs.begin(), inputs.end(), [func_node, &inputs](const auto &obj) {
    if (!FunctionNode::IsRequiresGradient(obj) && !FunctionNode::HasGradFunc(obj)) {
      return;
    }
    auto input = py::cast<py::object>(obj);
    auto node = GetOrCreateFunctionNode(input, py::none(), input, py::list());
    func_node->AddNextEdge(node, std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), input)));
  });
  return func_node;
}

void FunctionNode::RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs) {
  MS_LOG_DEBUG << "Record " << out.ptr() << " for auto gradient.";
  if (!py::isinstance<py::tuple>(out)) {
    (void)GetOrCreateFunctionNode(out, prim, out, inputs);
  } else {
    auto func_node = CreateFunctionNode(out, prim, out, inputs);
    std::for_each(out.begin(), out.end(), [&out, &func_node](const auto &obj) {
      if (!HasAttrReqGrad(obj) || HasGradFunc(obj)) {
        return;
      }
      auto tensor = py::cast<py::object>(obj);
      auto temp_node = GetOrCreateFunctionNode(tensor, py::none(), tensor, py::list());
      temp_node->index_ = std::distance(out.begin(), std::find(out.begin(), out.end(), tensor));
      temp_node->AddNextEdge(func_node, 0);
      py::setattr(tensor, "grad_fn", py::cast(temp_node));
      py::setattr(tensor, "requires_grad", py::bool_(True));
    });
  }
}

void FunctionNode::SetInputs(const py::list &inputs) {
  if (py::isinstance<py::none>(inputs) || inputs.empty()) {
    FunctionContext::SetInputs({});
  } else {
    FunctionContext::SetInputs(ParseInputs(GetFunction()->cast<PrimitivePyPtr>(), inputs));
  }
}

void FunctionNode::ApplyEdges(const ValuePtrList &grad_values) {
  MS_EXCEPTION_IF_CHECK_FAIL((grad_values.size() == edges_.size()), "The gradient values is not match.");
  for (size_t index = 0; index < edges_.size(); index++) {
    Notify(edges_[index]->GetNode(), grad_values[index]);
  }
}

void FunctionNode::ApplyNative() {
  ValuePtrList flatten_values = ValuePtrList(edges_.size(), GetGrad()[0]);
  if (backward_func_ != nullptr) {
    backward_func_->SetGradientIndexes({});
    std::for_each(edges_.begin(), edges_.end(),
                  [this](const auto &edge) { backward_func_->AddGradientIndex(edge->GetIndex()); });
    if (GetOutput()->isa<ValueTuple>()) {
      flatten_values = backward_func_->Run(GetInputs(), GetOutput(), MakeValue(GetGrad()));
    } else {
      flatten_values = backward_func_->Run(GetInputs(), GetOutput(), GetGrad()[index_]);
    }
  }
  ApplyEdges(flatten_values);
}

/// \brief Generate the bprop function.
void FunctionNode::GenerateBropFunction() {
  auto generate_task = std::make_shared<RunGenerateBpropTask>([this]() {
    MS_LOG_DEBUG << "Generate brop function for node " << tensor_.ptr() << ", tensor is " << tensor_.ptr();
    auto output = GetOutput();
    auto executor = GradExecutor::GetInstance();
    {
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      auto acc_fn = executor->GetAccumulateGraph(output);
      acc_fn_ = executor->PrimBpropGraphPass(acc_fn);
    }

    auto func = GetFunction();
    if (func->isa<None>()) {
      return;
    }
    try {
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      grad_fn_ = executor->GetBpropGraph(NewValueNode(func), GetInputs(), output, output);
    } catch (const std::exception &e) {
      MS_LOG_ERROR << "Prim : " << func->ToString() << " Output : " << output->ToString();
      MS_LOG_ERROR << e.what();
    }
  });
  GradExecutor::GetInstance()->DispatchGenerateTask(generate_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetGenerateTaskQueue()->Wait();
  }
}

void Visit(const FunctionNodePtr &node, const std::function<void(const FunctionNodePtr &)> &callback) {
  std::queue<FunctionNodePtr> nodes;
  nodes.push(node);
  while (!nodes.empty()) {
    auto fn = nodes.front();
    nodes.pop();
    callback(fn);
    std::for_each(fn->GetNextEdges().begin(), fn->GetNextEdges().end(),
                  [&nodes](const auto &edge) { nodes.push(edge->GetNode()); });
  }
}

void FunctionNode::SyncGradToPyObject() {
  auto sync_func = [](const FunctionNodePtr &node) {
    auto retains_grad = python_adapter::GetPyObjAttr(node->tensor_, "retains_grad");
    if (node->edges_.empty() || (py::isinstance<py::bool_>(retains_grad) && py::bool_(retains_grad))) {
      auto _grad = python_adapter::GetPyObjAttr(node->tensor_, "grad");
      if (!py::isinstance<py::none>(_grad)) {
        node->AccumulateGradient(Convert::PyObjToValue(_grad), node->index_);
      } else {
        if (node->GetGrad()[node->index_]->isa<None>()) {
          auto func = node->backward_func_;
          if (func == nullptr) {
            func = NativeBackwardFunc::GetInstance(prim::kPrimAdd);
          }
          node->SetGrad(func->Zeros(node->GetOutput()), node->index_);
        }
      }
      auto value = Convert::ValueToPyObj(node->GetGrad()[node->index_]);
      auto grad = python_adapter::CallPyFn("mindspore.common.api", "_convert_python_data", value);
      py::setattr(node->tensor_, "grad", grad);
    }
    node->SetGrad(ValuePtrList(node->GetGrad().size(), kNone));
  };
  Visit(shared_from_base<FunctionNode>(), sync_func);
}

void FunctionNode::Apply(const py::object &grad) {
  UpdateDependence();
  Notify(shared_from_base<FunctionNode>(), Convert::PyObjToValue(grad));
  SyncGradToPyObject();
  auto release_func = [](const FunctionNodePtr &node) { node->CleanResource(); };
  Visit(shared_from_base<FunctionNode>(), release_func);
}

void FunctionNode::ApplyInner(const ValuePtr &dout) {
  MS_LOG_DEBUG << "Start run apply() of " << tensor_.ptr() << ", tensor is " << tensor_.ptr();
  MS_LOG_DEBUG << "Prim is " << GetFunction()->ToString() << ", dout is " << dout->ToString();
  auto run_task = std::make_shared<RunBpropTask>(
    [this](const ValuePtr &dout) {
      AccumulateGradient(dout, index_);
      if (grad_fn_ == nullptr) {
        return;
      }
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      auto ret = GradExecutor::GetInstance()->RunGraph(grad_fn_, GetInputs(), GetOutput(), dout);
      if (!ret->isa<ValueTuple>()) {
        return;
      }
      auto tuple = ret->cast<ValueTuplePtr>();
      std::for_each(edges_.begin(), edges_.end(),
                    [&tuple](const auto &edge) { edge->GetNode()->ApplyInner(tuple->value()[edge->GetIndex()]); });
    },
    dout);

  GradExecutor::GetInstance()->DispatchRunTask(run_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetRunTaskQueue()->Wait();
  }
}

void FunctionNode::UpdateDependence() {
  auto mark_func = [](const FunctionNodePtr &node) { node->is_in_reverse_chain_ = true; };
  Visit(shared_from_base<FunctionNode>(), mark_func);
  auto update_func = [](const FunctionNodePtr &node) {
    for (auto iter = node->dependences_.begin(); iter != node->dependences_.end();) {
      if (!(*iter)->is_in_reverse_chain_) {
        iter = node->dependences_.erase(iter);
      } else {
        iter++;
      }
    }
  };
  dependences_.clear();
  dependences_.insert(shared_from_base<FunctionNode>());
  Visit(shared_from_base<FunctionNode>(), update_func);
}

void FunctionNode::Notify(const FunctionNodePtr &node, const ValuePtr &dout) {
  node->AccumulateGradient(dout, node->index_);
  node->depend_cnt_.fetch_add(1);
  if (!node->IsReady()) {
    return;
  }
  node->ApplyNative();
  node->depend_cnt_.store(0);
  node->dependences_.clear();
}

void FunctionNode::AccumulateGradient(const ValuePtr &dout, size_t index) {
  if (dout->isa<None>()) {
    return;
  }
  auto func = backward_func_;
  if (func == nullptr) {
    func = NativeBackwardFunc::GetInstance(prim::kPrimAdd);
  }
  std::unique_lock<std::mutex> lock(mutex_);
  auto value = GetGrad()[index];
  if (value->isa<None>()) {
    SetGrad(dout, index);
  } else {
    SetGrad(func->Add(dout, GetGrad()[index]), index);
  }
}

std::string FunctionNode::ToString() const {
  std::stringstream ss;
  Dump(ss, "");
  return ss.str();
}

void FunctionNode::Dump(std::stringstream &ss, const std::string &prefix) const {
  if (!prefix.empty()) {
    ss << prefix << "-->";
  }
  auto prim = GetFunction();
  ss << "FunctionNode(" << tensor_.ptr() << "(" << this << "), depend(" << dependences_.size() << "), "
     << (prim->isa<None>() ? "None" : prim->ToString()) << ", "
     << py::bool_(python_adapter::GetPyObjAttr(tensor_, "is_leaf")) << ")\n";
  std::for_each(edges_.begin(), edges_.end(),
                [&ss, &prefix](const auto &edge) { edge->GetNode()->Dump(ss, prefix + "   "); });
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
