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
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include "ir/func_graph_cloner.h"
#include "ops/sequence_ops.h"
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
tensor::TensorPtr CreateZerosTensorLike(const py::object &tensor) {
  auto tensor_abs = parse::ConvertTensorValue(tensor)->ToAbstract();
  auto tensor_shape = dyn_cast<abstract::Shape>(tensor_abs->BuildShape())->shape();
  return TensorConstructUtils::CreateZerosTensor(tensor_abs->BuildType(), tensor_shape);
}

bool IsRequiresGradient(const py::object &input) {
  auto requires_grad = python_adapter::GetPyObjAttr(input, "requires_grad");
  return py::isinstance<py::bool_>(requires_grad) && py::bool_(requires_grad);
}

FunctionNodePtr GetOrCreateFunctionNode(const py::object &tensor, const py::object &prim, const py::object &out,
                                        const py::list &inputs) {
  py::object grad_fn = python_adapter::GetPyObjAttr(tensor, "grad_fn");
  if (py::isinstance<grad::FunctionNode>(grad_fn)) {
    return grad_fn.cast<grad::FunctionNodePtr>();
  }
  auto func_node = std::make_shared<FunctionNode>(tensor, prim, out);
  MS_LOG_DEBUG << "Create a function node for " << tensor.ptr() << ", prim is " << func_node->GetFunction()->ToString();
  func_node->SetInputs(inputs);
  return func_node;
}

void PostBpropFunctionToEdges(const py::object &tensor) {
  py::object grad_fn = python_adapter::GetPyObjAttr(tensor, "grad_fn");
  if (py::isinstance<py::none>(grad_fn)) {
    return;
  }
  auto func_node = grad_fn.cast<grad::FunctionNodePtr>();
  func_node->GenerateBropFunction();
  auto edges = func_node->GetNextEdges();
  std::for_each(edges.begin(), edges.end(), [](const EdgePtr &edge) { edge->GetFunction()->GenerateBropFunction(); });
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

InputList ParseInputsByOpDef(const PrimitivePyPtr &prim, const py::list &inputs) {
  InputList input_values;
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def == nullptr) {
    bool enable_discard = false;
    std::for_each(inputs.begin(), inputs.end(), [&input_values, &enable_discard](const auto &input) {
      // if a argument is None, itself and the arguments that fellow it must be optional, so discard them
      enable_discard = (enable_discard || py::isinstance<py::none>(input));
      if (enable_discard) {
        return;
      }
      input_values.push_back(Convert::PyObjToValue(py::cast<py::object>(input)));
    });
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() <= op_def->args_.size(),
                               "The arguments of " + prim->name() + " is not match defined.");
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
  }
  return input_values;
}

void FunctionNode::RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs) {
  MS_LOG_DEBUG << "Record " << out.ptr() << " for auto gradient.";
  auto record_task = std::make_shared<RecordTask>(
    [](const py::object &prim, const py::object &out, const py::list &inputs) {
      // gil for PyObject accessing
      py::gil_scoped_acquire gil_acquire;
      auto grad_fn = python_adapter::GetPyObjAttr(out, "grad_fn");
      if (py::isinstance<grad::FunctionNode>(grad_fn)) {
        return;
      }
      auto func_node = GetOrCreateFunctionNode(out, prim, out, inputs);
      func_node->InitDataField(inputs);
    },
    prim, out, inputs);
  GradExecutor::GetInstance()->DispatchRecordTask(record_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetRecordTaskQueue()->Wait();
  }
  PostBpropFunctionToEdges(out);
}

void FunctionNode::InitDataField(const py::list &inputs) {
  if (py::isinstance<py::none>(inputs) || inputs.empty()) {
    return;
  }
  MS_LOG_DEBUG << "Init gradient function and edges for " << tensor_.ptr();
  std::for_each(inputs.begin(), inputs.end(), [this, &inputs](const auto &obj) {
    auto input = py::cast<py::object>(obj);
    if (!IsRequiresGradient(input)) {
      return;
    }
    auto edge = GetOrCreateFunctionNode(input, py::none(), input, py::list());
    AddNextEdge(edge, std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), input)));
  });
  if (edges_.empty()) {
    return;
  }
  py::setattr(tensor_, "grad_fn", py::cast(shared_from_base<FunctionNode>()));
  py::setattr(tensor_, "requires_grad", py::bool_(True));
}

void FunctionNode::SetInputs(const py::list &inputs) {
  if (py::isinstance<py::none>(inputs) || inputs.empty()) {
    FunctionContext::SetInputs({});
  } else {
    FunctionContext::SetInputs(ParseInputsByOpDef(GetFunction()->cast<PrimitivePyPtr>(), inputs));
  }
}

/// \brief Generate the bprop function.
void FunctionNode::GenerateBropFunction() {
  auto generate_task = std::make_shared<RunGenerateBpropTask>([this]() {
    MS_LOG_DEBUG << "Generate brop function for node " << this << ", tensor is " << tensor_.ptr();
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

void FunctionNode::SyncGradToPyObject() {
  std::for_each(edges_.begin(), edges_.end(), [](const auto &edge) { edge->GetFunction()->SyncGradToPyObject(); });
  auto retains_grad = python_adapter::GetPyObjAttr(tensor_, "retains_grad");
  if (!edges_.empty() && !(py::isinstance<py::bool_>(retains_grad) && py::cast<bool>(retains_grad))) {
    return;
  }
  auto _grad = python_adapter::GetPyObjAttr(tensor_, "grad");
  if (!py::isinstance<py::none>(_grad)) {
    AccumulateGradient(Convert::PyObjToValue(_grad));
  }
  MS_LOG_DEBUG << "sync the gradient to " << tensor_.ptr() << ", value is " << GetGrad()->ToString();
  auto value = Convert::ValueToPyObj(GetGrad());
  auto grad = python_adapter::CallPyFn("mindspore.common.api", "_convert_python_data", value);
  py::setattr(tensor_, "grad", grad);
}

void FunctionNode::Apply(const py::object &grad) {
  MS_LOG_DEBUG << ToString();
  GradExecutor::GetInstance()->Clear();
  ApplyInner(Convert::PyObjToValue(grad));
  SyncGradToPyObject();
}

void FunctionNode::ApplyInner(const ValuePtr &dout) {
  MS_LOG_DEBUG << "Start run apply() of " << this << ", tensor is " << tensor_.ptr();
  MS_LOG_DEBUG << "Prim is " << GetFunction()->ToString() << ", dout is " << dout->ToString();
  auto run_task = std::make_shared<RunBpropTask>(
    [this](const ValuePtr &dout) {
      AccumulateGradient(dout);
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
                    [&tuple](const auto &edge) { edge->GetFunction()->ApplyInner(tuple->value()[edge->GetIndex()]); });
    },
    dout);

  GradExecutor::GetInstance()->DispatchRunTask(run_task);
  {
    py::gil_scoped_release release;
    GradExecutor::GetInstance()->GetAsyncTaskManager()->GetRunTaskQueue()->Wait();
  }
}

void FunctionNode::AccumulateGradient(const ValuePtr &dout) {
  std::unique_lock<std::mutex> lock(mutex_);
  SetGrad(GradExecutor::GetInstance()->RunGraph(acc_fn_, {dout, GetGrad()}));
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
  ss << "FunctionNode(" << tensor_.ptr() << ", " << (prim->isa<None>() ? "None" : prim->ToString()) << ", "
     << py::bool_(python_adapter::GetPyObjAttr(tensor_, "is_leaf")) << ")\n";
  std::for_each(edges_.begin(), edges_.end(),
                [&ss, &prefix](const auto &edge) { edge->GetFunction()->Dump(ss, prefix + "   "); });
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
