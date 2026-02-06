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

#include "pynative/backward/hook/primitive_hook.h"
#include <memory>
#include <string>
#include "include/utils/operator/primitive_utils.h"
#include "include/utils/tensor_py.h"
#include "include/pynative/utils/pynative_execute.h"

namespace mindspore {
namespace {
constexpr auto kCellIDAttrName = "cell_id";
constexpr auto kCustomOpNameAttrName = "custom_op_name";
constexpr auto kIsRecomputeAttr = "is_recompute";

py::tuple UnfoldPyArgs(const py::tuple &py_args) {
  py::list list;
  for (size_t i = 0; i < py_args.size(); i++) {
    list = list + py::cast<py::list>(py_args[i]);
  }
  return py::cast<py::tuple>(list);
}
}  // namespace

BaseRef RunCustomOpBpropFunction(const PrimitivePyPtr &self, const py::tuple &ori_py_args) {
  auto is_custom_aot_node = self->HasAttr(kCustomOpNameAttrName) &&
                            GetValue<std::string>(self->GetAttr(kCustomOpNameAttrName)) == kCustomExtOpName;
  MS_LOG(DEBUG) << "Custom op:" << self->name() << ", is custom aot node: " << is_custom_aot_node;
  py::tuple py_args = is_custom_aot_node ? UnfoldPyArgs(ori_py_args) : ori_py_args;
  py::tuple converted_args = ConvertCTensorToPyTensor(py_args);
  MS_LOG(DEBUG) << "Get convert args size " << converted_args.size() << ", args are "
                << ConvertPyObjToString(converted_args);
  MS_LOG(DEBUG) << "Run custom op bprop start";
  py::object grads_obj = self->hook_fn()(*converted_args);
  auto grads = CheckBpropOut(grads_obj, py_args, self->bprop_cls_name());
  MS_LOG(DEBUG) << "Run custom op bprop end";
  return std::make_shared<PyObjectRef>(grads);
}

BaseRef RunCellHookFunction(const PrimitivePyPtr &self, const py::tuple &py_args) {
  const auto args_size = py_args.size();
  // Get the din passed to current bprop cut op.
  py::object grad_output = py_args[args_size - 1];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  const auto &hook_fn = self->hook_fn();
  MS_LOG(DEBUG) << "Get cell dout " << ConvertPyObjToString(grad_output);

  BaseRef res;
  if (self->hook_type() == HookType::kBackwardPreHook) {
    MS_LOG(DEBUG) << "Run cell backward pre hook function start.";
    py::object ret = hook_fn(grad_output);
    if (!py::isinstance<py::none>(ret)) {
      MS_LOG(DEBUG) << "Get cell backward pre hook new grad output " << ConvertPyObjToString(ret);
      const auto &code_obj = py::getattr(hook_fn, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      self->CheckHookConsistency(self->UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
      grad_output = ret;
    }
    MS_LOG(DEBUG) << "Run cell backward pre hook function end.";
    res = std::make_shared<PyObjectRef>(grad_output);
  } else {
    MS_LOG(DEBUG) << "Run cell backward hook function start.";
    py::object ret = hook_fn(grad_output);
    if (py::isinstance<py::str>(ret)) {
      MS_LOG(DEBUG) << "Run cell " << ret.cast<std::string>() << " backward hook function the first time";
      self->EmplaceUnpairBackwardHookGrad(ret.cast<std::string>(), hook_fn);
      return std::make_shared<PyObjectRef>(grad_output);
    }
    if (py::isinstance<py::none>(ret)) {
      MS_LOG(DEBUG) << "Run cell backward hook function the second time with return None.";
    } else {
      MS_LOG(DEBUG) << "Get cell backward hook new grad input " << ConvertPyObjToString(ret);
      const auto &code_obj = py::getattr(hook_fn, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      self->CheckHookConsistency(self->UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
      grad_output = ret;
    }
    self->EraseUnpairBackwardHookGrad(GetValue<std::string>(self->GetAttr(kCellIDAttrName)));
    MS_LOG(DEBUG) << "Run cell backward hook function end.";
    res = std::make_shared<PyObjectRef>(grad_output);
  }
  return res;
}

BaseRef RunVariableHookFunction(const PrimitivePyPtr &self, const py::tuple &py_args) {
  constexpr size_t grad_output_index = 0;
  py::object grad_output = py_args[grad_output_index];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  MS_LOG(DEBUG) << "Get grad output " << ConvertPyObjToString(grad_output);
  const auto &hook_fn = self->hook_fn();

  // Op maybe have multi outputs, so wrap to tuple for unitary.
  // Tensor hook just work on tensor, so keep origin input style
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  MS_LOG(DEBUG) << "Run HookBackward op function begin";

  auto ret = hook_fn(grad_output);
  if (!py::isinstance<py::none>(ret)) {
    MS_LOG(DEBUG) << "Get hook output " << ConvertPyObjToString(ret);
    grad_output = ret;
  }
  const auto &code_obj = py::getattr(hook_fn, "__code__");
  py::object co_name = py::getattr(code_obj, "co_name");
  self->CheckHookConsistency(self->UnpackRetValueOfCellHook(grad_output), py_args[grad_output_index], co_name);

  MS_LOG(DEBUG) << "Run HookBackward op function end";
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  return std::make_shared<PyObjectRef>(grad_output);
}

BaseRef RunHookFunction(const PrimitivePyPtr &self, const ValuePtrList &args) {
  py::tuple py_args = py::reinterpret_steal<py::tuple>(tensor::Wrap(args));
  MS_LOG(DEBUG) << "Get input args size " << py_args.size() << ", args are " << ConvertPyObjToString(py_args);
  BaseRef res;
  try {
    switch (self->hook_type()) {
      case HookType::kBackwardPreHook:  // For cell register backward pre hook
      case HookType::kBackwardHook:     // For cell register backward hook
        res = RunCellHookFunction(self, py_args);
        break;
      case HookType::kCustomOpBprop:
        res = RunCustomOpBpropFunction(self, py_args);  // For custom op, which define construct and bprop
        break;
      case HookType::kHookBackwardOp:
        res = RunVariableHookFunction(self, py_args);  // For HookBackward Op
        break;
      default:
        MS_LOG(EXCEPTION) << "Get unsupported hook type";
    }
  } catch (std::exception &) {
    const auto &inst = pynative::PyNativeExecutor::GetInstance();
    inst->ClearRes();
    if (self->hook_type() == HookType::kBackwardHook) {
      self->ClearUnpairBackwardHookGrad();
    }
    std::rethrow_exception(std::current_exception());
  }
  return res;
}

struct ProcessUnPairedCellHookRegister {
  ProcessUnPairedCellHookRegister() {
    python_adapter::PyAdapterCallback::SetProcessUnPairedCellHookHandler(
      [](bool execute_hook_fn) -> void { PrimitivePy::ProcessUnPairedCellHook(execute_hook_fn); });
  }
} cell_hook_callback_register;
}  // namespace mindspore
