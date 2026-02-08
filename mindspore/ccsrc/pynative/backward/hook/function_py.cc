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

#include "pynative/backward/hook/function_py.h"

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "utils/log_adapter.h"
#include "utils/ordered_map.h"
#include "include/pynative/utils/pynative_utils.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "include/utils/tensor_py.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/pynative/grad_state.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/utils/exception.h"
#include "include/utils/pyobj_manager.h"
#include "pynative/backward/backward_node_py.h"
#include "pybind_api/pynative/pynative_api.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
ValuePtrList ConvertOutputTensorList(const py::object &obj) {
  auto tuple = py::cast<py::tuple>(obj);
  ValuePtrList res;
  res.reserve(tuple.size());
  for (size_t i = 0; i < tuple.size(); i++) {
    auto tensor = tensor::ConvertToTensor(tuple[i]);
    if (tensor == nullptr) {
      res.emplace_back(kNone);
    } else {
      tensor->set_need_pipeline_sync(true);
      res.emplace_back(tensor);
    }
  }
  return res;
}

TensorPtr ViewAsSelfWithNoGrad(const TensorPtr &self) {
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  return kernel::pyboost::view(self, self->shape());
}

PyObject *PackForwardInput(PyObject *ctx, PyObject *inputs) {
  Py_ssize_t input_size = PyTuple_GET_SIZE(inputs);
  PyObject *forward_inputs = PyTuple_New(input_size + kSizeOne);
  Py_INCREF(ctx);
  PyTuple_SetItem(forward_inputs, 0, ctx);
  for (Py_ssize_t index = 0; index < input_size; ++index) {
    PyObject *input = PyTuple_GET_ITEM(inputs, index);
    Py_INCREF(input);
    PyTuple_SetItem(forward_inputs, index + 1, input);
  }
  return forward_inputs;
}

void ProcessInputs(const std::shared_ptr<FunctionContext> &context, FunctionBase *ctx, PyObject *inputs) {
  size_t inputs_size = static_cast<size_t>(PyTuple_GET_SIZE(inputs));
  std::vector<bool> is_tensor_input;
  is_tensor_input.reserve(inputs_size);
  PyObject *need_grad_input = PyTuple_New(inputs_size);
  // Convert input object to tensors.
  TensorPtrSet input_base_tensors;
  input_base_tensors.reserve(inputs_size);
  for (size_t i = 0; i < inputs_size; ++i) {
    const auto tensor = tensor::ConvertToTensor(PyTuple_GetItem(inputs, i));
    if (tensor != nullptr) {
      (void)is_tensor_input.emplace_back(true);
      tensor->set_need_pipeline_sync(true);
      PyObject *grad_flag = AutoGradUtil::NeedGrad(tensor) ? Py_True : Py_False;
      Py_INCREF(grad_flag);
      PyTuple_SET_ITEM(need_grad_input, i, grad_flag);
      (void)context->inputs.emplace_back(tensor);
      (void)context->input_value_grad_type.emplace_back(AutoGradUtil::SetValueGradInfo(tensor, InputType::kConstant));
      input_base_tensors.insert(tensor);
    } else {
      (void)is_tensor_input.emplace_back(false);
      Py_INCREF(Py_False);
      PyTuple_SET_ITEM(need_grad_input, i, Py_False);
      (void)context->inputs.emplace_back(kNone);
    }
  }
  ctx->is_tensor_input = is_tensor_input;
  ctx->needs_input_grad = need_grad_input;
  context->input_base_tensors = input_base_tensors;
}
}  // namespace

const char CUSTOM_FORWARD_NAME[] = "forward";
const char CUSTOM_BACKWARD_NAME[] = "backward";

static TensorPtrSet parse_mark_dirty(FunctionBase *fptr) {
  // versions of modified tensors should be increased.
  TensorPtrSet dirty;
  PyObject *dirty_tensors = fptr->dirty_tensors;
  if (!dirty_tensors) {
    return dirty;
  }
  if (!PyTuple_Check(dirty_tensors)) {
    MS_LOG(EXCEPTION) << "dirty_tensors of functionbase should be a tuple, but get a "
                      << Py_TYPE(dirty_tensors)->tp_name;
  }

  size_t num_dirty = static_cast<size_t>(PyTuple_GET_SIZE(dirty_tensors));
  for (size_t i = 0; i < num_dirty; i++) {
    PyObject *elem = PyTuple_GetItem(dirty_tensors, i);
    if (!tensor::IsTensorPy(elem)) {
      MS_LOG(EXCEPTION) << "element of dirty_tensors should be a tensor, but get a " << Py_TYPE(elem)->tp_name;
    }
    auto base_tensor = tensor::ConvertToTensor(elem);
    MS_EXCEPTION_IF_NULL(base_tensor);
    dirty.insert(base_tensor);
    base_tensor->BumpVersion();
  }
  Py_DECREF(dirty_tensors);
  fptr->dirty_tensors = Py_None;
  Py_INCREF(Py_None);
  return dirty;
}

static TensorPtrSet parse_non_differentiable(FunctionBase *fptr) {
  TensorPtrSet non_diff;
  PyObject *non_diff_obj = fptr->non_differentiable;
  if (!non_diff_obj) {
    return non_diff;
  }
  if (!PyTuple_Check(non_diff_obj)) {
    MS_LOG(EXCEPTION) << "non_differentiable of functionbase should be a tuple, but get a "
                      << Py_TYPE(non_diff_obj)->tp_name;
  }
  size_t num_non_diff = static_cast<size_t>(PyTuple_GET_SIZE(non_diff_obj));
  for (size_t i = 0; i < num_non_diff; i++) {
    PyObject *elem = PyTuple_GetItem(non_diff_obj, i);
    if (!tensor::IsTensorPy(elem)) {
      MS_LOG(EXCEPTION) << "element of non_differentiable should be a tensor, but get a " << Py_TYPE(elem)->tp_name;
    }
    auto base_tensor = tensor::ConvertToTensor(elem);
    MS_EXCEPTION_IF_NULL(base_tensor);
    non_diff.insert(base_tensor);
  }
  Py_DECREF(non_diff_obj);
  fptr->non_differentiable = Py_None;
  Py_INCREF(Py_None);
  return non_diff;
}

static TensorPtrList parse_to_save(const std::shared_ptr<FunctionContext> &context, bool need_do_grad,
                                   FunctionBase *fptr) {
  PyObject *to_save_obj = fptr->saved_tensors;
  if (!to_save_obj) {
    return {};
  }
  auto check_is_output = [&context](const ValuePtr &val) {
    return std::any_of(context->flatten_outputs.begin(), context->flatten_outputs.end(),
                       [&val](const ValuePtr &output) { return val.get() == output.get(); });
  };

  if (!PyTuple_Check(to_save_obj)) {
    MS_LOG(EXCEPTION) << "saved_tensors of functionbase should be a tuple, but get a " << Py_TYPE(to_save_obj)->tp_name;
  }

  bool has_saved_tensors_hooks = DefaultSavedTensorHookUtil::GetTopHook() != nullptr;
  size_t num_to_save = static_cast<size_t>(PyTuple_GET_SIZE(to_save_obj));
  TensorPtrList to_save_tensors;
  to_save_tensors.reserve(num_to_save);
  PyObject *py_data = PyTuple_New(num_to_save);
  for (size_t i = 0; i < num_to_save; i++) {
    TensorPtr tensor = nullptr;
    PyObject *elem = PyTuple_GET_ITEM(to_save_obj, i);
    if (tensor::IsPyObjectTensorPy(elem)) {
      tensor = tensor::ConvertToTensor(elem);
      if (check_is_output(tensor) || has_saved_tensors_hooks) {
        (void)to_save_tensors.emplace_back(tensor);
        continue;
      }
    } else if (elem != Py_None && need_do_grad) {
      Py_DECREF(py_data);
      MS_LOG(EXCEPTION) << "Please check your custom function, that save_for_backward() "
                           "only support None and tensor, but got other type!";
    }
    Py_INCREF(elem);
    PyTuple_SetItem(py_data, i, elem);
    to_save_tensors.emplace_back(tensor);
  }
  Py_DECREF(to_save_obj);
  fptr->saved_tensors = py_data;
  return to_save_tensors;
}

void UpdateTensorSetIfNeeded(const std::shared_ptr<FunctionContext> &context, tensor::TensorPtr old_value,
                             tensor::TensorPtr new_value) {
  if (context->input_base_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update input old: " << old_value << " new: " << new_value;
    context->input_base_tensors.erase(old_value);
    context->input_base_tensors.insert(new_value);
  }
  if (context->dirty_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update dirty old: " << old_value << " new: " << new_value;
    context->dirty_tensors.erase(old_value);
    context->dirty_tensors.insert(new_value);
  }
  if (context->non_diff_tensors.count(old_value) > 0) {
    MS_LOG(DEBUG) << "update non_diff old: " << old_value << " new: " << new_value;
    context->non_diff_tensors.erase(old_value);
    context->non_diff_tensors.insert(new_value);
  }
}

void CleanBackwardUnusedTensorDeviceAddress(const std::shared_ptr<FunctionContext> &context) {
  std::unordered_map<tensor::TensorPtr, tensor::TensorPtr> changed;
  std::unordered_set<ValuePtr> save_tensors_set(context->to_save_tensors.begin(), context->to_save_tensors.end());
  for (size_t i = 0; i < context->inputs.size(); i++) {
    if (context->inputs[i]->isa<tensor::Tensor>()) {
      auto base_tensor = context->inputs[i]->cast<tensor::TensorPtr>();
      if (save_tensors_set.count(context->inputs[i]) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::TensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::TensorPtr>());
        context->inputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean input tensor address, index: " << i;
      }
    }
  }
  for (size_t i = 0; i < context->flatten_outputs.size(); i++) {
    if (context->flatten_outputs[i]->isa<tensor::Tensor>()) {
      auto base_tensor = context->flatten_outputs[i]->cast<tensor::TensorPtr>();
      if (save_tensors_set.count(base_tensor) == 0) {
        ValuePtr fake_value;
        if (changed.count(base_tensor) == 0) {
          fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(base_tensor);
          changed.emplace(base_tensor, fake_value->cast<tensor::TensorPtr>());
        } else {
          fake_value = changed[base_tensor];
        }
        UpdateTensorSetIfNeeded(context, base_tensor, fake_value->cast<tensor::TensorPtr>());
        context->flatten_outputs[i] = fake_value;
        MS_LOG(DEBUG) << "clean output tensor address, index: " << i;
      }
    }
  }
}

void ConstructContextAfterForward(const std::shared_ptr<FunctionContext> &context, FunctionBase *ctx,
                                  const py::object &outputs, bool need_do_grad) {
  // Convert output object to tensors.
  context->flatten_outputs = ConvertOutputTensorList(outputs);
  MS_LOG(DEBUG) << "function base info, has dirty_tensors: " << static_cast<bool>(ctx->dirty_tensors)
                << "has non_differentiable" << static_cast<bool>(ctx->non_differentiable);
  // Convert object use decided to tensors.
  context->dirty_tensors = parse_mark_dirty(ctx);
  context->non_diff_tensors = parse_non_differentiable(ctx);
  context->to_save_tensors = parse_to_save(context, need_do_grad, ctx);
  MS_LOG(DEBUG) << "Parse info, dirty size: " << context->dirty_tensors.size()
                << ", non_diff size: " << context->non_diff_tensors.size()
                << "saved_tensors size: " << context->to_save_tensors.size();
}

PyObject *FunctionBase_saved_tensors(FunctionBase *self, void *unused) {
  HANDLE_MS_EXCEPTION
  // forward
  auto grad_node = self->weak_grad_node.lock();
  if (grad_node == nullptr) {
    if (self->saved_tensors) {
      Py_INCREF(self->saved_tensors);
      return self->saved_tensors;
    }
    return PyTuple_New(0);
  }
  auto saved_tensors = grad_node->GetSavedTensors();
  PyObject *res = PyTuple_New(saved_tensors.size());
  for (size_t i = 0; i < saved_tensors.size(); i++) {
    const auto &saved_tensor = saved_tensors[i];
    if (saved_tensor == nullptr) {
      Py_INCREF(Py_None);
      PyTuple_SetItem(res, i, Py_None);
      continue;
    }
    PyObject *elem = PyTuple_GetItem(self->saved_tensors, i);
    if (elem != nullptr) {  // should not be replaced
      saved_tensor->UnWrap(grad_node);
      Py_INCREF(elem);
      PyTuple_SetItem(res, i, elem);
    } else {
      PyTuple_SetItem(res, i, tensor::Wrap(saved_tensor->UnWrapToTensor(grad_node)));
    }
  }
  return res;
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_apply(PyObject *cls, PyObject *inputs) {
  HANDLE_MS_EXCEPTION
  MS_LOG(DEBUG) << "enter apply function.";
  auto context = std::make_shared<FunctionContext>();
  auto forward_fn = py::reinterpret_steal<py::object>(PyObject_GetAttrString(cls, CUSTOM_FORWARD_NAME));
  auto backward_fn = py::reinterpret_steal<py::object>(PyObject_GetAttrString(cls, CUSTOM_BACKWARD_NAME));
  // New a python object.
  auto ctx_obj = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(cls, nullptr));
  auto ctx = reinterpret_cast<FunctionBase *>(ctx_obj.ptr());
  MS_EXCEPTION_IF_NULL(ctx);
  Py_ssize_t args_size = PyTuple_GET_SIZE(inputs);
  context->inputs.reserve(args_size);
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();  // wait to get inputs value
  ProcessInputs(context, ctx, inputs);
  std::string type_name(((PyTypeObject *)cls)->tp_name);
  const auto custom_fn = BackwardNode::Create<PyBackwardNode>(std::move(type_name), backward_fn, ctx_obj);
  ctx->weak_grad_node = custom_fn;
  UpdateNextEdges(custom_fn, context->inputs);
  // Get need grad before forward.
  bool need_do_grad = GradState::Get().RequiresGrad() && AutoGradUtil::NeedGrad(context->inputs);
  // Call forward function.
  py::object outputs;
  {
    NoGradGuard no_grad;
    auto forward_inputs = py::reinterpret_steal<py::object>(PackForwardInput(ctx_obj.ptr(), inputs));
    outputs = py::reinterpret_steal<py::object>(PyObject_CallObject(forward_fn.ptr(), forward_inputs.ptr()));
    if (!outputs) {
      return nullptr;
    }
  }
  bool modified = ensure_obj_tuple(&outputs);

  runtime::Pipeline::Get().WaitFrontend();
  ConstructContextAfterForward(context, ctx, outputs, need_do_grad);
  custom_fn->SetOutputSize(context->flatten_outputs.size());
  context->grad_node = custom_fn;
  auto &flatten_outputs = context->flatten_outputs;
  const auto &non_diff_tensors = context->non_diff_tensors;
  const auto &input_tensor_set = context->input_base_tensors;
  const auto &dirty_tensor_set = context->dirty_tensors;

  size_t num_output = py::cast<py::tuple>(outputs).size();
  py::tuple output_ret(num_output);
  MS_LOG(DEBUG) << "Output info, modified: " << modified << ", num_output: " << num_output;
  for (size_t i = 0; i < num_output; ++i) {
    if (!flatten_outputs[i]->isa<tensor::Tensor>()) {
      output_ret[i] = py::cast<py::tuple>(outputs)[i];
      continue;
    }
    auto tensor = flatten_outputs[i]->cast<tensor::TensorPtr>();
    bool is_diff = non_diff_tensors.count(tensor) == 0 && need_do_grad;
    bool is_same_as_input = input_tensor_set.count(tensor) > 0;
    bool is_dirty_tensor = dirty_tensor_set.count(tensor) > 0;
    if (is_diff) {
      if (is_same_as_input && !is_dirty_tensor) {
        tensor = ViewAsSelfWithNoGrad(tensor);
        flatten_outputs[i] = tensor;
      }
      AutoGradUtil::SetValueGradInfo(tensor, InputType::kOpOutput);
      output_ret[i] = CValueToPybindObj(tensor);
      continue;
    }
    if (!tensor->requires_grad()) {
      if (is_same_as_input && !is_dirty_tensor) {
        tensor = ViewAsSelfWithNoGrad(tensor);
        flatten_outputs[i] = tensor;
      }
    } else if (is_same_as_input) {
      tensor = std::make_shared<Tensor>(*tensor);
      tensor->set_auto_grad_meta_data(nullptr);
      flatten_outputs[i] = tensor;
    } else if (impl::GetViewAutogradMetaImpl(tensor) == nullptr) {
      tensor->set_auto_grad_meta_data(nullptr);
    }
    output_ret[i] = CValueToPybindObj(tensor);
  }
  if (!need_do_grad) {
    MS_LOG(DEBUG) << "no need to do grad.";
    if (modified) {
      py::object output = output_ret[0];
      return output.release().ptr();
    }
    return output_ret.release().ptr();
  }

  // Clean device address to reduce the occupation of resources.
  CleanBackwardUnusedTensorDeviceAddress(context);

  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &grad_executor = pynative_executor->grad_executor();
  if (forward_executor->enable_async()) {
    auto task = [new_context = std::move(context)]() mutable { CallCustomPyFunction(new_context); };
    grad_executor->DispatchGradQueueTask(std::move(task));
  } else {
    CallCustomPyFunction(context);
  }

  MS_LOG(DEBUG) << "Leave apply function.";
  if (modified) {
    py::object output = output_ret[0];
    return output.release().ptr();
  }
  return output_ret.release().ptr();
  HANDLE_MS_EXCEPTION_END
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore

namespace mindspore::pynative::autograd {
namespace py = pybind11;
namespace {
void FunctionBase_clear(FunctionBase *self) {
  Py_CLEAR(self->saved_tensors);
  Py_CLEAR(self->dirty_tensors);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->saved_tensors);
  self->is_tensor_input.clear();
}

static void FunctionBase_PyDealloc(FunctionBase *self) {
  FunctionBase_clear(self);
  self->weak_grad_node.~weak_ptr<PyBackwardNode>();
  self->is_tensor_input.~vector();
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *FunctionBase_str(FunctionBase *self) {
  HANDLE_MS_EXCEPTION
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return PyUnicode_FromFormat("<%s, seq=%zu>", backward_node->name().c_str(), backward_node->seq_id());
  HANDLE_MS_EXCEPTION_END
}

static PyObject *FunctionBase_repr(FunctionBase *self) { return FunctionBase_str(self); }

PyObject *FunctionBase_get_next_edges(FunctionBase *self, void *) {
  HANDLE_MS_EXCEPTION
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return BackwardNode_get_next_edges(backward_node);
  HANDLE_MS_EXCEPTION_END
}

int FunctionBase_set_materialize_grads(FunctionBase *self, PyObject *value, void *unused) {
  HANDLE_MS_EXCEPTION
  MS_EXCEPTION_IF_NULL(self);
  MS_EXCEPTION_IF_NULL(value);
  if (!PyBool_Check(value)) {
    MS_LOG(EXCEPTION) << "set materialize_grads value should be bool! but got " << Py_TYPE(value)->tp_name;
  }
  self->materialize_grads = (value == Py_True);
  return 0;
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

template <PyObject *FunctionBase::*ptr>
PyObject *get_property(PyObject *obj, void *unused) {
  auto self = reinterpret_cast<FunctionBase *>(obj);
  PyObject *val = self->*ptr;
  if (!val) {
    Py_RETURN_NONE;
  }
  Py_INCREF(val);
  return val;
}

template <PyObject *FunctionBase::*ptr>
int set_property(PyObject *obj, PyObject *value, void *unused) {
  auto self = reinterpret_cast<FunctionBase *>(obj);
  if (value == Py_None) {
    value = nullptr;
  }
  Py_XDECREF(self->*ptr);
  Py_XINCREF(value);
  self->*ptr = value;
  return 0;
}

PyGetSetDef FunctionBase_getseters[] = {
  {"next_functions", (getter)FunctionBase_get_next_edges, nullptr, "FunctionBase node next edges", nullptr},
  {"saved_tensors", (getter)FunctionBase_saved_tensors, &set_property<&FunctionBase::saved_tensors>,
   "FunctionBase node saved tensors", nullptr},
  {"needs_input_grad", &get_property<&FunctionBase::needs_input_grad>, &set_property<&FunctionBase::needs_input_grad>,
   nullptr, nullptr},
  {"non_differentiable", &get_property<&FunctionBase::non_differentiable>,
   &set_property<&FunctionBase::non_differentiable>, nullptr, nullptr},
  {"dirty_tensors", &get_property<&FunctionBase::dirty_tensors>, &set_property<&FunctionBase::dirty_tensors>, nullptr,
   nullptr},
  {"materialize_grads", nullptr, (setter)FunctionBase_set_materialize_grads, nullptr, nullptr},
  {nullptr}};

PyObject *FunctionBase_name(FunctionBase *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return PyUnicode_FromString(backward_node->name().c_str());
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_is_leaf(FunctionBase *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return PyBool_FromLong(backward_node->IsLeaf());
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_seq_nr(FunctionBase *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return PyLong_FromSize_t(backward_node->seq_id());
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_register_pre_hook(FunctionBase *self, PyObject *arg) {
  HANDLE_MS_EXCEPTION
  if (!PyCallable_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, "hook_fn must be callable");
    return nullptr;
  }
  auto backward_node = self->weak_grad_node.lock();
  MS_EXCEPTION_IF_NULL(backward_node);
  return BackwardNode_register_pre_hook(backward_node, arg);
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_register_post_hook(FunctionBase *self, PyObject *arg) {
  HANDLE_MS_EXCEPTION
  if (!PyCallable_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, "hook_fn must be callable");
    return nullptr;
  }
  auto backward_node = self->weak_grad_node.lock();
  return BackwardNode_register_post_hook(backward_node, arg);
  HANDLE_MS_EXCEPTION_END
}

PyObject *FunctionBase_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  PyObject *obj = type->tp_alloc(type, 0);
  if (!obj) {
    return nullptr;
  }
  FunctionBase *self = (FunctionBase *)obj;
  new (&self->weak_grad_node) std::weak_ptr<PyBackwardNode>;
  self->materialize_grads = true;
  return obj;
}

PyMethodDef FunctionBase_methods[] = {
  {"apply", (PyCFunction)FunctionBase_apply, METH_CLASS | METH_VARARGS, "apply py function"},
  {"_sequence_nr", (PyCFunction)FunctionBase_seq_nr, METH_NOARGS, "backward node sequence number"},
  {"name", (PyCFunction)FunctionBase_name, METH_NOARGS, "backward node name"},
  {"is_leaf", (PyCFunction)FunctionBase_is_leaf, METH_NOARGS, "whether backward node is a leaf node"},
  {"register_prehook", (PyCFunction)FunctionBase_register_pre_hook, METH_O, "register post hook on BackwardNode"},
  {"register_hook", (PyCFunction)FunctionBase_register_post_hook, METH_O, "register post hook on BackwardNode"},
  {nullptr, nullptr, 0, nullptr}};

PyTypeObject FunctionBasePyType = {
  PyVarObject_HEAD_INIT(nullptr, 0) "FunctionBase", /* tp_name */
  sizeof(FunctionBase),                             /* tp_basicsize */
  0,                                                /* tp_itemsize */
  (destructor)FunctionBase_PyDealloc,               /* tp_dealloc */
  0,                                                /* tp_vectorcall_offset */
  nullptr,                                          /* tp_getattr */
  nullptr,                                          /* tp_setattr */
  nullptr,                                          /* tp_reserved */
  (reprfunc)FunctionBase_repr,                      /* tp_repr */
  nullptr,                                          /* tp_as_number */
  nullptr,                                          /* tp_as_sequence */
  nullptr,                                          /* tp_as_mapping */
  nullptr,                                          /* tp_hash  */
  nullptr,                                          /* tp_call */
  (reprfunc)FunctionBase_str,                       /* tp_str */
  nullptr,                                          /* tp_getattro */
  nullptr,                                          /* tp_setattro */
  nullptr,                                          /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,         /* tp_flags */
  nullptr,                                          /* tp_doc */
  nullptr,                                          /* tp_traverse */
  nullptr,                                          /* tp_clear */
  nullptr,                                          /* tp_richcompare */
  0,                                                /* tp_weaklistoffset */
  nullptr,                                          /* tp_iter */
  nullptr,                                          /* tp_iternext */
  FunctionBase_methods,                             /* tp_methods */
  nullptr,                                          /* tp_members */
  FunctionBase_getseters,                           /* tp_getset */
  nullptr,                                          /* tp_base */
  nullptr,                                          /* tp_dict */
  nullptr,                                          /* tp_descr_get */
  nullptr,                                          /* tp_descr_set */
  0,                                                /* tp_dictoffset */
  nullptr,                                          /* tp_init */
  nullptr,                                          /* tp_alloc */
  FunctionBase_new,                                 /* tp_new */
};
}  // namespace

namespace py = pybind11;
void RegFunctionBase(py::module *m) {
  if (PyType_Ready(&FunctionBasePyType) < 0) {
    return;
  }
  Py_INCREF(&FunctionBasePyType);
  m->add_object("FunctionBase", reinterpret_cast<PyObject *>(&FunctionBasePyType));
}
}  // namespace mindspore::pynative::autograd
