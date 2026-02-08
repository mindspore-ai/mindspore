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

#include "include/utils/pynative/hook.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/pyobj_manager.h"
#include "include/utils/tensor_py.h"

namespace mindspore::pynative::autograd {
namespace {
PyObject *RunHookDict(PyObject *hook_dict, PyObject *hook_args) {
  PyObject *hook_utils_class = PyObjManager::Get().GetHookUtilsClass();
  PyObject *run_hook_fn = PyObject_GetAttrString(hook_utils_class, "run_hook");
  PyObject *res = PyObject_CallFunctionObjArgs(run_hook_fn, hook_dict, hook_args, nullptr);
  Py_XDECREF(run_hook_fn);
  return res;
}
}  // namespace

CppTensorBackwardNodePreHook::CppTensorBackwardNodePreHook(CppHookFn hook_fn, size_t output_idx)
    : hook_fn_(std::move(hook_fn)), output_idx_(output_idx) {}

void CppTensorBackwardNodePreHook::operator()(ValuePtrList *grad) {
  if (output_idx_ >= grad->size()) {
    MS_LOG(EXCEPTION) << "CppTensor hook output_idx out of range";
  }
  const auto grad_in = (*grad)[output_idx_];
  if (!grad_in->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "input grad is not a Tensor";
  } else {
    (*grad)[output_idx_] = hook_fn_(grad_in->cast<tensor::TensorPtr>());
  }
}

PyTensorBackwardNodePreHook::PyTensorBackwardNodePreHook(const py::function &hook_fn, size_t output_idx)
    : hook_fn_(hook_fn), output_idx_(output_idx) {}

PyTensorBackwardNodePreHook::~PyTensorBackwardNodePreHook() {
  py::gil_scoped_acquire gil;
  hook_fn_ = py::object();
}

void PyTensorBackwardNodePreHook::operator()(ValuePtrList *grad_outputs) {
  if (output_idx_ >= grad_outputs->size()) {
    MS_LOG(EXCEPTION) << "PyTensor hook output_idx out of range";
  }

  py::gil_scoped_acquire gil;
  const auto py_grad = CValueToPybindObj((*grad_outputs)[output_idx_]);
  const auto ret = hook_fn_(py_grad);
  if (!ret.is_none()) {
    if (tensor::IsTensorPy(ret)) {
      (*grad_outputs)[output_idx_] = tensor::ConvertToTensor(ret);
    } else {
      MS_LOG(EXCEPTION) << "Tensor hook should be return Tensor, but get type: "
                        << py::str(ret.get_type().attr("__name__")).cast<std::string>() << ".";
    }
  }
}

PyBackwardNodePreHook::PyBackwardNodePreHook(PyObject *hook_dict) : hook_dict_(hook_dict) { Py_INCREF(hook_dict); }

PyBackwardNodePreHook::~PyBackwardNodePreHook() {
  py::gil_scoped_acquire gil;
  Py_DECREF(hook_dict_);
}

void PyBackwardNodePreHook::operator()(ValuePtrList *grad_outputs) {
  py::gil_scoped_acquire gil;
  PyObject *grad_py = tensor::Wrap(*grad_outputs);
  PyObject *hook_args = PyTuple_New(1);
  PyTuple_SetItem(hook_args, 0, grad_py);
  auto res = py::reinterpret_steal<py::object>(RunHookDict(this->hook_dict_, hook_args));
  if (!res.is_none()) {
    grad_outputs->clear();
    ConvertPybindTupleGradToCValue(res.cast<py::tuple>(), grad_outputs);
  }
  Py_DECREF(hook_args);
}

RetainGradHook::RetainGradHook(RetainGradHookFn hook_fn) : hook_fn_(std::move(hook_fn)) {}

void RetainGradHook::operator()(const ValuePtr &grad) {
  if (grad->isa<None>()) {
    return;
  }
  const auto grad_tensor = grad->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad_tensor);
  hook_fn_(grad_tensor);
}

PyBackwardNodePostHook::PyBackwardNodePostHook(PyObject *hook_dict) : hook_dict_(hook_dict) { Py_INCREF(hook_dict); }

PyBackwardNodePostHook::~PyBackwardNodePostHook() {
  py::gil_scoped_acquire gil;
  Py_DECREF(hook_dict_);
}

void PyBackwardNodePostHook::operator()(ValuePtrList *grad_inputs, const ValuePtrList &grad_outputs) {
  py::gil_scoped_acquire gil;
  PyObject *grad_inputs_py = tensor::Wrap(*grad_inputs);
  PyObject *grad_outputs_py = tensor::Wrap(grad_outputs);
  PyObject *hook_args = PyTuple_New(2);
  PyTuple_SetItem(hook_args, 0, grad_inputs_py);
  PyTuple_SetItem(hook_args, 1, grad_outputs_py);
  auto res = py::reinterpret_steal<py::object>(RunHookDict(this->hook_dict_, hook_args));
  if (!res.is_none()) {
    grad_inputs->clear();
    ConvertPybindTupleGradToCValue(res.cast<py::tuple>(), grad_inputs);
  }
  Py_DECREF(hook_args);
}
}  // namespace mindspore::pynative::autograd
