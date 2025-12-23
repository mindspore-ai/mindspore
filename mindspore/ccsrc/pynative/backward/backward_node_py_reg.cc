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

#include <memory>
#include "pynative/backward/backward_node_py.h"
#include "include/utils/exception.h"
#include "include/utils/pyobj_manager.h"
#include "pynative/backward/hook/custom_function.h"
#include "pybind_api/pynative/pynative_api.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;
namespace {
void BackwardNode_PyDealloc(PyObject *self) {
  reinterpret_cast<BackwardNodePy *>(self)->cdata.reset();
  reinterpret_cast<BackwardNodePy *>(self)->cdata.~shared_ptr();
  Py_TYPE(self)->tp_free(self);
}

PyObject *PyBackwardNode_str(PyObject *self) {
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return PyUnicode_FromFormat("<%s, seq=%zu>", backward_node->name().c_str(), backward_node->seq_id());
}

PyObject *PyBackwardNode_repr(PyObject *self) { return PyBackwardNode_str(self); }

PyObject *PyBackwardNode_get_next_edges(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return BackwardNode_get_next_edges(backward_node);
  HANDLE_MS_EXCEPTION_END
}

PyGetSetDef PyBackwardNode_getseters[] = {
  {"next_functions", (getter)PyBackwardNode_get_next_edges, nullptr, "backward node next edges", nullptr}, {nullptr}};

PyObject *PyBackwardNode_name(PyObject *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return PyUnicode_FromString(backward_node->name().c_str());
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyBackwardNode_is_leaf(PyObject *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return PyBool_FromLong(backward_node->IsLeaf());
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyBackwardNode_seq_nr(PyObject *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return PyLong_FromSize_t(backward_node->seq_id());
  HANDLE_MS_EXCEPTION_END
}

PyObject *RunRegisterHookFn(PyObject *hook_dict, PyObject *hook_fn) {
  PyObject *hook_utils_class = PyObjManager::Get().GetHookUtilsClass();
  PyObject *register_hook_fn = PyObject_GetAttrString(hook_utils_class, "register_hook");
  PyObject *res = PyObject_CallFunctionObjArgs(register_hook_fn, hook_dict, hook_fn, nullptr);
  Py_XDECREF(register_hook_fn);
  return res;
}

PyObject *PyBackwardNode_register_pre_hook(PyObject *self, PyObject *arg) {
  HANDLE_MS_EXCEPTION
  if (!PyCallable_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, "hook_fn must be callable");
    return nullptr;
  }
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return BackwardNode_register_pre_hook(backward_node, arg);
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyBackwardNode_register_post_hook(PyObject *self, PyObject *arg) {
  HANDLE_MS_EXCEPTION
  if (!PyCallable_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, "hook_fn must be callable");
    return nullptr;
  }
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return BackwardNode_register_post_hook(backward_node, arg);
  HANDLE_MS_EXCEPTION_END
}

PyMethodDef PyBackwardNode_methods[] = {
  {"_sequence_nr", (PyCFunction)PyBackwardNode_seq_nr, METH_NOARGS, "backward node sequence number"},
  {"name", (PyCFunction)PyBackwardNode_name, METH_NOARGS, "backward node name"},
  {"is_leaf", (PyCFunction)PyBackwardNode_is_leaf, METH_NOARGS, "whether backward node is a leaf node"},
  {"register_prehook", (PyCFunction)PyBackwardNode_register_pre_hook, METH_O, "register post hook on BackwardNode"},
  {"register_hook", (PyCFunction)PyBackwardNode_register_post_hook, METH_O, "register post hook on BackwardNode"},
  {nullptr, nullptr, 0, nullptr}};

PyTypeObject BackwardNodePyType = {
  PyVarObject_HEAD_INIT(nullptr, 0) "BackwardNode", /* tp_name */
  sizeof(BackwardNodePy),                           /* tp_basicsize */
  0,                                                /* tp_itemsize */
  BackwardNode_PyDealloc,                           /* tp_dealloc */
  0,                                                /* tp_vectorcall_offset */
  nullptr,                                          /* tp_getattr */
  nullptr,                                          /* tp_setattr */
  nullptr,                                          /* tp_reserved */
  PyBackwardNode_repr,                              /* tp_repr */
  nullptr,                                          /* tp_as_number */
  nullptr,                                          /* tp_as_sequence */
  nullptr,                                          /* tp_as_mapping */
  nullptr,                                          /* tp_hash  */
  nullptr,                                          /* tp_call */
  PyBackwardNode_str,                               /* tp_str */
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
  PyBackwardNode_methods,                           /* tp_methods */
  nullptr,                                          /* tp_members */
  PyBackwardNode_getseters,                         /* tp_getset */
  nullptr,                                          /* tp_base */
  nullptr,                                          /* tp_dict */
  nullptr,                                          /* tp_descr_get */
  nullptr,                                          /* tp_descr_set */
  0,                                                /* tp_dictoffset */
  nullptr,                                          /* tp_init */
  nullptr,                                          /* tp_alloc */
  nullptr,                                          /* tp_new */
};
}  // namespace

PyObject *Wrap(const BackwardNodePtr &backward_node) {
  if (!backward_node) {
    Py_RETURN_NONE;
  }
  if (autograd::isa<PyBackwardNode>(backward_node)) {
    auto py_node = std::static_pointer_cast<PyBackwardNode>(backward_node);
    if (!py_node->obj()) {
      MS_LOG(EXCEPTION) << "Try to get grad node of custom function which ctx has been freed!";
    }
    Py_INCREF(py_node->obj().ptr());
    return py_node->obj().ptr();
  }
  PyTypeObject *type = &BackwardNodePyType;
  PyObject *obj = type->tp_alloc(type, 0);
  auto bn = reinterpret_cast<BackwardNodePy *>(obj);
  if (bn == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create BackwardNodePy object");
    return nullptr;
  }
  new (&bn->cdata) BackwardNodePtr(backward_node);
  return obj;
}

PyObject *BackwardNode_get_next_edges(const BackwardNodePtr &backward_node) {
  const auto &next_edges = backward_node->next_edges();
  PyObject *output = PyTuple_New(next_edges.size());
  for (size_t i = 0; i < next_edges.size(); i++) {
    const auto &edge = next_edges[i];
    PyObject *item = PyTuple_New(2);
    PyTuple_SetItem(item, 0, Wrap(edge.grad_node));
    PyTuple_SetItem(item, 1, PyLong_FromSize_t(edge.input_index));
    PyTuple_SetItem(output, i, item);
  }
  return output;
}

PyObject *BackwardNode_register_pre_hook(const BackwardNodePtr &backward_node, PyObject *hook_fn) {
  PyObject *hook_dict = Py_None;
  if (const auto &py_pre_hook = backward_node->py_pre_hook()) {
    hook_dict = py_pre_hook->hook_dict_;
  }
  PyObject *result = RunRegisterHookFn(hook_dict, hook_fn);

  if (hook_dict == Py_None) {
    hook_dict = PyTuple_GetItem(result, 0);
    backward_node->SetPyPreHook(std::make_unique<PyBackwardNodePreHook>(hook_dict));
  }

  PyObject *hook_handle = PyTuple_GetItem(result, 1);
  Py_INCREF(hook_handle);
  Py_DECREF(result);
  return hook_handle;
}

PyObject *BackwardNode_register_post_hook(const BackwardNodePtr &backward_node, PyObject *hook_fn) {
  PyObject *hook_dict = Py_None;
  if (const auto &py_post_hook = backward_node->py_post_hook()) {
    hook_dict = py_post_hook->hook_dict_;
  }
  PyObject *result = RunRegisterHookFn(hook_dict, hook_fn);

  if (hook_dict == Py_None) {
    hook_dict = PyTuple_GetItem(result, 0);
    backward_node->SetPyPostHook(std::make_unique<PyBackwardNodePostHook>(hook_dict));
  }

  PyObject *hook_handle = PyTuple_GetItem(result, 1);
  Py_INCREF(hook_handle);
  Py_DECREF(result);
  return hook_handle;
}

namespace py = pybind11;
void RegBackwardNode(py::module *m) {
  if (PyType_Ready(&BackwardNodePyType) < 0) {
    return;
  }
  Py_INCREF(&BackwardNodePyType);
  m->add_object("BackwardNodePy", reinterpret_cast<PyObject *>(&BackwardNodePyType));
}
}  // namespace mindspore::pynative::autograd
