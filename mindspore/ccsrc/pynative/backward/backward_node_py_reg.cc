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

#include <cstdint>
#include <memory>
#include "pynative/backward/tensor_meta_py.h"
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

PyObject *PyBackwardNode_get_tensor_meta(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  const auto &metadata = backward_node->mutable_metadata();
  PyObject *output = PyTuple_New(metadata.size());
  for (size_t i = 0; i < metadata.size(); i++) {
    PyObject *tensor_meta_obj = WrapTensorMeta(metadata[i]);
    if (tensor_meta_obj == nullptr) {
      Py_DECREF(output);
      return nullptr;
    }
    PyTuple_SetItem(output, i, tensor_meta_obj);
  }
  return output;
  HANDLE_MS_EXCEPTION_END
}

PyGetSetDef PyBackwardNode_getseters[] = {
  {"next_functions", (getter)PyBackwardNode_get_next_edges, nullptr, "backward node next edges", nullptr},
  {"_input_metadata", (getter)PyBackwardNode_get_tensor_meta, nullptr, "backward node tensor metadata", nullptr},
  {nullptr}};

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

PyObject *PyBackwardNode_unique_id(PyObject *self, PyObject *) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  auto raw_ptr = reinterpret_cast<uintptr_t>(backward_node.get());
  return PyLong_FromUnsignedLongLong(static_cast<uint64_t>(raw_ptr));
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

PyObject *PyBackwardNode_set_next_edge(PyObject *self, PyObject *args) {
  HANDLE_MS_EXCEPTION
  const auto &backward_node = reinterpret_cast<BackwardNodePy *>(self)->cdata;
  return BackwardNode_set_next_edge(backward_node, args);
  HANDLE_MS_EXCEPTION_END
}

PyMethodDef PyBackwardNode_methods[] = {
  {"_sequence_nr", (PyCFunction)PyBackwardNode_seq_nr, METH_NOARGS, "backward node sequence number"},
  {"_unique_id", (PyCFunction)PyBackwardNode_unique_id, METH_NOARGS, "backward node unique id"},
  {"name", (PyCFunction)PyBackwardNode_name, METH_NOARGS, "backward node name"},
  {"is_leaf", (PyCFunction)PyBackwardNode_is_leaf, METH_NOARGS, "whether backward node is a leaf node"},
  {"register_prehook", (PyCFunction)PyBackwardNode_register_pre_hook, METH_O, "register post hook on BackwardNode"},
  {"register_hook", (PyCFunction)PyBackwardNode_register_post_hook, METH_O, "register post hook on BackwardNode"},
  {"_set_next_edge", (PyCFunction)PyBackwardNode_set_next_edge, METH_VARARGS,
   "set an entry of next_edges by index; pass None to clear the edge"},
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

bool IsBackwardNodePy(const py::handle &obj) {
  if (!obj.check()) {
    return false;
  }
  PyObject *raw_ptr = obj.ptr();
  return PyObject_TypeCheck(raw_ptr, &BackwardNodePyType);
}

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

PyObject *BackwardNode_set_next_edge(const BackwardNodePtr &backward_node, PyObject *args) {
  Py_ssize_t edge_index = 0;
  PyObject *edge_obj = Py_None;
  if (!PyArg_ParseTuple(args, "n|O", &edge_index, &edge_obj)) {
    return nullptr;
  }
  if (edge_index < 0) {
    PyErr_SetString(PyExc_ValueError, "edge_index must be non-negative");
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(backward_node);
  const auto &next_edges = backward_node->next_edges();
  if (static_cast<size_t>(edge_index) >= next_edges.size()) {
    PyErr_SetString(PyExc_IndexError, "edge_index out of range");
    return nullptr;
  }

  // Default: clear edge.
  if (edge_obj == Py_None) {
    backward_node->set_next_edge(Edge(), static_cast<size_t>(edge_index));
    Py_RETURN_NONE;
  }

  // Optional: allow setting a specific edge: (grad_node, input_index).
  constexpr Py_ssize_t kEdgeTupleSize = 2;
  if (!PyTuple_Check(edge_obj) || PyTuple_Size(edge_obj) != kEdgeTupleSize) {
    PyErr_SetString(PyExc_TypeError, "edge must be None or a tuple(grad_node, input_index)");
    return nullptr;
  }

  PyObject *grad_node_obj = PyTuple_GetItem(edge_obj, 0);
  PyObject *input_index_obj = PyTuple_GetItem(edge_obj, 1);
  if (grad_node_obj == nullptr || input_index_obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse edge tuple");
    return nullptr;
  }

  size_t input_index = PyLong_AsSize_t(input_index_obj);
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "input_index must be an int");
    return nullptr;
  }

  BackwardNodePtr grad_node_ptr = nullptr;
  if (grad_node_obj != Py_None) {
    py::handle h(grad_node_obj);
    if (!IsBackwardNodePy(h)) {
      PyErr_SetString(PyExc_TypeError, "grad_node must be a BackwardNode or None");
      return nullptr;
    }
    grad_node_ptr = reinterpret_cast<BackwardNodePy *>(grad_node_obj)->cdata;
  }

  backward_node->set_next_edge(Edge(std::move(grad_node_ptr), input_index), static_cast<size_t>(edge_index));
  Py_RETURN_NONE;
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
