/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/utils/pytree.h"
#include <numeric>
#include <sstream>
#include <utility>
#include "include/utils/tensor_py.h"

namespace py = pybind11;

namespace mindspore {
namespace pytree {
namespace {

// Cached type objects for fast comparison
static PyTypeObject *tuple_type = nullptr;
static PyTypeObject *list_type = nullptr;
static PyTypeObject *dict_type = nullptr;

void EnsureBuiltinTypes() {
  if (tuple_type == nullptr) {
    tuple_type = &PyTuple_Type;
    list_type = &PyList_Type;
    dict_type = &PyDict_Type;
  }
}

// Forward declaration
PyTreeDefPtr FlattenImpl(PyObject *tree, PyObject *is_leaf_fn, PyObject *leaves_list, bool tensors_only_leaf);

// Helper to flatten children from a tuple-like sequence
bool FlattenTupleChildren(PyObject *tree, Py_ssize_t size, PyObject *is_leaf_fn, PyObject *leaves_list,
                          bool tensors_only_leaf, std::vector<PyTreeDefPtr> *children) {
  children->reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *child = PyTuple_GET_ITEM(tree, i);
    auto child_def = FlattenImpl(child, is_leaf_fn, leaves_list, tensors_only_leaf);
    if (child_def == nullptr) {
      return false;
    }
    children->push_back(std::move(child_def));
  }
  return true;
}

// Helper to flatten children from a list
bool FlattenListChildren(PyObject *tree, Py_ssize_t size, PyObject *is_leaf_fn, PyObject *leaves_list,
                         bool tensors_only_leaf, std::vector<PyTreeDefPtr> *children) {
  children->reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *child = PyList_GET_ITEM(tree, i);
    auto child_def = FlattenImpl(child, is_leaf_fn, leaves_list, tensors_only_leaf);
    if (child_def == nullptr) {
      return false;
    }
    children->push_back(std::move(child_def));
  }
  return true;
}

// Helper to flatten children from a sequence (PySequence_Fast)
bool FlattenSeqChildren(PyObject *children_seq, Py_ssize_t size, PyObject *is_leaf_fn, PyObject *leaves_list,
                        bool tensors_only_leaf, std::vector<PyTreeDefPtr> *children) {
  children->reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *child = PySequence_Fast_GET_ITEM(children_seq, i);
    auto child_def = FlattenImpl(child, is_leaf_fn, leaves_list, tensors_only_leaf);
    if (child_def == nullptr) {
      return false;
    }
    children->emplace_back(std::move(child_def));
  }
  return true;
}

// Helper to build tree recursively
PyTreeDefPtr FlattenImpl(PyObject *tree, PyObject *is_leaf_fn, PyObject *leaves_list, bool tensors_only_leaf) {
  EnsureBuiltinTypes();

  // Check custom is_leaf function
  if (is_leaf_fn != nullptr && is_leaf_fn != Py_None) {
    PyObject *result = PyObject_CallFunctionObjArgs(is_leaf_fn, tree, nullptr);
    if (result == nullptr) {
      return nullptr;
    }
    int is_true = PyObject_IsTrue(result);
    Py_DECREF(result);
    if (is_true < 0) {
      return nullptr;
    }
    if (is_true) {
      if (PyList_Append(leaves_list, tree) < 0) {
        return nullptr;
      }
      return PyTreeDef::Leaf();
    }
  }

  // Fast path: tensors only as leaves when enabled
  if (tensors_only_leaf && tensor::IsTensorPy(py::handle(tree))) {
    if (PyList_Append(leaves_list, tree) < 0) {
      return nullptr;
    }
    return PyTreeDef::Leaf();
  }

  PyTypeObject *obj_type = Py_TYPE(tree);

  // Check for namedtuple
  if (IsNamedTuple(tree)) {
    Py_ssize_t size = PyTuple_GET_SIZE(tree);
    std::vector<PyTreeDefPtr> children;
    if (!FlattenTupleChildren(tree, size, is_leaf_fn, leaves_list, tensors_only_leaf, &children)) {
      return nullptr;
    }
    return std::make_shared<PyTreeDef>(NodeType::kNamedTuple, obj_type, reinterpret_cast<PyObject *>(obj_type),
                                       std::move(children));
  }

  // Check builtin tuple
  if (obj_type == tuple_type) {
    Py_ssize_t size = PyTuple_GET_SIZE(tree);
    std::vector<PyTreeDefPtr> children;
    if (!FlattenTupleChildren(tree, size, is_leaf_fn, leaves_list, tensors_only_leaf, &children)) {
      return nullptr;
    }
    return std::make_shared<PyTreeDef>(NodeType::kTuple, tuple_type, nullptr, std::move(children));
  }

  // Check builtin list
  if (obj_type == list_type) {
    Py_ssize_t size = PyList_GET_SIZE(tree);
    std::vector<PyTreeDefPtr> children;
    if (!FlattenListChildren(tree, size, is_leaf_fn, leaves_list, tensors_only_leaf, &children)) {
      return nullptr;
    }
    return std::make_shared<PyTreeDef>(NodeType::kList, list_type, nullptr, std::move(children));
  }

  // Check builtin dict
  if (obj_type == dict_type) {
    PyObject *keys = PyDict_Keys(tree);
    if (keys == nullptr) {
      return nullptr;
    }
    // Sort keys for deterministic order
    if (PyList_Sort(keys) < 0) {
      Py_DECREF(keys);
      return nullptr;
    }
    Py_ssize_t size = PyList_GET_SIZE(keys);
    std::vector<PyTreeDefPtr> children;
    children.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *key = PyList_GET_ITEM(keys, i);
      PyObject *value = PyDict_GetItem(tree, key);
      auto child_def = FlattenImpl(value, is_leaf_fn, leaves_list, tensors_only_leaf);
      if (child_def == nullptr) {
        Py_DECREF(keys);
        return nullptr;
      }
      children.push_back(std::move(child_def));
    }
    // Convert keys list to tuple for aux_data
    PyObject *keys_tuple = PyList_AsTuple(keys);
    Py_DECREF(keys);
    if (keys_tuple == nullptr) {
      return nullptr;
    }
    auto result = std::make_shared<PyTreeDef>(NodeType::kDict, dict_type, keys_tuple, std::move(children));
    Py_DECREF(keys_tuple);  // PyTreeDef constructor increfs it
    return result;
  }

  // Check custom registered types
  const RegistryEntry *entry = PyTreeRegistry::Instance().Find(obj_type);
  if (entry != nullptr) {
    PyObject *flatten_result = PyObject_CallFunctionObjArgs(entry->flatten_func, tree, nullptr);
    if (flatten_result == nullptr) {
      return nullptr;
    }
    constexpr Py_ssize_t kFlattenResultSize = 2;
    if (!PyTuple_Check(flatten_result) || PyTuple_GET_SIZE(flatten_result) != kFlattenResultSize) {
      Py_DECREF(flatten_result);
      PyErr_SetString(PyExc_TypeError, "flatten_func must return a tuple of (children, aux_data)");
      return nullptr;
    }
    PyObject *children_obj = PyTuple_GET_ITEM(flatten_result, 0);
    PyObject *aux_data = PyTuple_GET_ITEM(flatten_result, 1);

    PyObject *children_seq = PySequence_Fast(children_obj, "children must be iterable");
    if (children_seq == nullptr) {
      Py_DECREF(flatten_result);
      return nullptr;
    }

    Py_ssize_t size = PySequence_Fast_GET_SIZE(children_seq);
    std::vector<PyTreeDefPtr> children;
    bool success = FlattenSeqChildren(children_seq, size, is_leaf_fn, leaves_list, tensors_only_leaf, &children);
    Py_DECREF(children_seq);
    if (!success) {
      Py_DECREF(flatten_result);
      return nullptr;
    }

    auto result = std::make_shared<PyTreeDef>(NodeType::kCustom, obj_type, aux_data, std::move(children));
    Py_DECREF(flatten_result);
    return result;
  }

  // Treat as leaf
  if (tensors_only_leaf) {
    // Keep in structure as constant node but do not add to leaves.
    Py_INCREF(tree);
    return std::make_shared<PyTreeDef>(NodeType::kConst, Py_TYPE(tree), tree, std::vector<PyTreeDefPtr>());
  }
  if (PyList_Append(leaves_list, tree) < 0) {
    return nullptr;
  }
  return PyTreeDef::Leaf();
}

// Helper class for iterating leaves during unflatten
class LeafIterator {
 public:
  explicit LeafIterator(PyObject *leaves)
      : leaves_(leaves), index_(0), size_(PyList_Check(leaves) ? PyList_GET_SIZE(leaves) : PyTuple_GET_SIZE(leaves)) {}

  PyObject *Next() {
    if (index_ >= size_) {
      PyErr_SetString(PyExc_ValueError, "Not enough leaves to unflatten");
      return nullptr;
    }
    PyObject *leaf = PyList_Check(leaves_) ? PyList_GET_ITEM(leaves_, index_) : PyTuple_GET_ITEM(leaves_, index_);
    ++index_;
    return leaf;
  }

  bool HasRemaining() const { return index_ < size_; }

 private:
  PyObject *leaves_;
  Py_ssize_t index_;
  Py_ssize_t size_;
};

// Forward declaration
PyObject *UnflattenImpl(const PyTreeDef &node_def, LeafIterator &leaf_iter);

// Helper to unflatten children into a tuple
PyObject *UnflattenChildrenToTuple(const std::vector<PyTreeDefPtr> &children, LeafIterator &leaf_iter) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  PyObject *result = PyTuple_New(num_children);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    PyObject *child_val = UnflattenImpl(*children[i], leaf_iter);
    if (child_val == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, i, child_val);
  }
  return result;
}

// Helper to unflatten children into a list
PyObject *UnflattenChildrenToList(const std::vector<PyTreeDefPtr> &children, LeafIterator &leaf_iter) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  PyObject *result = PyList_New(num_children);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    PyObject *child_val = UnflattenImpl(*children[i], leaf_iter);
    if (child_val == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SET_ITEM(result, i, child_val);
  }
  return result;
}

// Helper to unflatten children into a dict
PyObject *UnflattenChildrenToDict(const std::vector<PyTreeDefPtr> &children, PyObject *keys, LeafIterator &leaf_iter) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  PyObject *result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    PyObject *key = PyTuple_GET_ITEM(keys, i);
    PyObject *child_val = UnflattenImpl(*children[i], leaf_iter);
    if (child_val == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    if (PyDict_SetItem(result, key, child_val) < 0) {
      Py_DECREF(child_val);
      Py_DECREF(result);
      return nullptr;
    }
    Py_DECREF(child_val);
  }
  return result;
}

PyObject *UnflattenImpl(const PyTreeDef &node_def, LeafIterator &leaf_iter) {
  if (node_def.node_type() == NodeType::kLeaf) {
    PyObject *leaf = leaf_iter.Next();
    if (leaf == nullptr) {
      return nullptr;
    }
    Py_INCREF(leaf);
    return leaf;
  }

  const auto &children = node_def.children();

  switch (node_def.node_type()) {
    case NodeType::kTuple:
      return UnflattenChildrenToTuple(children, leaf_iter);

    case NodeType::kList:
      return UnflattenChildrenToList(children, leaf_iter);

    case NodeType::kDict:
      return UnflattenChildrenToDict(children, node_def.aux_data(), leaf_iter);

    case NodeType::kNamedTuple: {
      PyObject *child_vals = UnflattenChildrenToTuple(children, leaf_iter);
      if (child_vals == nullptr) {
        return nullptr;
      }
      PyObject *result = PyObject_Call(node_def.aux_data(), child_vals, nullptr);
      Py_DECREF(child_vals);
      return result;
    }

    case NodeType::kCustom: {
      PyObject *child_vals = UnflattenChildrenToTuple(children, leaf_iter);
      if (child_vals == nullptr) {
        return nullptr;
      }
      const RegistryEntry *entry = PyTreeRegistry::Instance().Find(node_def.type_object());
      if (entry == nullptr) {
        Py_DECREF(child_vals);
        PyErr_Format(PyExc_TypeError, "Unknown pytree node type: %s", node_def.type_object()->tp_name);
        return nullptr;
      }
      PyObject *result = PyObject_CallFunctionObjArgs(entry->unflatten_func, node_def.aux_data(), child_vals, nullptr);
      Py_DECREF(child_vals);
      return result;
    }

    case NodeType::kConst: {
      PyObject *val = node_def.aux_data();
      Py_INCREF(val);
      return val;
    }

    default:
      PyErr_SetString(PyExc_RuntimeError, "Unknown node type in unflatten");
      return nullptr;
  }
}

// Forward declaration for FlattenUpToImpl
bool FlattenUpToImpl(const PyTreeDef &node_def, PyObject *obj, PyObject *leaves_list);

// Helper to flatten tuple children up to a structure
bool FlattenUpToTupleChildren(const std::vector<PyTreeDefPtr> &children, PyObject *obj, PyObject *leaves_list) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    if (!FlattenUpToImpl(*children[i], PyTuple_GET_ITEM(obj, i), leaves_list)) {
      return false;
    }
  }
  return true;
}

// Helper to flatten list children up to a structure
bool FlattenUpToListChildren(const std::vector<PyTreeDefPtr> &children, PyObject *obj, PyObject *leaves_list) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    if (!FlattenUpToImpl(*children[i], PyList_GET_ITEM(obj, i), leaves_list)) {
      return false;
    }
  }
  return true;
}

// Helper to flatten dict children up to a structure
bool FlattenUpToDictChildren(const std::vector<PyTreeDefPtr> &children, PyObject *keys, PyObject *obj,
                             PyObject *leaves_list) {
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());
  for (Py_ssize_t i = 0; i < num_children; ++i) {
    PyObject *key = PyTuple_GET_ITEM(keys, i);
    PyObject *value = PyDict_GetItem(obj, key);
    if (value == nullptr) {
      PyErr_Format(PyExc_TypeError, "Structure mismatch: missing key in dict");
      return false;
    }
    if (!FlattenUpToImpl(*children[i], value, leaves_list)) {
      return false;
    }
  }
  return true;
}

// Helper to handle custom type in FlattenUpTo
bool FlattenUpToCustom(const PyTreeDef &node_def, PyObject *obj, PyObject *leaves_list) {
  PyTypeObject *obj_type = Py_TYPE(obj);
  const auto &children = node_def.children();
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());

  const RegistryEntry *entry = PyTreeRegistry::Instance().Find(obj_type);
  if (entry == nullptr) {
    PyErr_Format(PyExc_TypeError, "Unknown pytree node type: %s", obj_type->tp_name);
    return false;
  }
  PyObject *flatten_result = PyObject_CallFunctionObjArgs(entry->flatten_func, obj, nullptr);
  if (flatten_result == nullptr) {
    return false;
  }
  PyObject *children_obj = PyTuple_GET_ITEM(flatten_result, 0);
  PyObject *aux_data = PyTuple_GET_ITEM(flatten_result, 1);

  int cmp = PyObject_RichCompareBool(aux_data, node_def.aux_data(), Py_EQ);
  if (cmp <= 0) {
    Py_DECREF(flatten_result);
    if (cmp == 0) {
      PyErr_SetString(PyExc_TypeError, "Structure mismatch: aux_data differs");
    }
    return false;
  }

  PyObject *children_seq = PySequence_Fast(children_obj, "children must be iterable");
  if (children_seq == nullptr) {
    Py_DECREF(flatten_result);
    return false;
  }

  Py_ssize_t size = PySequence_Fast_GET_SIZE(children_seq);
  if (size != num_children) {
    Py_DECREF(children_seq);
    Py_DECREF(flatten_result);
    PyErr_SetString(PyExc_TypeError, "Structure mismatch: number of children differs");
    return false;
  }

  for (Py_ssize_t i = 0; i < num_children; ++i) {
    if (!FlattenUpToImpl(*children[i], PySequence_Fast_GET_ITEM(children_seq, i), leaves_list)) {
      Py_DECREF(children_seq);
      Py_DECREF(flatten_result);
      return false;
    }
  }
  Py_DECREF(children_seq);
  Py_DECREF(flatten_result);
  return true;
}

bool FlattenUpToImpl(const PyTreeDef &node_def, PyObject *obj, PyObject *leaves_list) {
  if (node_def.node_type() == NodeType::kLeaf) {
    return PyList_Append(leaves_list, obj) == 0;
  }

  PyTypeObject *obj_type = Py_TYPE(obj);
  const auto &children = node_def.children();
  Py_ssize_t num_children = static_cast<Py_ssize_t>(children.size());

  switch (node_def.node_type()) {
    case NodeType::kTuple:
      if (obj_type != &PyTuple_Type) {
        PyErr_Format(PyExc_TypeError, "Structure mismatch: expected tuple, got %s", obj_type->tp_name);
        return false;
      }
      if (PyTuple_GET_SIZE(obj) != num_children) {
        PyErr_SetString(PyExc_TypeError, "Structure mismatch: tuple size differs");
        return false;
      }
      return FlattenUpToTupleChildren(children, obj, leaves_list);

    case NodeType::kList:
      if (obj_type != &PyList_Type) {
        PyErr_Format(PyExc_TypeError, "Structure mismatch: expected list, got %s", obj_type->tp_name);
        return false;
      }
      if (PyList_GET_SIZE(obj) != num_children) {
        PyErr_SetString(PyExc_TypeError, "Structure mismatch: list size differs");
        return false;
      }
      return FlattenUpToListChildren(children, obj, leaves_list);

    case NodeType::kDict:
      if (obj_type != &PyDict_Type) {
        PyErr_Format(PyExc_TypeError, "Structure mismatch: expected dict, got %s", obj_type->tp_name);
        return false;
      }
      if (PyDict_Size(obj) != num_children) {
        PyErr_SetString(PyExc_TypeError, "Structure mismatch: dict size differs");
        return false;
      }
      return FlattenUpToDictChildren(children, node_def.aux_data(), obj, leaves_list);

    case NodeType::kNamedTuple:
      if (!IsNamedTuple(obj) || obj_type != node_def.type_object()) {
        PyErr_Format(PyExc_TypeError, "Structure mismatch: expected namedtuple %s, got %s",
                     node_def.type_object()->tp_name, obj_type->tp_name);
        return false;
      }
      if (PyTuple_GET_SIZE(obj) != num_children) {
        PyErr_SetString(PyExc_TypeError, "Structure mismatch: namedtuple size differs");
        return false;
      }
      return FlattenUpToTupleChildren(children, obj, leaves_list);

    case NodeType::kCustom:
      if (obj_type != node_def.type_object()) {
        PyErr_Format(PyExc_TypeError, "Structure mismatch: expected %s, got %s", node_def.type_object()->tp_name,
                     obj_type->tp_name);
        return false;
      }
      return FlattenUpToCustom(node_def, obj, leaves_list);

    case NodeType::kConst: {
      int cmp = PyObject_RichCompareBool(obj, node_def.aux_data(), Py_EQ);
      if (cmp <= 0) {
        if (cmp == 0) {
          PyErr_SetString(PyExc_TypeError, "Structure mismatch: constant value differs");
        }
        return false;
      }
      return true;
    }

    default:
      PyErr_SetString(PyExc_RuntimeError, "Unknown node type in flatten_up_to");
      return false;
  }
}

}  // namespace

// ============================================================================
// PyTreeRegistry implementation
// ============================================================================

PyTreeRegistry &PyTreeRegistry::Instance() {
  static PyTreeRegistry instance;
  return instance;
}

PyTreeRegistry::~PyTreeRegistry() {
  // Note: Do NOT call Py_XDECREF here!
  // When this destructor runs (during static destruction), the Python
  // interpreter may already be finalized, making Py_XDECREF unsafe.
  // The Python objects will be cleaned up by Python's garbage collector
  // before interpreter shutdown anyway.
}

void PyTreeRegistry::Register(PyTypeObject *type, PyObject *flatten_func, PyObject *unflatten_func) {
  auto it = registry_.find(type);
  if (it != registry_.end()) {
    Py_XDECREF(it->second.flatten_func);
    Py_XDECREF(it->second.unflatten_func);
  }
  Py_INCREF(flatten_func);
  Py_INCREF(unflatten_func);
  registry_[type] = {flatten_func, unflatten_func};
}

const RegistryEntry *PyTreeRegistry::Find(PyTypeObject *type) const {
  auto it = registry_.find(type);
  return it != registry_.end() ? &it->second : nullptr;
}

bool PyTreeRegistry::Contains(PyTypeObject *type) const { return registry_.find(type) != registry_.end(); }

void PyTreeRegistry::Clear() {
  if (Py_IsInitialized()) {
    for (auto &pair : registry_) {
      Py_XDECREF(pair.second.flatten_func);
      Py_XDECREF(pair.second.unflatten_func);
    }
  }
  registry_.clear();
}

// ============================================================================
// PyTreeDef implementation
// ============================================================================

PyTreeDef::~PyTreeDef() {
  // Only decref if Python interpreter is still initialized.
  // During program shutdown, static PyTreeDef objects may be destroyed
  // after Python has been finalized.
  if (aux_data_ != nullptr && Py_IsInitialized()) {
    Py_DECREF(aux_data_);
  }
}

PyTreeDefPtr PyTreeDef::Leaf() {
  // Create a new leaf each time instead of using a static singleton.
  // Static singletons can cause issues during Python interpreter shutdown
  // when the destructor runs after Python has been finalized.
  return std::make_shared<PyTreeDef>();
}

size_t PyTreeDef::NumLeaves() const {
  if (node_type_ == NodeType::kLeaf) {
    return 1;
  }
  if (node_type_ == NodeType::kConst) {
    return 0;
  }
  return std::accumulate(children_.begin(), children_.end(), size_t{0},
                         [](size_t sum, const auto &child) { return sum + child->NumLeaves(); });
}

size_t PyTreeDef::NumNodes() const {
  if (node_type_ == NodeType::kLeaf) {
    return 1;
  }
  return std::accumulate(children_.begin(), children_.end(), size_t{1},
                         [](size_t sum, const auto &child) { return sum + child->NumNodes(); });
}

PyObject *PyTreeDef::Unflatten(PyObject *leaves) const {
  LeafIterator iter(leaves);
  PyObject *result = UnflattenImpl(*this, iter);
  if (result == nullptr) {
    return nullptr;
  }
  if (iter.HasRemaining()) {
    Py_DECREF(result);
    PyErr_SetString(PyExc_ValueError, "Too many leaves provided for unflatten");
    return nullptr;
  }
  return result;
}

PyObject *PyTreeDef::FlattenUpTo(PyObject *tree) const {
  PyObject *leaves = PyList_New(0);
  if (leaves == nullptr) {
    return nullptr;
  }
  if (!FlattenUpToImpl(*this, tree, leaves)) {
    Py_DECREF(leaves);
    return nullptr;
  }
  return leaves;
}

bool PyTreeDef::Equals(const PyTreeDef &other) const {
  if (node_type_ != other.node_type_) {
    return false;
  }
  if (type_object_ != other.type_object_) {
    return false;
  }
  if (children_.size() != other.children_.size()) {
    return false;
  }
  // Compare aux_data
  if (aux_data_ == nullptr && other.aux_data_ == nullptr) {
    // Both null, ok
  } else if (aux_data_ == nullptr || other.aux_data_ == nullptr) {
    return false;
  } else {
    int cmp = PyObject_RichCompareBool(aux_data_, other.aux_data_, Py_EQ);
    if (cmp <= 0) {
      return false;
    }
  }
  // Compare children
  for (size_t i = 0; i < children_.size(); ++i) {
    if (!children_[i]->Equals(*other.children_[i])) {
      return false;
    }
  }
  return true;
}

std::string PyTreeDef::ToString() const {
  std::ostringstream oss;
  switch (node_type_) {
    case NodeType::kLeaf:
      oss << "*";
      break;
    case NodeType::kTuple:
      oss << "PyTreeDef((";
      for (size_t i = 0; i < children_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << children_[i]->ToString();
      }
      oss << "))";
      break;
    case NodeType::kList:
      oss << "PyTreeDef([";
      for (size_t i = 0; i < children_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << children_[i]->ToString();
      }
      oss << "])";
      break;
    case NodeType::kDict:
      oss << "PyTreeDef({...})";
      break;
    case NodeType::kNamedTuple:
      oss << "PyTreeDef(" << type_object_->tp_name << "(...))";
      break;
    case NodeType::kCustom:
      oss << "PyTreeDef(Custom[" << type_object_->tp_name << "])";
      break;
    case NodeType::kConst:
      oss << "Const(" << (type_object_ != nullptr ? type_object_->tp_name : "None") << ")";
      break;
  }
  return oss.str();
}

// ============================================================================
// Public API
// ============================================================================

std::pair<PyObject *, PyTreeDefPtr> Flatten(PyObject *tree, PyObject *is_leaf_fn, bool tensors_only_leaf) {
  PyObject *leaves = PyList_New(0);
  if (leaves == nullptr) {
    return {nullptr, nullptr};
  }
  auto treedef = FlattenImpl(tree, is_leaf_fn, leaves, tensors_only_leaf);
  if (treedef == nullptr) {
    Py_DECREF(leaves);
    return {nullptr, nullptr};
  }
  return {leaves, treedef};
}

bool IsNamedTuple(PyObject *obj) {
  if (!PyTuple_Check(obj)) {
    return false;
  }
  return PyObject_HasAttrString(reinterpret_cast<PyObject *>(Py_TYPE(obj)), "_fields") != 0;
}

// ============================================================================
// Python bindings
// ============================================================================

static void PyTreeDef_dealloc(PyTreeDefObject *self) {
  self->treedef.~shared_ptr<PyTreeDef>();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *PyTreeDef_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyTreeDefObject *self = reinterpret_cast<PyTreeDefObject *>(type->tp_alloc(type, 0));
  if (self != nullptr) {
    new (&self->treedef) PyTreeDefPtr();
  }
  return reinterpret_cast<PyObject *>(self);
}

static PyObject *PyTreeDef_num_leaves(PyTreeDefObject *self, void *) {
  return PyLong_FromSize_t(self->treedef->NumLeaves());
}

static PyObject *PyTreeDef_num_nodes(PyTreeDefObject *self, void *) {
  return PyLong_FromSize_t(self->treedef->NumNodes());
}

static PyObject *PyTreeDef_unflatten(PyTreeDefObject *self, PyObject *leaves) {
  return self->treedef->Unflatten(leaves);
}

static PyObject *PyTreeDef_flatten_up_to(PyTreeDefObject *self, PyObject *tree) {
  return self->treedef->FlattenUpTo(tree);
}

static PyObject *PyTreeDef_repr(PyTreeDefObject *self) {
  std::string s = self->treedef->ToString();
  return PyUnicode_FromString(s.c_str());
}

static PyObject *PyTreeDef_richcompare(PyTreeDefObject *self, PyObject *other, int op) {
  if (op != Py_EQ && op != Py_NE) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  if (!PyObject_TypeCheck(other, &PyTreeDefType)) {
    if (op == Py_EQ) {
      Py_RETURN_FALSE;
    } else {
      Py_RETURN_TRUE;
    }
  }
  auto other_def = reinterpret_cast<PyTreeDefObject *>(other);
  bool equal = self->treedef->Equals(*other_def->treedef);
  if (op == Py_EQ) {
    return PyBool_FromLong(equal);
  } else {
    return PyBool_FromLong(!equal);
  }
}

static PyGetSetDef PyTreeDef_getsetters[] = {
  {"num_leaves", reinterpret_cast<getter>(PyTreeDef_num_leaves), nullptr, "Number of leaves", nullptr},
  {"num_nodes", reinterpret_cast<getter>(PyTreeDef_num_nodes), nullptr, "Number of nodes", nullptr},
  {nullptr}};

static PyMethodDef PyTreeDef_methods[] = {
  {"unflatten", reinterpret_cast<PyCFunction>(PyTreeDef_unflatten), METH_O, "Unflatten leaves into tree"},
  {"flatten_up_to", reinterpret_cast<PyCFunction>(PyTreeDef_flatten_up_to), METH_O, "Flatten tree following structure"},
  {nullptr}};

PyTypeObject PyTreeDefType = {
  PyVarObject_HEAD_INIT(nullptr, 0) "PyTreeDef",        /* tp_name */
  sizeof(PyTreeDefObject),                              /* tp_basicsize */
  0,                                                    /* tp_itemsize */
  reinterpret_cast<destructor>(PyTreeDef_dealloc),      /* tp_dealloc */
  0,                                                    /* tp_vectorcall_offset */
  nullptr,                                              /* tp_getattr */
  nullptr,                                              /* tp_setattr */
  nullptr,                                              /* tp_reserved */
  reinterpret_cast<reprfunc>(PyTreeDef_repr),           /* tp_repr */
  nullptr,                                              /* tp_as_number */
  nullptr,                                              /* tp_as_sequence */
  nullptr,                                              /* tp_as_mapping */
  nullptr,                                              /* tp_hash */
  nullptr,                                              /* tp_call */
  reinterpret_cast<reprfunc>(PyTreeDef_repr),           /* tp_str */
  nullptr,                                              /* tp_getattro */
  nullptr,                                              /* tp_setattro */
  nullptr,                                              /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                                   /* tp_flags */
  "PyTree structure definition",                        /* tp_doc */
  nullptr,                                              /* tp_traverse */
  nullptr,                                              /* tp_clear */
  reinterpret_cast<richcmpfunc>(PyTreeDef_richcompare), /* tp_richcompare */
  0,                                                    /* tp_weaklistoffset */
  nullptr,                                              /* tp_iter */
  nullptr,                                              /* tp_iternext */
  PyTreeDef_methods,                                    /* tp_methods */
  nullptr,                                              /* tp_members */
  PyTreeDef_getsetters,                                 /* tp_getset */
  nullptr,                                              /* tp_base */
  nullptr,                                              /* tp_dict */
  nullptr,                                              /* tp_descr_get */
  nullptr,                                              /* tp_descr_set */
  0,                                                    /* tp_dictoffset */
  nullptr,                                              /* tp_init */
  nullptr,                                              /* tp_alloc */
  PyTreeDef_new,                                        /* tp_new */
};

// Module-level functions
static PyObject *py_tree_flatten(PyObject *self, PyObject *args, PyObject *kwargs) {
  static const char *kwlist[] = {"tree", "is_leaf", "tensors_only_leaf", nullptr};
  PyObject *tree = nullptr;
  PyObject *is_leaf = nullptr;
  int tensors_only_leaf = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", const_cast<char **>(kwlist), &tree, &is_leaf,
                                   &tensors_only_leaf)) {
    return nullptr;
  }

  auto [leaves, treedef] = Flatten(tree, is_leaf, tensors_only_leaf != 0);
  if (leaves == nullptr) {
    return nullptr;
  }

  // Wrap treedef in Python object
  PyTreeDefObject *treedef_obj = PyObject_New(PyTreeDefObject, &PyTreeDefType);
  if (treedef_obj == nullptr) {
    Py_DECREF(leaves);
    return nullptr;
  }
  new (&treedef_obj->treedef) PyTreeDefPtr(std::move(treedef));

  PyObject *result = PyTuple_Pack(2, leaves, treedef_obj);
  Py_DECREF(leaves);
  Py_DECREF(treedef_obj);
  return result;
}

static PyObject *py_tree_unflatten(PyObject *self, PyObject *args) {
  PyObject *treedef_obj = nullptr;
  PyObject *leaves = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &treedef_obj, &leaves)) {
    return nullptr;
  }

  if (!PyObject_TypeCheck(treedef_obj, &PyTreeDefType)) {
    PyErr_SetString(PyExc_TypeError, "First argument must be a PyTreeDef");
    return nullptr;
  }

  auto *treedef = reinterpret_cast<PyTreeDefObject *>(treedef_obj);
  return treedef->treedef->Unflatten(leaves);
}

static PyObject *py_tree_leaves(PyObject *self, PyObject *args, PyObject *kwargs) {
  static const char *kwlist[] = {"tree", "is_leaf", "tensors_only_leaf", nullptr};
  PyObject *tree = nullptr;
  PyObject *is_leaf = nullptr;
  int tensors_only_leaf = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", const_cast<char **>(kwlist), &tree, &is_leaf,
                                   &tensors_only_leaf)) {
    return nullptr;
  }

  auto [leaves, treedef] = Flatten(tree, is_leaf, tensors_only_leaf != 0);
  (void)treedef;  // unused
  return leaves;
}

static PyObject *py_tree_structure(PyObject *self, PyObject *args, PyObject *kwargs) {
  static const char *kwlist[] = {"tree", "is_leaf", "tensors_only_leaf", nullptr};
  PyObject *tree = nullptr;
  PyObject *is_leaf = nullptr;
  int tensors_only_leaf = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", const_cast<char **>(kwlist), &tree, &is_leaf,
                                   &tensors_only_leaf)) {
    return nullptr;
  }

  auto [leaves, treedef] = Flatten(tree, is_leaf, tensors_only_leaf != 0);
  if (leaves == nullptr) {
    return nullptr;
  }
  Py_DECREF(leaves);

  PyTreeDefObject *treedef_obj = PyObject_New(PyTreeDefObject, &PyTreeDefType);
  if (treedef_obj == nullptr) {
    return nullptr;
  }
  new (&treedef_obj->treedef) PyTreeDefPtr(std::move(treedef));
  return reinterpret_cast<PyObject *>(treedef_obj);
}

static PyObject *py_tree_map(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *func = nullptr;
  PyObject *tree = nullptr;
  PyObject *is_leaf = nullptr;
  int tensors_only_leaf = 0;

  // Parse first 3 positional/keyword arguments
  constexpr Py_ssize_t kMinTreeMapArgs = 2;
  Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  if (nargs < kMinTreeMapArgs) {
    PyErr_SetString(PyExc_TypeError, "tree_map requires at least func and tree arguments");
    return nullptr;
  }

  func = PyTuple_GET_ITEM(args, 0);
  tree = PyTuple_GET_ITEM(args, 1);

  if (kwargs) {
    is_leaf = PyDict_GetItemString(kwargs, "is_leaf");
    PyObject *tol = PyDict_GetItemString(kwargs, "tensors_only_leaf");
    if (tol != nullptr) {
      tensors_only_leaf = PyObject_IsTrue(tol);
    }
  }

  // Flatten the first tree
  auto [leaves, treedef] = Flatten(tree, is_leaf, tensors_only_leaf != 0);
  if (leaves == nullptr) {
    return nullptr;
  }

  // Collect leaves from all trees
  std::vector<PyObject *> all_leaves;
  all_leaves.push_back(leaves);

  for (Py_ssize_t i = 2; i < nargs; ++i) {
    PyObject *other_tree = PyTuple_GET_ITEM(args, i);
    PyObject *other_leaves = treedef->FlattenUpTo(other_tree);
    if (other_leaves == nullptr) {
      for (auto *l : all_leaves) {
        Py_DECREF(l);
      }
      return nullptr;
    }
    all_leaves.push_back(other_leaves);
  }

  // Apply function to each set of leaves
  Py_ssize_t num_leaves = PyList_GET_SIZE(leaves);
  PyObject *result_leaves = PyList_New(num_leaves);
  if (result_leaves == nullptr) {
    for (auto *l : all_leaves) {
      Py_DECREF(l);
    }
    return nullptr;
  }

  for (Py_ssize_t i = 0; i < num_leaves; ++i) {
    PyObject *func_args = PyTuple_New(static_cast<Py_ssize_t>(all_leaves.size()));
    if (func_args == nullptr) {
      for (auto *l : all_leaves) {
        Py_DECREF(l);
      }
      Py_DECREF(result_leaves);
      return nullptr;
    }
    for (size_t j = 0; j < all_leaves.size(); ++j) {
      PyObject *leaf = PyList_GET_ITEM(all_leaves[j], i);
      Py_INCREF(leaf);
      PyTuple_SET_ITEM(func_args, static_cast<Py_ssize_t>(j), leaf);
    }
    PyObject *result_leaf = PyObject_Call(func, func_args, nullptr);
    Py_DECREF(func_args);
    if (result_leaf == nullptr) {
      for (auto *l : all_leaves) {
        Py_DECREF(l);
      }
      Py_DECREF(result_leaves);
      return nullptr;
    }
    PyList_SET_ITEM(result_leaves, i, result_leaf);
  }

  // Clean up
  for (auto *l : all_leaves) {
    Py_DECREF(l);
  }

  // Unflatten result
  PyObject *result = treedef->Unflatten(result_leaves);
  Py_DECREF(result_leaves);
  return result;
}

static PyObject *py_register_pytree_node(PyObject *self, PyObject *args) {
  PyObject *node_type = nullptr;
  PyObject *flatten_func = nullptr;
  PyObject *unflatten_func = nullptr;

  if (!PyArg_ParseTuple(args, "OOO", &node_type, &flatten_func, &unflatten_func)) {
    return nullptr;
  }

  if (!PyType_Check(node_type)) {
    PyErr_SetString(PyExc_TypeError, "First argument must be a type");
    return nullptr;
  }

  if (!PyCallable_Check(flatten_func)) {
    PyErr_SetString(PyExc_TypeError, "flatten_func must be callable");
    return nullptr;
  }

  if (!PyCallable_Check(unflatten_func)) {
    PyErr_SetString(PyExc_TypeError, "unflatten_func must be callable");
    return nullptr;
  }

  PyTreeRegistry::Instance().Register(reinterpret_cast<PyTypeObject *>(node_type), flatten_func, unflatten_func);
  Py_RETURN_NONE;
}

static PyMethodDef pytree_methods[] = {
  {"_tree_flatten", reinterpret_cast<PyCFunction>(py_tree_flatten), METH_VARARGS | METH_KEYWORDS,
   "Flatten a pytree into leaves and structure"},
  {"_tree_unflatten", py_tree_unflatten, METH_VARARGS, "Unflatten leaves into a pytree"},
  {"_tree_leaves", reinterpret_cast<PyCFunction>(py_tree_leaves), METH_VARARGS | METH_KEYWORDS,
   "Get leaves of a pytree"},
  {"_tree_structure", reinterpret_cast<PyCFunction>(py_tree_structure), METH_VARARGS | METH_KEYWORDS,
   "Get structure of a pytree"},
  {"_tree_map", reinterpret_cast<PyCFunction>(py_tree_map), METH_VARARGS | METH_KEYWORDS,
   "Map a function over pytree(s)"},
  {"_register_pytree_node", py_register_pytree_node, METH_VARARGS, "Register a custom pytree node type"},
  {nullptr, nullptr, 0, nullptr}};

void InitPyTreeModule(PyObject *module) {
  if (PyType_Ready(&PyTreeDefType) < 0) {
    return;
  }
}

namespace py = pybind11;

void RegPyTree(py::module *m) {
  if (PyType_Ready(&PyTreeDefType) < 0) {
    return;
  }

  Py_INCREF(&PyTreeDefType);
  m->add_object("PyTreeDef", reinterpret_cast<PyObject *>(&PyTreeDefType));

  // Add functions
  for (PyMethodDef *method = pytree_methods; method->ml_name != nullptr; ++method) {
    PyObject *func = PyCFunction_New(method, nullptr);
    if (func != nullptr) {
      m->add_object(method->ml_name, func);
    }
  }
}

}  // namespace pytree
}  // namespace mindspore
