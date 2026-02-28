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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_PYTREE_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_PYTREE_H_

#include <cmath>
#include <Python.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pytree {
enum class NodeType : uint8_t { kLeaf = 0, kTuple, kList, kDict, kNamedTuple, kCustom, kConst };

class PyTreeDef;
using PyTreeDefPtr = std::shared_ptr<PyTreeDef>;

/// Registry entry for custom pytree node types
struct RegistryEntry {
  PyObject *flatten_func;    // callable(obj) -> (children, aux_data)
  PyObject *unflatten_func;  // callable(aux_data, children) -> obj
};

/// Global registry for pytree node types
class COMMON_EXPORT PyTreeRegistry {
 public:
  static PyTreeRegistry &Instance();

  /// Register a custom node type
  void Register(PyTypeObject *type, PyObject *flatten_func, PyObject *unflatten_func);

  /// Find entry for a type
  const RegistryEntry *Find(PyTypeObject *type) const;

  /// Check if a type is registered
  bool Contains(PyTypeObject *type) const;

  /// Clear all registrations (for testing)
  void Clear();

 private:
  PyTreeRegistry() = default;
  ~PyTreeRegistry();
  std::unordered_map<PyTypeObject *, RegistryEntry> registry_;
};

/// PyTreeDef represents the structure of a pytree
class COMMON_EXPORT PyTreeDef {
 public:
  PyTreeDef() : node_type_(NodeType::kLeaf), type_object_(nullptr), aux_data_(nullptr) {}

  explicit PyTreeDef(NodeType node_type, PyTypeObject *type_object, PyObject *aux_data,
                     std::vector<PyTreeDefPtr> children)
      : node_type_(node_type), type_object_(type_object), aux_data_(aux_data), children_(std::move(children)) {
    if (aux_data_ != nullptr) {
      Py_INCREF(aux_data_);
    }
  }

  ~PyTreeDef();

  /// Create a leaf node
  static PyTreeDefPtr Leaf();

  /// Get number of leaves in the tree
  size_t NumLeaves() const;

  /// Get number of nodes in the tree
  size_t NumNodes() const;

  /// Unflatten leaves into a pytree following this structure
  PyObject *Unflatten(PyObject *leaves) const;

  /// Flatten a tree following this structure, ensuring it matches
  PyObject *FlattenUpTo(PyObject *tree) const;

  /// Check equality with another treedef
  bool Equals(const PyTreeDef &other) const;

  /// Get string representation
  std::string ToString() const;

  /// Accessors
  NodeType node_type() const { return node_type_; }
  PyTypeObject *type_object() const { return type_object_; }
  PyObject *aux_data() const { return aux_data_; }
  const std::vector<PyTreeDefPtr> &children() const { return children_; }

 private:
  NodeType node_type_;
  PyTypeObject *type_object_;  // borrowed reference for builtin, owned for custom
  PyObject *aux_data_;         // owned reference
  std::vector<PyTreeDefPtr> children_;
};

/// Core flatten operation
/// Returns (leaves_list, treedef)
COMMON_EXPORT std::pair<PyObject *, PyTreeDefPtr> Flatten(PyObject *tree, PyObject *is_leaf_fn, bool tensors_only_leaf);

/// Check if an object is a namedtuple instance
COMMON_EXPORT bool IsNamedTuple(PyObject *obj);

/// Python type object for PyTreeDef
COMMON_EXPORT extern PyTypeObject PyTreeDefType;

/// Python object wrapper for PyTreeDef
struct PyTreeDefObject {
  PyObject_HEAD PyTreeDefPtr treedef;
};

/// Initialize pytree module (for CPython module)
COMMON_EXPORT void InitPyTreeModule(PyObject *module);

/// Register pytree types to pybind11 module
COMMON_EXPORT void RegPyTree(pybind11::module *m);

}  // namespace pytree
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_PYTREE_H_
