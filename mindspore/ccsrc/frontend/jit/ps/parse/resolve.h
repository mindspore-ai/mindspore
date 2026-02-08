/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_RESOLVE_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_RESOLVE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "ir/manager.h"
#include "include/utils/python_adapter.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"
#include "utils/tensor_hook_map.h"

// forward declaration of ResourceBase
namespace mindspore {
namespace pipeline {
class ResourceBase;
using ResourceBasePtr = std::shared_ptr<ResourceBase>;
}  // namespace pipeline
}  // namespace mindspore

namespace mindspore {
namespace parse {
// NameSpace class for resolving python code.
class NameSpace final : public Named {
 public:
  NameSpace(const std::string &module, const py::object &namespace_obj, const py::object &module_obj = py::none())
      : Named(module + ": \'" + std::string(py::str(namespace_obj)) + "\'"),
        module_(module),
        namespace_obj_(namespace_obj),
        module_obj_(module_obj) {}
  ~NameSpace() override {
    py::gil_scoped_acquire gil_acquire;
    namespace_obj_ = py::object();
    module_obj_ = py::object();
  }
  MS_DECLARE_PARENT(NameSpace, Named);

  const py::object &namespace_obj() const { return namespace_obj_; }
  const py::object &module_obj() const { return module_obj_; }
  void set_module_obj(py::object module_obj) { module_obj_ = module_obj; }
  const std::string &module() const { return module_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<NameSpace>(), std::make_shared<External>());
  }

 private:
  // namespace of the module
  std::string module_;
  // namespace object
  py::object namespace_obj_;
  // module object
  py::object module_obj_;
};
using NameSpacePtr = std::shared_ptr<NameSpace>;

// Symbol in NameSpace or Class which shall be resolved.
class Symbol final : public Named {
 public:
  explicit Symbol(const std::string &symbol) : Named(symbol), symbol_(symbol) {}
  Symbol(const std::string &symbol, const std::string &name) : Named(name), symbol_(symbol) {}

  ~Symbol() override = default;
  MS_DECLARE_PARENT(Symbol, Named);

  const std::string &symbol() const { return symbol_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<Symbol>(), std::make_shared<External>());
  }

 private:
  std::string symbol_;
};
using SymbolPtr = std::shared_ptr<Symbol>;

class Script final : public Named {
 public:
  explicit Script(const std::string &script) : Named(script), script_(script) {}
  Script(const std::string &script, const std::string &name) : Named(name), script_(script) {}

  ~Script() override = default;
  MS_DECLARE_PARENT(Script, Named);

  std::string script() const { return script_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScript>(shared_from_base<Script>());
  }
  std::string ToString() const override { return "\'" + name() + "\'"; }

 private:
  std::string script_;
};
using ScriptPtr = std::shared_ptr<Script>;

// PyObjectWrapper class wrappers resolved python object for further processing.
class PyObjectWrapper : public Named {
 public:
  explicit PyObjectWrapper(const py::object &obj, const std::string &name = "Python object")
      : Named(name), obj_(std::make_unique<py::object>(obj)) {}
  ~PyObjectWrapper() {
    py::gil_scoped_acquire acquire_gil;
    obj_ = nullptr;
  }

  MS_DECLARE_PARENT(PyObjectWrapper, Named);
  py::object obj() const { return *obj_; }

  std::size_t hash() const override { return tid(); }

  virtual bool operator==(const PyObjectWrapper &other) const {
    py::gil_scoped_acquire acquire_gil;
    if (obj().get_type() != other.obj().get_type()) {
      return false;
    }
    try {
      return obj().equal(other.obj());
    } catch (const std::exception &e) {
      // Return false if the comparison is ambiguous. Such as numpy.array.
      MS_LOG(INFO) << e.what() << "\n"
                   << "This: {" << py::str(obj()) << ", " << py::str(obj().get_type()) << "}, Other: {"
                   << py::str(other.obj()) << ", " << py::str(other.obj().get_type()) << "}";
      return false;
    }
  }
  bool operator==(const Named &other) const override {
    if (other.isa<PyObjectWrapper>()) {
      auto &other_py_obj = static_cast<const PyObjectWrapper &>(other);
      return *this == other_py_obj;
    }
    return false;
  }

 private:
  // The object that needs to be resolved
  std::unique_ptr<py::object> obj_;
};
using PyObjectWrapperPtr = std::shared_ptr<PyObjectWrapper>;

// InterpretedObject class wrappers interpreted python object.
class InterpretedObject final : public PyObjectWrapper {
 public:
  explicit InterpretedObject(const py::object &obj);
  ~InterpretedObject() override = default;
  MS_DECLARE_PARENT(InterpretedObject, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<InterpretedObject>(),
                                                      std::make_shared<External>());
  }
  void set_has_converted(bool has_converted) { has_converted_ = has_converted; }
  bool has_converted() const { return has_converted_; }

 private:
  bool has_converted_ = false;
};
using InterpretedObjectPtr = std::shared_ptr<InterpretedObject>;

class MsClassObject final : public PyObjectWrapper {
 public:
  explicit MsClassObject(const py::object &obj, const std::string &name)
      : PyObjectWrapper(obj, "MsClassObject: \'" + name + "\'") {}
  ~MsClassObject() override = default;
  MS_DECLARE_PARENT(MsClassObject, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override;
};
using MsClassObjectPtr = std::shared_ptr<MsClassObject>;

// ClassType class wrappers class name in python
class ClassType final : public PyObjectWrapper {
 public:
  explicit ClassType(const py::object &obj, const std::string &name = "Python class type")
      : PyObjectWrapper(obj, name) {}
  ~ClassType() override = default;
  MS_DECLARE_PARENT(ClassType, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override;
};
using ClassTypePtr = std::shared_ptr<ClassType>;

class Resolver {
 public:
  explicit Resolver(const FuncGraphPtr &top_func_graph) : top_func_graph_(top_func_graph) {}
  virtual ~Resolver() = default;

  AnfNodePtr ResolveParameterObj(const FuncGraphPtr &func_graph, const py::object &obj);
  bool ResolveObjectToNode(const AnfNodePtr &origin_node, const py::object &obj, AnfNodePtr *const node,
                           bool is_element_obj = false);
  // Resolve symbol in namespace.
  AnfNodePtr ResolveSymbol(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space, const SymbolPtr &symbol,
                           const AnfNodePtr &node);
  AnfNodePtr ResolveSymbolWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &object_node,
                                   const AnfNodePtr &attr_node, const AnfNodePtr &node);
  AnfNodePtr ResolveGetItemWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &getitem_node,
                                    const AnfNodePtr &attr_node, const AnfNodePtr &node);

 private:
  AnfNodePtr ResolveObjectAndAddToManager(const FuncGraphManagerPtr &manager, const py::object &obj,
                                          const AnfNodePtr &node);
  AnfNodePtr ResolveCellWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                                 const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                                 const AnfNodePtr &get_attr_node);
  AnfNodePtr ResolveSequenceWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                                     const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                                     const CNodePtr &get_attr_node);
  AnfNodePtr ConvertObjectToNode(const AnfNodePtr &origin_node, const py::object &obj, const FuncGraphPtr &func_graph,
                                 bool is_element_obj);
  void ConvertLoadedGraph(const FuncGraphPtr &func_graph, const ValuePtr &value);

  FuncGraphPtr top_func_graph_{nullptr};
};

AnfNodePtr ResolveClassObjectWithAttr(const py::object &cls_obj, const AnfNodePtr &attr, const AnfNodePtr &node);

AnfNodePtr ResolveInterpretedObjectOfSetAttr(const AnfNodePtr &target_node, const AnfNodePtr &attr_node,
                                             const AnfNodePtr &value_node);

bool IsParameterObject(const py::object &obj);
bool ContainsParameter(const py::object &obj);

// Check if node is cnode with getitem.
bool IsGetItemCNode(const AnfNodePtr &node);

// Resolve one graph which normally is the root graph. FuncGraph shall be managed by res->manager().
bool ResolveFuncGraph(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &res, bool use_profile = true);

// Resolve all graphs in manager which is defined outside of pipeline::Resource.
// Mainly used for test cases or resolve graphs which will not be managed by manager.
bool ResolveAll(const FuncGraphManagerPtr &manager);

TensorHookMapPtr ResolveTensorHooks(const pipeline::ResourceBasePtr &resource, const tensor::TensorPtr &tensor);
FuncGraphPtr ResolveCellHook(const py::object &obj, const std::string &hook_dict_name,
                             const std::string &hook_func_name);
FuncGraphPtr ResolveSaveTensorHook(const py::object &obj, const std::string &hook_name);
py::object GetSymbolObject(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node);
ValuePtr GetParameterValue(const py::object &param_obj);

// Convert py::object to py::str, and then convert it to std::string.
// Return std::nullopt if obj.ptr() is nullptr, or if exception is caught.
std::optional<std::string> ToPyStr(const py::object &obj);
// Convert py::object to py::str, and then convert it to std::string.
// Return default_value if obj.ptr() is nullptr, or if exception is caught.
std::string ToPyStr(const py::object &obj, const std::string &default_value);
void ClearCNodeAbstract(const FuncGraphPtr &func_graph);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_RESOLVE_H_
