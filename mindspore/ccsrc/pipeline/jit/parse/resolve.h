/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_RESOLVE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_RESOLVE_H_

#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/parse_base.h"
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"

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
class NameSpace : public Named {
 public:
  NameSpace(const std::string &module, const py::object &obj, const py::object &module_obj = py::object())
      : Named(module), module_(module), obj_(obj), module_obj_(module_obj) {}
  ~NameSpace() override = default;
  MS_DECLARE_PARENT(NameSpace, Named);

  py::object obj() { return obj_; }
  py::object module_obj() { return module_obj_; }
  std::string module() { return module_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<NameSpace>(), std::make_shared<External>());
  }

 private:
  // namespace of the module
  std::string module_;
  // namespace object
  py::object obj_;
  // module object
  py::object module_obj_;
};
using NameSpacePtr = std::shared_ptr<NameSpace>;

// Symbol in NameSpace or Class which shall be resolved.
class Symbol : public Named {
 public:
  explicit Symbol(const std::string &symbol) : Named(symbol), symbol_(symbol) {}
  Symbol(const std::string &symbol, const std::string &name) : Named(name), symbol_(symbol) {}

  ~Symbol() override = default;
  MS_DECLARE_PARENT(Symbol, Named);

  std::string symbol() { return symbol_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<Symbol>(), std::make_shared<External>());
  }

 private:
  std::string symbol_;
};
using SymbolPtr = std::shared_ptr<Symbol>;

class Script : public Named {
 public:
  explicit Script(const std::string &script) : Named(script), script_(script) {}
  Script(const std::string &script, const std::string &name) : Named(name), script_(script) {}

  ~Script() override = default;
  MS_DECLARE_PARENT(Script, Named);

  std::string script() { return script_; }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScript>(shared_from_base<Script>());
  }
  std::string ToString() const override { return "`" + name() + "`"; }

 private:
  std::string script_;
};
using ScriptPtr = std::shared_ptr<Script>;

// PyObjectWrapper class wrappers resolved python object for further processing.
class PyObjectWrapper : public Named {
 public:
  explicit PyObjectWrapper(const py::object &obj, const std::string &name = "Python object") : Named(name), obj_(obj) {}
  ~PyObjectWrapper() override = default;
  MS_DECLARE_PARENT(PyObjectWrapper, Named);
  py::object obj() { return obj_; }

 private:
  // the object that needs to be resolved
  py::object obj_;
};

// InterpretedObject class wrappers interpreted python object.
class InterpretedObject : public PyObjectWrapper {
 public:
  explicit InterpretedObject(const py::object &obj, const std::string &name = "Interpreted object")
      : PyObjectWrapper(obj, name) {}
  ~InterpretedObject() override = default;
  MS_DECLARE_PARENT(InterpretedObject, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override;
};

// ClassObject class wrappers dataclass
class ClassObject : public PyObjectWrapper {
 public:
  explicit ClassObject(const py::object &obj, const std::string &name = "Python dataclass")
      : PyObjectWrapper(obj, name) {}
  ~ClassObject() override = default;
  MS_DECLARE_PARENT(ClassObject, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override;
};

// ClassType class wrappers class name in python
class ClassType : public PyObjectWrapper {
 public:
  explicit ClassType(const py::object &obj, const std::string &name = "Python class type")
      : PyObjectWrapper(obj, name) {}
  ~ClassType() override = default;
  MS_DECLARE_PARENT(ClassType, PyObjectWrapper);
  abstract::AbstractBasePtr ToAbstract() override;
};
using ClassTypePtr = std::shared_ptr<ClassType>;

// SymbolResolver class for resolving symbol extracted from AnfNode.
class SymbolResolver {
 public:
  SymbolResolver(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node)
      : namespace_(name_space), symbol_(symbol), resolved_node_(node) {}

  ~SymbolResolver() = default;

  // resolve symbol in namespace and save it in result_;
  bool Resolve();

  NameSpacePtr get_namespace() { return namespace_; }

  SymbolPtr symbol() { return symbol_; }

  const py::object &result() { return result_; }

  AnfNodePtr resolved_node() { return resolved_node_; }

 private:
  // namespace where the symbol locates
  NameSpacePtr namespace_;
  // the symbol that needs to be resovled
  SymbolPtr symbol_;
  // the node that has been resolved
  AnfNodePtr resolved_node_;
  // Resolve result
  py::object result_;
};
using SymbolResolverPtr = std::shared_ptr<SymbolResolver>;
// Resolve symbol in namespace.
AnfNodePtr ResolveSymbol(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space, const SymbolPtr &symbol,
                         const AnfNodePtr &node);

// Resolve Cell with attr name.
AnfNodePtr ResolveCellwithAttr(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space,
                               const SymbolPtr &symbol, const AnfNodePtr &node, const AnfNodePtr &attr);

// Resolve one graph which normally is the root graph. FuncGraph shall be managed by res->manager().
bool ResolveFuncGraph(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &res, bool use_profile = true);

// Resolve all graphs in manager which is defined outside of pipeline::Resource.
// Mainly used for test cases or resolve graphs which will not be managed by manager.
bool ResolveAll(const FuncGraphManagerPtr &manager);

}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_RESOLVE_H_
