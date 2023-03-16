/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_VM_VMIMPL_H_
#define MINDSPORE_CCSRC_VM_VMIMPL_H_

#include <set>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/tensor.h"
#include "pybind_api/ir/base_ref_py.h"

namespace mindspore {
namespace compile {

using AnfNodePtrList = std::vector<AnfNodePtr>;
using AnfNodePtrToBaseRefMap = mindspore::HashMap<AnfNodePtr, BaseRef>;
using AnfNodePtrToAnfNodePtrMap = mindspore::HashMap<AnfNodePtr, AnfNodePtr>;

using FuncGraphPtrToBaseRefMap = mindspore::HashMap<FuncGraphPtr, BaseRef>;

using TensorList = std::vector<tensor::TensorPtr>;

class Closure;
using ClosurePtr = std::shared_ptr<Closure>;

class VMFrame;
using VMFramePtr = std::shared_ptr<VMFrame>;
using VMFramePtrList = std::vector<VMFramePtr>;

class VM;
using VMPtr = std::shared_ptr<VM>;

class Partial;
using PartialPtr = std::shared_ptr<Partial>;

using RunFunc = std::function<VectorRef(const VectorRef &args)>;
using RunFuncPtr = std::shared_ptr<RunFunc>;

using SuccFunc = std::function<AnfNodePtrList(AnfNodePtr node)>;

class VMImpl {
 public:
  virtual VectorRef RunGraph(const FuncGraphPtr &fg, const VectorRef &args) = 0;
  virtual ~VMImpl() = default;
};

// An execution frame.
// This holds the state for an application of a graph. The nodes list
// must contain free variables of graphs encountered before the
// graph themselves.
// You can index a frame with a node to get its value in the context
// of this frame (if it has already been evaluated).
// Attributes:
//   nodes: list of nodes remaining to execute
//   values: Mapping of node to their values in this application
//   closure: values for the closure if the current application is a closure
class VMFrame {
 public:
  VMFrame(const AnfNodePtrList &nodes, const AnfNodePtrToBaseRefMap &values, const AnfNodePtrToBaseRefMap &closure);
  const BaseRef operator[](const AnfNodePtr &node);
  const AnfNodePtrList &todo() const { return todo_; }

  AnfNodePtrToBaseRefMap &values() { return values_; }

  virtual ~VMFrame() = default;

  AnfNodePtrToBaseRefMap values_;

 private:
  AnfNodePtrList todo_;
  AnfNodePtrToBaseRefMap closure_;
};

// Representation of a closure.
class Closure : public Base {
 public:
  Closure(const FuncGraphPtr &graph, const AnfNodePtrToBaseRefMap &values);
  BaseRef operator()(const VectorRef &args);

  const VMPtr &vm() const { return vm_; }

  void set_vm(const VMPtr &vm) { vm_ = vm; }

  const FuncGraphPtr &func_graph() const { return func_graph_; }

  const AnfNodePtrToBaseRefMap &values() const { return values_; }

  virtual ~Closure() = default;

  MS_DECLARE_PARENT(Closure, Base)

 private:
  FuncGraphPtr func_graph_;
  AnfNodePtrToBaseRefMap values_;
  VMPtr vm_;
};

// Representation of a partial application.
class Partial : public Base {
 public:
  Partial(const BaseRef &fn, const VectorRef &args, const VMPtr &vm);
  BaseRef operator()(const VectorRef &nodes);
  const BaseRef &fn() const { return fn_; }

  const VectorRef &args() const { return args_; }

  virtual ~Partial() = default;
  MS_DECLARE_PARENT(Partial, Base)

 private:
  BaseRef fn_;
  VectorRef args_;
  VMPtr vm_;
};

// Virtual Machine interface.
class VM : public std::enable_shared_from_this<VM>, public VMImpl {
 public:
  SetRef ComputeFvs(const FuncGraphPtr &graph) const;

  void AcquireGraph(const FuncGraphPtr &graph);

  VectorRef ExportSequence(const VectorRef &seq);

  BaseRef ExportPrimitive(const PrimitivePtr &) const { return kValueAny; }

  ClosurePtr ExportClosure(const ClosurePtr &clos);

  // Return an object that executes `fg` when called on arguments.
  ClosurePtr ExportGraph(const FuncGraphPtr &g);

  BaseRef ExportObj(const BaseRef &obj) const;

  BaseRef Export(const BaseRef &value);

  // Run a graph.
  // This will evaluate the passed-in graph and return the
  // resulting value.
  BaseRef Evaluate(const FuncGraphPtr &graph, const VectorRef &args,
                   const AnfNodePtrToBaseRefMap &closure = AnfNodePtrToBaseRefMap());

  // Return a visitor for the graph.
  SuccFunc SuccVm(const FuncGraphPtr &graph);

  // Call the `fn` object.
  // `fn` can be anything that would be valid as the first element of an apply.
  BaseRef Call(const BaseRef &fn, const VectorRef &args);

  BaseRef _Call(const BaseRef &graph, const VectorRef &args);

  ClosurePtr MakeClosure(const FuncGraphPtr &graph, const VMFramePtr &frame);

  BaseRef DispatchCall(const AnfNodePtr &node, const VMFramePtr &frame, const BaseRef &fn, const VectorRef &args);

  BaseRef HandleNode(const AnfNodePtr &node, const VMFramePtr &frame);

  VectorRef RunGraph(const FuncGraphPtr &g, const VectorRef &args) override;

 private:
  FuncGraphManagerPtr manager_;
  FuncGraphPtrToBaseRefMap vars_;
};

extern BaseRef RunOperation(const PrimitivePtr &prim, const VectorRef &args);

}  // namespace compile
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_VM_VMIMPL_H_
