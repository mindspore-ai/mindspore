/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "vm/vmimpl.h"

#include <algorithm>
#include <exception>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <set>

#include "ir/tensor.h"
#include "operator/ops.h"
#include "ir/manager.h"
#include "ir/func_graph_cloner.h"
#include "ir/primitive.h"
#include "utils/convert_utils.h"
#include "utils/primitive_utils.h"
#include "debug/draw.h"

namespace mindspore {
namespace compile {

// Indicate a call to a new frame.
struct CallWrap : public Base {
  explicit CallWrap(const VMFramePtr &vm_frame) : frame(vm_frame) {}
  VMFramePtr frame{nullptr};
};
using CallWrapPtr = std::shared_ptr<CallWrap>;

// Indicates a return with its value.
struct ReturnWrap : public Base {
  explicit ReturnWrap(const BaseRef &r_value) : value(r_value) {}
  BaseRef value{BaseRef()};
};
using ReturnWrapPtr = std::shared_ptr<ReturnWrap>;

VMFrame::VMFrame(const AnfNodePtrList &nodes, const AnfNodePtrToBaseRefMap &values,
                 const AnfNodePtrToBaseRefMap &closure)
    : values_(values), todo_(nodes), closure_(closure) {
  std::reverse(std::begin(todo_), std::end(todo_));
}

const BaseRef VMFrame::operator[](const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto ret = values_.find(node);
  if (ret != values_.end()) {
    return ret->second;
  }

  ret = closure_.find(node);
  if (ret != closure_.end()) {
    return ret->second;
  }

  if (node->isa<ValueNode>()) {
    return GetValueNode(node);
  }

  MS_LOG(EXCEPTION) << "ValueError " << node->type_name();
}

Closure::Closure(const FuncGraphPtr &graph, const AnfNodePtrToBaseRefMap &values)
    : func_graph_(graph), values_(values) {}

BaseRef Closure::operator()(const VectorRef &args) {
  MS_LOG(DEBUG) << "start closure";
  return vm_->Evaluate(func_graph_, args, values_);
}

Partial::Partial(const BaseRef &fn, const VectorRef &args, const VMPtr &vm) : fn_(fn), args_(args), vm_(vm) {}

BaseRef Partial::operator()(const VectorRef &nodes) {
  VectorRef arglist;
  (void)arglist.insert(arglist.end(), args_.begin(), args_.end());
  (void)arglist.insert(arglist.end(), nodes.begin(), nodes.end());
  return vm_->Call(fn_, arglist);
}

SetRef VM::ComputeFvs(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  SetRef rval;
  for (auto &fkv : graph->free_variables_total()) {
    if (utils::isa<FuncGraphPtr>(fkv.first)) {
      // Add all value_nodes of g that refer to a fv graph
      auto g = utils::cast<FuncGraphPtr>(fkv.first);
      for (auto &ctkv : g->value_nodes()) {
        auto ct = ctkv.first;
        if (GetValueNode(ct) == g) {
          (void)rval.insert(ct);
        }
      }
    } else {
      // Add a normal fv
      (void)rval.insert(fkv.first);
    }
  }

  return rval;
}

void VM::AcquireGraph(const FuncGraphPtr &graph) {
  // Already acquired
  if (vars_.find(graph) != vars_.end()) {
    return;
  }
  // Add g to manager
  manager_->AddFuncGraph(graph);
  // Compute fvs for all acquired graph
  auto graphs = graph->manager()->func_graphs();
  for (auto g = graphs.begin(); g != graphs.end(); ++g) {
    vars_[*g] = ComputeFvs(*g);
  }
}

VectorRef VM::ExportSequence(const VectorRef &seq) {
  std::vector<BaseRef> ret;
  (void)std::transform(std::begin(seq), std::end(seq), std::back_inserter(ret),
                       [&, this](const BaseRef &x) -> BaseRef { return Export(x); });
  return VectorRef(ret);
}

ClosurePtr VM::ExportClosure(const ClosurePtr &clos) {
  MS_EXCEPTION_IF_NULL(clos);
  clos->set_vm(shared_from_this());
  return clos;
}

// transform graph to executable closure
ClosurePtr VM::ExportGraph(const FuncGraphPtr &g) {
  auto c = std::make_shared<Closure>(g, AnfNodePtrToBaseRefMap());
  MS_EXCEPTION_IF_NULL(c);
  c->set_vm(shared_from_this());
  return c;
}

BaseRef VM::ExportObj(const BaseRef &obj) const { return obj; }

BaseRef VM::Export(const BaseRef &value) {
  if (utils::isa<ValuePtr>(value) && utils::cast<ValuePtr>(value)->isa<FuncGraph>()) {
    return ExportGraph(utils::cast<ValuePtr>(value)->cast<FuncGraphPtr>());
  }

  if (utils::isa<ValuePtr>(value) && utils::cast<ValuePtr>(value)->isa<Primitive>()) {
    return ExportPrimitive(utils::cast<ValuePtr>(value)->cast<PrimitivePtr>());
  }

  if (utils::isa<FuncGraphPtr>(value)) {
    return ExportGraph(utils::cast<FuncGraphPtr>(value));
  }

  if (utils::isa<ClosurePtr>(value)) {
    return ExportClosure(utils::cast<ClosurePtr>(value));
  }

  if (utils::isa<PrimitivePtr>(value)) {
    return ExportPrimitive(utils::cast<PrimitivePtr>(value));
  }

  if (utils::isa<VectorRef>(value)) {
    return ExportSequence(utils::cast<VectorRef>(value));
  }

  return ExportObj(value);
}

// Run a graph.
// This will evaluate the passed-in graph and return the resulting value.
BaseRef VM::Evaluate(const FuncGraphPtr &graph, const VectorRef &args, const AnfNodePtrToBaseRefMap &closure) {
  AcquireGraph(graph);
  MS_LOG(DEBUG) << "evalue arg size: " << args.size();
  if (args.size() != graph->parameters().size()) {
    MS_LOG(EXCEPTION) << "Call with wrong number of arguments, expect " << graph->parameters().size() << ", but got "
                      << args.size();
  }

  // toposort graph nodes, the order will be reversed by frame so that the dependent be computed first
  auto nodes = TopoSort(graph->get_return(), SuccVm(graph));
  // mapping parameters to args
  AnfNodePtrToBaseRefMap values;
  for (size_t i = 0; i < args.size(); i++) {
    values[graph->parameters()[i]] = args[i];
  }
  // create top frame with params initialized
  VMFramePtrList frames{std::make_shared<VMFrame>(nodes, values, closure)};
  // execute frames starting from top frame
  while (!frames.empty()) {
    auto frame = frames[frames.size() - 1];
    auto todo = frame->todo();
    while (!todo.empty()) {
      auto except = HandleNode(todo[todo.size() - 1], frame);
      if (utils::isa<CallWrapPtr>(except)) {
        if (todo.size() == 2) {
          // The last element is always a return, replace the ret with call frame
          frames[frames.size() - 1] = utils::cast<CallWrapPtr>(except)->frame;
        } else {
          frames.push_back(utils::cast<CallWrapPtr>(except)->frame);
        }
        break;
      }
      if (utils::isa<ReturnWrapPtr>(except)) {
        (void)frames.erase(frames.begin() + (static_cast<ssize_t>(frames.size()) - 1));
        if (frames.size() > 0) {
          auto top = frames[frames.size() - 1];
          auto td = top->todo();
          // set value for top frame's last evaluated node
          if (td.empty()) {
            MS_LOG(EXCEPTION) << "The td is empty";
          }
          top->values()[td[td.size() - 1]] = utils::cast<ReturnWrapPtr>(except)->value;
          (void)td.erase(td.begin() + (static_cast<ssize_t>(td.size()) - 1));
        } else {
          return Export(utils::cast<ReturnWrapPtr>(except)->value);
        }
        break;
      }
      (void)todo.erase(todo.begin() + (static_cast<ssize_t>(todo.size()) - 1));
    }
  }
  MS_LOG(EXCEPTION) << "VM Evaluate error";
}

SuccFunc VM::SuccVm(const FuncGraphPtr &graph) {
  auto fn = [&, this](const AnfNodePtr &node) -> AnfNodePtrList {
    MS_EXCEPTION_IF_NULL(node);
    AnfNodePtrList ret;

    // Follow node.incoming
    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      for (auto &i : inputs) {
        if (i->func_graph() == node->func_graph() ||
            (IsValueNode<FuncGraph>(i) && GetValueNode<FuncGraphPtr>(i)->parent() == graph)) {
          ret.push_back(i);
        }
      }
    }

    // for subgraph input, add their fvs as succ nodes
    if (IsValueNode<FuncGraph>(node) && GetValueNode<FuncGraphPtr>(node)->parent() == graph) {
      auto fvs = utils::cast<SetRef>(vars_[GetValueNode<FuncGraphPtr>(node)]);
      (void)std::transform(fvs.begin(), fvs.end(), std::back_inserter(ret),
                           [](const BaseRef &value) -> AnfNodePtr { return utils::cast<AnfNodePtr>(value); });
    }

    return ret;
  };
  return fn;
}

BaseRef VM::Call(const BaseRef &fn, const VectorRef &args) {
  if (utils::isa<PrimitivePtr>(fn)) {
    return RunOperation(utils::cast<PrimitivePtr>(fn), args);
  }

  if (utils::isa<FuncGraphPtr>(fn)) {
    return Evaluate(utils::cast<FuncGraphPtr>(fn), args);
  }

  if (utils::isa<ClosurePtr>(fn)) {
    auto clos = utils::cast<ClosurePtr>(fn);
    return Evaluate(clos->func_graph(), args, clos->values());
  }

  MS_LOG(EXCEPTION) << "Can't call fn";
}

// make call frame for graph
BaseRef VM::_Call(const BaseRef &graph, const VectorRef &args) {
  AnfNodePtrToBaseRefMap clos;
  auto func_graph = graph;
  if (utils::isa<ClosurePtr>(func_graph)) {
    clos = utils::cast<ClosurePtr>(func_graph)->values();
    func_graph = utils::cast<ClosurePtr>(func_graph)->func_graph();
  }
  if (utils::isa<ValuePtr>(func_graph)) {
    func_graph = utils::cast<ValuePtr>(func_graph)->cast<FuncGraphPtr>();
  }

  if (!utils::isa<FuncGraphPtr>(func_graph)) {
    MS_LOG(EXCEPTION) << "Graph type error";
  }

  auto graphPtr = utils::cast<FuncGraphPtr>(func_graph);

  if (vars_.find(graphPtr) == vars_.end()) {
    AcquireGraph(graphPtr);
  }

  if (args.size() != graphPtr->parameters().size()) {
    MS_LOG(EXCEPTION) << "Call with wrong number of arguments, expect " << graphPtr->parameters().size() << ", but got "
                      << args.size();
  }

  auto nodes = TopoSort(graphPtr->get_return(), SuccVm(graphPtr));
  AnfNodePtrToBaseRefMap values;
  for (size_t i = 0; i < args.size(); i++) {
    values[graphPtr->parameters()[i]] = args[i];
  }

  return std::make_shared<CallWrap>(std::make_shared<VMFrame>(nodes, values, clos));
}

// make closure out of graph with fv values from frame
ClosurePtr VM::MakeClosure(const FuncGraphPtr &graph, const VMFramePtr &frame) {
  MS_EXCEPTION_IF_NULL(frame);
  AnfNodePtrToBaseRefMap clos;

  for (auto &v : utils::cast<SetRef>(vars_[graph])) {
    auto anf = utils::cast<AnfNodePtr>(v);
    clos[anf] = (*frame)[anf];
  }

  return std::make_shared<Closure>(graph, clos);
}

BaseRef VM::DispatchCall(const AnfNodePtr &node, const VMFramePtr &frame, const BaseRef &fn, const VectorRef &args) {
  if (utils::isa<ValuePtr>(fn) && utils::cast<ValuePtr>(fn)->isa<Primitive>()) {
    auto fnval = utils::cast<ValuePtr>(fn)->cast<PrimitivePtr>();
    MS_LOG(DEBUG) << "DispatchCall prim:" << fnval->name() << ", node:" << node->DebugString(true);
    if (args.empty()) {
      MS_LOG(EXCEPTION) << "args is empty";
    }
    if (fnval == prim::kPrimReturn) {
      MS_LOG(DEBUG) << "return args:" << args.size();
      return std::make_shared<ReturnWrap>(args[0]);
    }

    if (fnval == prim::kPrimMakeTuple) {
      frame->values()[node] = args;
      return BaseRef();
    }

    if (fnval == prim::kPrimPartial) {
      VectorRef partial_args(args.begin() + 1, args.end());
      frame->values()[node] = (std::make_shared<Partial>(args[0], partial_args, shared_from_this()));
      return BaseRef();
    }

    // call prim implementation
    frame->values()[node] = RunOperation(fnval, args);
    return BaseRef();
  }

  // partial args logic
  if (utils::isa<PartialPtr>(fn)) {
    auto fnPtr = utils::cast<PartialPtr>(fn);

    VectorRef arglist;
    (void)arglist.insert(arglist.end(), fnPtr->args().begin(), fnPtr->args().end());
    (void)arglist.insert(arglist.end(), args.begin(), args.end());

    auto ret = DispatchCall(node, frame, fnPtr->fn(), arglist);
    if (utils::isa<CallWrapPtr>(ret) || utils::isa<ReturnWrapPtr>(ret)) {
      return ret;
    }
  }

  // create frame for graph and closure
  if ((utils::isa<ValuePtr>(fn) && utils::cast<ValuePtr>(fn)->isa<FuncGraph>()) || utils::isa<ClosurePtr>(fn)) {
    auto ret = _Call(fn, args);
    if (utils::isa<CallWrapPtr>(ret) || utils::isa<ReturnWrapPtr>(ret)) {
      return ret;
    }
  }

  MS_LOG(EXCEPTION) << "Invalid fn to call";
}

BaseRef VM::HandleNode(const AnfNodePtr &node, const VMFramePtr &frame) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>()) {
    // pass
    return BaseRef();
  }

  if (node->isa<ValueNode>()) {
    // We only visit valuenode graphs
    if (!IsValueNode<FuncGraph>(node)) {
      MS_LOG(EXCEPTION) << "We only visit valuenode graphs ";
    }
    auto g = GetValueNode<FuncGraphPtr>(node);

    // if g is a graph with fvs, we need to make a closure for it
    auto iterG = vars_.find(g);
    if (iterG != vars_.end() && utils::cast<SetRef>(iterG->second).size() != 0) {
      frame->values()[node] = MakeClosure(g, frame);
    }

    return BaseRef();
  }

  if (node->isa<CNode>()) {
    std::vector<BaseRef> fnArgs;
    auto &inputs = node->cast<CNodePtr>()->inputs();
    // set args' values in frame
    (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(fnArgs),
                         [&](const AnfNodePtr &inp) -> BaseRef { return (*frame)[inp]; });
    if (fnArgs.empty()) {
      MS_LOG(EXCEPTION) << "function arguments is empty";
    } else {
      auto args = VectorRef(fnArgs.begin() + 1, fnArgs.end());
      auto except = DispatchCall(node, frame, fnArgs[0], args);
      return except;
    }
  }

  MS_LOG(EXCEPTION) << "Unknown node type";
}

VectorRef VM::RunGraph(const FuncGraphPtr &g, const VectorRef &args) {
  this->manager_ = Manage(g);

  auto fn = utils::cast<ClosurePtr>(Export(g));
  auto result = (*fn)(args);

  if (utils::isa<VectorRef>(result)) {
    return utils::cast<VectorRef>(result);
  } else {
    VectorRef ret({result});
    return ret;
  }
}

BaseRef RunOperation(const PrimitivePtr &prim, const VectorRef &args) {
  PrimitivePyPtr operation = dyn_cast<PrimitivePy>(prim);

  MS_LOG(DEBUG) << "operation start " << prim->name();
  auto func = operation != nullptr ? operation->GetComputeFunction() : GetComputeFunction(prim->name());
  if (py::isinstance<py::none>(func)) {
    MS_LOG(EXCEPTION) << prim->name() << " 's compute function is not implemented";
  }

  py::tuple py_args = py::tuple(args.size());
  MS_LOG(DEBUG) << "input for operation:";
  size_t i = 0;
  for (auto &arg : args) {
    py_args[i] = BaseRefToPyData(arg);
    MS_LOG(DEBUG) << "arg: " << i << ":";
    i++;
  }
  py::object obj = func(*py_args);
  MS_LOG(DEBUG) << "result:" << py::str(obj);
  return obj;
}

}  // namespace compile
}  // namespace mindspore
