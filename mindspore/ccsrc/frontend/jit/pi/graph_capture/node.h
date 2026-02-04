/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_NODE_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_NODE_H

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <optional>
#include <memory>
#include "utils/log_adapter.h"
#include "frontend/jit/pi/graph_capture/abstract_object.h"
#include "frontend/jit/pi/graph_capture/constant_info.h"
#include "frontend/jit/pi/graph_guard/trace.h"
#include "frontend/jit/pi/graph_capture/abstract_wrapper.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "frontend/jit/pi/utils/inline_reason.h"

namespace mindspore {
namespace pijit {
class Graph;
class Block;

class AbstractNode {
 public:
  enum Type {
    Abstract,
    kInstr,
    Value,
    Call,      // call node, it is also a value produced operation
    Param,     // parameter value node
    CellVar,   // cell value node
    FreeVar,   // free value node
    kUnbound,  // unbound value node
  };
  explicit AbstractNode(Type t) : type_(t), graph_(nullptr), block_(nullptr), marker_(0) {}
  virtual ~AbstractNode() {}

  Type GetType() const { return type_; }
  Graph *GetGraph() const { return graph_; }
  void SetGraph(Graph *g) { graph_ = g; }
  Block *GetBlock() { return block_; }
  void SetBlock(Block *b) { block_ = b; }

  virtual std::string ToString() const;

 private:
  const Type type_;
  Graph *graph_;
  Block *block_;

 public:
  // remove it
  int marker_;  // for visit
};

class InstrNode : public AbstractNode {
 public:
  InstrNode(int op, int arg) : AbstractNode(kInstr), op_(op), arg_(arg) {}
  virtual ~InstrNode() {}
  int GetOpcode() const { return op_; }
  int GetOparg() const { return arg_; }
  int GetLineNo() const { return line_; }
  void SetOparg(int arg) { this->arg_ = arg; }
  void SetOpcode(int op) { this->op_ = op; }
  void SetLineNo(int l) { this->line_ = l; }
  void SetName(const std::string &n) { name_ = n; }
  const std::string &GetName() const { return name_; }
  std::string ToString() const override;

  int bci() const { return bci_; }
  void set_bci(int i) { bci_ = i; }

  AbstractWrapperPtr abstract_wrapper() const { return abstract_wrapper_; }
  void set_abstract_wrapper(const AbstractWrapperPtr &abstract_wrapper) { abstract_wrapper_ = abstract_wrapper; }
  bool has_abstract_wrapper() const { return abstract_wrapper_ != nullptr; }

 protected:
  InstrNode(Type t, int op, int arg) : AbstractNode(t), op_(op), arg_(arg), line_(-1) {}

 private:
  int bci_ = -1;
  int op_;
  int arg_;
  int line_ = -1;
  std::string name_;
  AbstractWrapperPtr abstract_wrapper_ = nullptr;
};

class ValueNode : public InstrNode {
 public:
  static ValueNode kUnboundLocal;
  static ValueNode kStackNull;

  ValueNode(AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : InstrNode(Value, opcode, oparg), vobj_(vobj), inputs_(inputs) {}
  virtual ~ValueNode() {}

  std::vector<ValueNode *> &inputs() { return inputs_; }
  const std::vector<ValueNode *> &inputs() const { return inputs_; }
  ValueNode *input(int i) const { return inputs_[i]; }
  void AddInput(ValueNode *v) { inputs_.push_back(v); }
  void ClearInputs() { inputs_.clear(); }

  void SetVobj(AObject *vobj);
  const auto GetVobj() const { return vobj_ == nullptr ? vobj_ : vobj_->GetLatestVersion(); }
  const auto &GetOwnVobj() const { return vobj_; }
  AObject *get_attr(const std::string &nam);

  std::string ToString() const override;

  bool IsConstantValue() const;
  void SetConstantValue(bool constant);
  const std::unique_ptr<ConstantInfo> &MakeConstantInfo();
  const std::unique_ptr<ConstantInfo> &GetConstantInfo() const { return constant_info_; }

  TracePtr GetTrace() { return trace_; }
  void SetTrace(TracePtr t) { trace_ = t; }
  const auto GetScope() const { return vobj_ == nullptr ? AObject::Scope::SCOPE_NOT_SPECIFIED : vobj_->GetScope(); }
  void SetScope(AObject::Scope scope) { vobj_ == nullptr ? void(0) : vobj_->SetScope(scope); }
  void AddScope(AObject::Scope scope) { vobj_ == nullptr ? void(0) : vobj_->AddScope(scope); }
  const auto GetScopeDesc() const { return vobj_ == nullptr ? "SCOPE_NOT_SPECIFIED" : vobj_->GetScopeDesc(); }
  void MarkVmNode() { flag_ = (flag_ & ~NODE_GRAPH) | MODE_BYTECODE; }
  bool IsVmNode() const { return flag_ & MODE_BYTECODE; }
  void MarkGraphNode() { flag_ = (flag_ & ~MODE_BYTECODE) | NODE_GRAPH; }
  bool IsGraphNode() const { return flag_ & NODE_GRAPH; }
  void MarkVmGraphNode() { flag_ = flag_ | NODE_GRAPH | MODE_BYTECODE; }
  bool IsSideEffectNode() const { return flag_ & NODE_SIDE_EFFECT; }
  void MarkSideEffectNode() { flag_ = flag_ | NODE_SIDE_EFFECT; }

 protected:
  ValueNode(Type type, AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : InstrNode(type, opcode, oparg), vobj_(vobj), inputs_(inputs) {}

 private:
  static constexpr int MODE_BYTECODE = 1;
  static constexpr int NODE_GRAPH = 1 << 1;
  static constexpr int NODE_SIDE_EFFECT = 2 << 1;

  // value info
  AObject *vobj_;

  // constant info
  std::unique_ptr<ConstantInfo> constant_info_;

  // which nodes are used, ordered parameter
  std::vector<ValueNode *> inputs_;

  // Trace cache to be reused
  TracePtr trace_;

  // mark the node used in bytecode(VM) or graph
  int flag_{MODE_BYTECODE};
};

// simulate PyCellObject, oparg is index
class CellVarNode : public ValueNode {
 public:
  explicit CellVarNode(Type t) : ValueNode(t, nullptr, LOAD_CLOSURE, 0), val_(nullptr) {}

  // The object stored in this cell
  auto GetValue() const { return val_; }
  void SetValue(ValueNode *v) { val_ = v; }
  const auto &GetCellOper() const { return cell_oper_; }
  auto &GetCellOper() { return cell_oper_; }
  void AddCellOper(ValueNode *i) { cell_oper_.push_back(i); }
  virtual ~CellVarNode() {}

 private:
  ValueNode *val_;
  std::vector<ValueNode *> cell_oper_;  // record cell operation
};

class ParamNode : public ValueNode {
 public:
  ParamNode(AObject *o, int index) : ValueNode(Param, o, 0, index, {}) {}
  std::string ToString() const override;
  bool IsMixedPrecisionType() { return mixedPrecisionType_ != nullptr; }
  PyObject *GetMixedPrecisionType() { return mixedPrecisionType_; }
  void SetMixedPrecisionType(PyObject *type) { mixedPrecisionType_ = type; }
  virtual ~ParamNode() {}

 protected:
  PyObject *mixedPrecisionType_{nullptr};
};

class CallNode : public ValueNode {
 public:
  CallNode(int opcode, int oparg, const std::vector<ValueNode *> &inputs);
  virtual ~CallNode() {}

  // python3.11 ~ python3.13 only
  void set_kw_names(const py::object &kw) { kw_names_ = kw; }
  const auto &kw_names() const { return kw_names_; }

  Graph *GetSubGraph() const { return sub_graph_; }
  void SetSubGraph(Graph *n);
  bool IsCallKW();
  bool IsCallEX();

  // The input arguments when calling subgraph's FuncGraph.
  const std::vector<AbstractWrapperPtr> &subgraph_args() const { return subgraph_args_; }
  void set_subgraph_args(const std::vector<AbstractWrapperPtr> &subgraph_args) { subgraph_args_ = subgraph_args; }

  std::string ToString() const override;
  void SetInlineReason(InlineReason r) { reason_ = r; }
  InlineReason GetInlineReason() { return reason_; }

  void AddParam(ValueNode *p) { params_.push_back(p); }

  const auto &GetParams() const { return params_; }
  std::vector<py::object> GetArgs();

  ValueNode *GetSelf() const;

  void UpdateVobj();

 private:
  // sub-graph if traced function
  Graph *sub_graph_;
  // The input arguments when calling subgraph's FuncGraph.
  std::vector<AbstractWrapperPtr> subgraph_args_{};

  InlineReason reason_ = InlineReason::kInlineUnknown;

  std::vector<ValueNode *> params_;  // extra values for inline function

  py::object kw_names_;
};

class IterNode : public ValueNode {
 public:
  IterNode(ValueNode *iterable, AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : ValueNode(vobj, opcode, oparg, inputs), iterable_(iterable), index_(0) {}
  ~IterNode() override = default;

  ValueNode *iterable() const { return iterable_; }
  void set_iterable(ValueNode *iterable_node) { iterable_ = iterable_node; }
  size_t index() const { return index_; }
  void set_index(size_t idx) { index_ = idx; }

 private:
  ValueNode *iterable_;
  size_t index_;
};

bool IsNonLocalValue(ValueNode *i);

std::string ToString(const pijit::AbstractNode *node);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_NODE_H
