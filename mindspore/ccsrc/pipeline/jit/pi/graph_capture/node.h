/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_NODE_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_NODE_H

#include <string>
#include <vector>
#include "utils/log_adapter.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace jit {
namespace graph {
class Graph;
class MergeNode;
class Block;
class AbstractNode {
 public:
  enum Type {
    Abstract,
    kInstr,
    Value,
    Call,     // call node, it is also a value produced operation
    Merged,   // merged value node
    Param,    // parameter value node
    CellVar,  // cell value node
    FreeVar,  // free value node
    Unbound,  // unbound value node
  };
  explicit AbstractNode(Type t)
      : marker_(0), next_(nullptr), jump_(nullptr), owner_(nullptr), block_(nullptr), type_(t) {}
  virtual ~AbstractNode() {}

  bool insertBack(AbstractNode *n) {
    if (!n || n->pre_.size() || this->next_) {
      return false;
    }
    n->pre_.push_back(this);
    this->next_ = n;
    return true;
  }

  Type GetType() const { return type_; }
  std::vector<AbstractNode *> &GetPres() { return pre_; }
  void AddPre(AbstractNode *n) { pre_.push_back(n); }
  AbstractNode *GetNext() const { return next_; }
  void SetNext(AbstractNode *v) { next_ = v; }
  AbstractNode *GetJump() const { return jump_; }
  void SetJump(AbstractNode *v) { jump_ = v; }
  Graph *GetGraph() const { return owner_; }
  void SetGraph(Graph *g) { owner_ = g; }
  virtual std::string to_str() const;

  Block *GetBlock() { return block_; }
  void SetBlock(Block *b) { block_ = b; }

  int marker_;  // for visit

 private:
  std::vector<AbstractNode *> pre_;  // pre nodes, in edge
  AbstractNode *next_;               // next nodes, out edge
  AbstractNode *jump_;               // branch target node, out edge
  Graph *owner_;
  Block *block_;
  const Type type_;
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
  std::string to_str() const override;

  int bci() const { return bci_; }
  void set_bci(int i) { bci_ = i; }

 protected:
  InstrNode(Type t, int op, int arg) : AbstractNode(t), op_(op), arg_(arg), line_(-1) {}

 private:
  int bci_ = -1;
  int op_;
  int arg_;
  int line_ = -1;
  std::string name_;
};

class ValueNode : public InstrNode {
 public:
  static ValueNode UnboundLocal;

  ValueNode(AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : InstrNode(Value, opcode, oparg), vobj_(vobj), inputs_(inputs), attr_(false), subscr_(false) {}
  virtual ~ValueNode() {}

  std::vector<ValueNode *> &getInputs() { return inputs_; }
  const std::vector<ValueNode *> &getInputs() const { return inputs_; }
  ValueNode *input(int i) const { return inputs_[i]; }
  void AddInput(ValueNode *v) { inputs_.push_back(v); }
  void ClearInputs() { inputs_.clear(); }

  void SetVobj(AObject *vobj) { vobj_ = vobj; }
  const auto &GetVobj() const { return vobj_; }

  std::map<std::string, ValueNode *> &GetAttrs() { return attrs_; }

  void store_attr(const std::string &nam, ValueNode *v);
  void del_attr(const std::string &nam) {}
  AObject *get_attr(const std::string &nam) { return vobj_ ? vobj_->GetAttr(nam) : nullptr; }

  void store_subscr(ValueNode *sub, ValueNode *v);
  void del_subscr(ValueNode *sub) {}
  AObject *binary_subscr(ValueNode *sub) { return vobj_ ? vobj_->GetItem(sub->GetVobj()) : nullptr; }

  MergeNode *GetOutput(int i) const { return outputs_[i]; }
  void SetOutput(int i, MergeNode *n) { outputs_[i] = n; }
  void AddOutput(MergeNode *n) { outputs_.push_back(n); }
  const auto &GetOutputs() const { return outputs_; }

  bool IsMindsporeSupportedOperation();

  std::string to_str() const override;

 protected:
  ValueNode(Type type, AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : InstrNode(type, opcode, oparg), vobj_(vobj), inputs_(inputs), attr_(false), subscr_(false) {}

 private:
  AObject *vobj_;                             // NOTE: vobj_ is not compute
  std::vector<ValueNode *> inputs_;           // which nodes are used, ordered parameter
  std::vector<MergeNode *> outputs_;          // usage merge nodes
  std::map<std::string, ValueNode *> attrs_;  // store attrs
  bool attr_;                                 // track store attr not implement, marked as modified
  bool subscr_;                               // track store subscr not implement, marked as modified
};

// simulate PyCellObject, oparg is index
class CellVarNode : public ValueNode {
 public:
  explicit CellVarNode(Type t) : ValueNode(t, nullptr, -1, CO_CELL_NOT_AN_ARG), val_(nullptr) {}
  void SetFromParam(int i) { SetOparg(i); }
  int GetFromParam() const { return GetOparg(); }
  void SetIndex(int i) { return SetOpcode(i); }
  int GetIndex() const { return GetOpcode(); }
  auto GetValue() const { return val_; }
  void SetValue(ValueNode *v) { val_ = v; }
  const auto &GetCellOper() const { return cell_oper_; }
  void AddCellOper(InstrNode *i) { cell_oper_.push_back(i); }
  virtual ~CellVarNode() {}
  std::string to_str() const override;

 private:
  ValueNode *val_;
  std::vector<InstrNode *> cell_oper_;  // record cell operation
};

class ParamNode : public ValueNode {
 public:
  ParamNode(AObject *o, int index) : ValueNode(Param, o, 0, index, {}) {}
  std::string to_str() const override;
  virtual ~ParamNode() {}
};

class CallNode : public ValueNode {
 public:
  CallNode(int opcode, int oparg, const std::vector<ValueNode *> &inputs)
      : ValueNode(Call, nullptr, opcode, oparg, inputs), sub_graph_(nullptr), extra_oper_(nullptr) {}
  virtual ~CallNode() {}

  Graph *GetSubGraph() const { return sub_graph_; }
  void SetSubGraph(Graph *n) { sub_graph_ = n; }
  InstrNode *GetExtraOper() const { return extra_oper_; }
  void SetExtraOper(InstrNode *root) { extra_oper_ = root; }
  std::string to_str() const override;
  void SetInlineReason(InlineReason r) { reason_ = r; }
  InlineReason GetInlineReason() { return reason_; }

  void AddParam(ValueNode *p) { params_.push_back(p); }
  const auto &GetParams() const { return params_; }

 private:
  // sub-graph if traced function
  Graph *sub_graph_;

  /**
   * this is unused if not need execute inlined bytecode
   * this is unused if parameter processed by graph executor
   */
  InstrNode *extra_oper_;  // handle parameters by bytecode

  InlineReason reason_ = InlineReason::kInlineUnknown;

  std::vector<ValueNode *> params_;  // handle parameters by bytecode
};

class MergeNode : public ValueNode {
 public:
  MergeNode() : ValueNode(AbstractNode::Merged, nullptr, 0, 0) {}
  virtual ~MergeNode() {}
  void AddInput(ValueNode *);
  void Merge(MergeNode *);
  std::string to_str() const override;

 private:
  std::vector<int> predecessor_index_;
};

// NOTE: not check whether node is in list
class AbstractNodeList {
 public:
  AbstractNodeList() : head_(nullptr), back_(nullptr) {}
  AbstractNodeList(AbstractNode *a, AbstractNode *b) : head_(a), back_(b) {}
  void push_back(AbstractNode *n);
  void push_front(AbstractNode *n);
  bool insert(AbstractNode *pos, AbstractNodeList *l);
  void cutList(AbstractNode *pos, AbstractNodeList *second_half);
  void erase(AbstractNode *n);
  const auto &head() const { return head_; }
  const auto &back() const { return back_; }

 private:
  AbstractNode *head_;
  AbstractNode *back_;
};

bool IsNonLocalValue(ValueNode *i);
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_NODE_H
