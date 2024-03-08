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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_NODE_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_NODE_H

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <optional>
#include <memory>
#include "utils/log_adapter.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/graph_capture/constant_info.h"
#include "pipeline/jit/pi/utils/utils.h"

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
  static ValueNode kUnboundLocal;

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
  AObject *get_attr(const std::string &nam);

  void store_subscr(ValueNode *sub, ValueNode *v);
  void del_subscr(ValueNode *sub) {}
  AObject *binary_subscr(ValueNode *sub);

  std::string ToString() const override;
  ValueNode *GetParent() { return parent_.value_or(nullptr); }
  void SetParent(ValueNode *parent);

  bool IsConstantValue() const;
  void SetConstantValue(bool constant);
  const std::unique_ptr<ConstantInfo> &MakeConstantInfo();
  const std::unique_ptr<ConstantInfo> &GetConstantInfo() const { return constant_info_; }

 protected:
  ValueNode(Type type, AObject *vobj, int opcode, int oparg, const std::vector<ValueNode *> &inputs = {})
      : InstrNode(type, opcode, oparg), vobj_(vobj), inputs_(inputs), attr_(false), subscr_(false) {}

 private:
  // value info
  AObject *vobj_;

  // constant info
  std::unique_ptr<ConstantInfo> constant_info_;

  // which nodes are used, ordered parameter
  std::vector<ValueNode *> inputs_;

  // store attrs
  std::map<std::string, ValueNode *> attrs_;

  // track store attr not implement, marked as modified
  bool attr_;

  // track store subscr not implement, marked as modified
  bool subscr_;

  // recode relationship between local and CallNode
  std::optional<ValueNode *> parent_;
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
  std::string ToString() const override;

 private:
  ValueNode *val_;
  std::vector<InstrNode *> cell_oper_;  // record cell operation
};

class ParamNode : public ValueNode {
 public:
  ParamNode(AObject *o, int index) : ValueNode(Param, o, 0, index, {}) {}
  std::string ToString() const override;
  virtual ~ParamNode() {}
};

class CallNode : public ValueNode {
 public:
  CallNode(int opcode, int oparg, const std::vector<ValueNode *> &inputs)
      : ValueNode(Call, nullptr, opcode, oparg, inputs), sub_graph_(nullptr) {}
  virtual ~CallNode() {}

  Graph *GetSubGraph() const { return sub_graph_; }
  void SetSubGraph(Graph *n);
  std::string ToString() const override;
  void SetInlineReason(InlineReason r) { reason_ = r; }
  InlineReason GetInlineReason() { return reason_; }

  void AddParam(ValueNode *p) {
    params_.push_back(p);
    if (p) {
      p->SetParent(this);
    }
  }

  const auto &GetParams() const { return params_; }
  std::vector<py::object> GetArgs() {
    std::vector<py::object> args;
    std::transform(getInputs().begin() + 1, getInputs().end(), std::back_inserter(args),
                   [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
    return args;
  }

 private:
  // sub-graph if traced function
  Graph *sub_graph_;

  InlineReason reason_ = InlineReason::kInlineUnknown;

  std::vector<ValueNode *> params_;  // extra values for inline function
};

bool IsNonLocalValue(ValueNode *i);
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_NODE_H
