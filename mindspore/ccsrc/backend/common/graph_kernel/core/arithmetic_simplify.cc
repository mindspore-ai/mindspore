/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/arithmetic_simplify.h"

#include <algorithm>
#include <list>
#include <string>
#include <functional>
#include <set>
#include <vector>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/model/graph_builder.h"

namespace mindspore::graphkernel {
// operator which follows commutative rules
static mindspore::HashSet<std::string> commutative_ops{"Add", "Mul"};

class PatternNode;
using PatternNodePtr = std::shared_ptr<PatternNode>;
using PatternNodePtrList = std::vector<PatternNodePtr>;

class PatternNode {
 public:
  explicit PatternNode(const std::string &op) : op_(op) {}
  ~PatternNode() = default;
  std::string op() const { return op_; }
  std::vector<PatternNodePtr> inputs() const { return inputs_; }
  void AddInput(const PatternNodePtr &input) { inputs_.push_back(input); }

 private:
  std::string op_ = "";  // ex. "Add","const1","A","0.5" (any op, const or parameter)
  std::vector<PatternNodePtr> inputs_;
};

using ParaMap = mindspore::HashMap<char, inner::NodePtr>;
using ConstMap = mindspore::HashMap<std::string, inner::NodePtr>;

/* This class works to store a kind of pattern tree; it needs a string expression to construct;
 Ex."Pow(Exp(A),B)=Exp(Mul(A,B))"
 then the left tree is
          A                             A    B
           \                             \  /
            Exp    B                       Mul
             \   /                           \
 left tree:   Pow         right tree:         Exp
 lhs_root_ is Pow ;lhs_root_ is Exp */
class PatternTree {
 public:
  // pattern_str->ex."Pow(Exp(A),B)=Exp(Mul(A,B))"
  explicit PatternTree(const std::string &pattern_str) { (void)BuildTree(pattern_str); }
  virtual ~PatternTree() = default;

  PatternNodePtr lhs_root() { return lhs_root_; }
  PatternNodePtr rhs_root() { return rhs_root_; }
  std::string GetRootOp() const { return lhs_root_ == nullptr ? "" : lhs_root_->op(); }
  // build tree with expression string
  PatternNodePtr BuildTree(const std::string &pattern_str);
  // traverse pattern tree, return order is topological order
  void DfsTraverse(const std::shared_ptr<PatternNodePtrList> &res, const PatternNodePtr &cur) const;
  // leverage pattern tree node and lite node's mapping relation to build lite node graph from pattern tree's right
  // side
  inner::NodePtr AlterGraph(const std::shared_ptr<ParaMap> &para_to_ref, const std::shared_ptr<ConstMap> &const_to_ref,
                            const inner::NodePtr &origin_root);
  // invoke DfsMatchGraph
  inner::NodePtrList MatchGraph(const inner::NodePtr &root, const std::shared_ptr<ParaMap> &para_to_ref,
                                const std::shared_ptr<ConstMap> &const_to_ref);

 protected:
  // set attributes for certain pattern node if needed;
  virtual mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &) {
    auto right_pattern = std::make_shared<PatternNodePtrList>();
    DfsTraverse(right_pattern, rhs_root_);
    mindspore::HashMap<PatternNodePtr, inner::DAttrs> attrs_map;
    for (auto &i : (*right_pattern)) {
      attrs_map[i] = {};
    }
    return attrs_map;
  }
  // check attributes meet requirements for certain pattern node if needed;
  virtual bool CheckAttributes(const inner::NodePtr &) const { return true; }

 private:
  PatternNodePtr lhs_root_ = nullptr;  // left side's root
  PatternNodePtr rhs_root_ = nullptr;  // right side's root
};

std::string CutStr(const string &s, size_t start_pos = 0, size_t len = std::string::npos) {
  std::string new_str = "";
  if (start_pos >= s.length()) {
    MS_LOG(EXCEPTION) << "Start index " << start_pos << " is out of range [0, " << s.length() << ") in string: " << s;
  }
  for (size_t i = 0; i < len; i++) {
    if (start_pos + i >= s.length()) {
      break;
    }
    new_str += s[start_pos + i];
  }
  return new_str;
}

bool StartWith(const std::string &s, const std::string &prefix) {
  if (s.length() < prefix.length()) {
    return false;
  }
  return s.find(prefix) == 0;
}

// build pattern tree ;left side's root is lhs_root_ ; right side's root is rhs_root_
PatternNodePtr PatternTree::BuildTree(const std::string &pattern_str) {
  size_t pos = pattern_str.find("=");
  if (pos != std::string::npos) {
    auto left_expression = CutStr(pattern_str, 0, pos);
    lhs_root_ = BuildTree(left_expression);
    auto right_expression = CutStr(pattern_str, pos + 1);
    rhs_root_ = BuildTree(right_expression);
  } else {
    size_t p_start = pattern_str.find("(");
    if (p_start != std::string::npos) {
      size_t p_end = pattern_str.rfind(")");
      auto op_name = CutStr(pattern_str, 0, p_start);
      auto op_inputs = CutStr(pattern_str, p_start + 1, (p_end - p_start) - 1);
      PatternNodePtr cur_node = std::make_shared<PatternNode>(op_name);
      int tmp = 0;
      size_t comma = 0;
      while (comma < op_inputs.length()) {
        if (op_inputs[comma] == '(') {
          tmp++;
        }
        if (op_inputs[comma] == ')') {
          tmp--;
        }
        if (op_inputs[comma] == ',' && tmp == 0) {
          auto first_half = CutStr(op_inputs, 0, comma);
          cur_node->AddInput(BuildTree(first_half));
          auto second_half = CutStr(op_inputs, comma + 1);
          op_inputs = second_half;
          comma = 0;
        } else {
          comma++;
        }
      }
      cur_node->AddInput(BuildTree(op_inputs));
      return cur_node;
    } else {
      return std::make_shared<PatternNode>(pattern_str);
    }
  }
  return nullptr;
}

inner::NType PatternNodeType(const std::string &n) {
  // return (Primitive， Parameter or Value)
  if (n.length() > 0 && n[n.length() - 1] >= '0' && n[n.length() - 1] <= '9') {
    return inner::NType::Value;
  } else if (n.length() == 1 && n[0] >= 'A' && n[0] <= 'Z') {
    return inner::NType::Parameter;
  } else {
    return inner::NType::Primitive;
  }
}

std::string CleanStr(const std::string &s) {
  std::string res = "";
  (void)std::for_each(s.begin(), s.end(), [&res](const char &c) {
    if (c != '[' && c != ']' && c != ' ') {
      res += c;
    }
  });
  return res;
}

bool CheckCurNode(const inner::NodePtr &tmp_node, const std::string &tmp_pattern_op,
                  const std::shared_ptr<ParaMap> &para_to_ref, const std::shared_ptr<ConstMap> &const_to_ref) {
  // put lite graph node's mapping to pattern node into "para_to_ref" and "const_to_ref"
  switch (PatternNodeType(tmp_pattern_op)) {
    case inner::NType::Parameter: {
      if (para_to_ref->find(tmp_pattern_op[0]) != para_to_ref->end()) {
        if ((*para_to_ref)[tmp_pattern_op[0]] != tmp_node) {
          return false;
        }
      } else {
        (*para_to_ref)[tmp_pattern_op[0]] = tmp_node;
      }
      break;
    }
    case inner::NType::Value: {
      if (tmp_node->NodeType() != inner::NType::Value) {
        return false;
      }
      auto node_value_str = std::static_pointer_cast<inner::ConstTensorNode>(tmp_node)->ToString();
      double node_value = std::stod(CleanStr(node_value_str));
      if (StartWith(tmp_pattern_op, "const")) {
        if (const_to_ref->find(tmp_pattern_op) != const_to_ref->end()) {
          auto pattern_value_str =
            std::static_pointer_cast<inner::ConstTensorNode>((*const_to_ref)[tmp_pattern_op])->ToString();
          double pattern_value = std::stod(CleanStr(pattern_value_str));
          if (pattern_value != node_value) {
            return false;
          }
        } else {
          (*const_to_ref)[tmp_pattern_op] = tmp_node;
        }
      } else {
        double pattern_value = std::stod(tmp_pattern_op);
        if (pattern_value != node_value) {
          return false;
        }
      }
      break;
    }
    case inner::NType::Primitive: {
      if (tmp_node->NodeType() != inner::NType::Primitive ||
          std::static_pointer_cast<inner::PrimOp>(tmp_node)->op() != tmp_pattern_op) {
        return false;
      }
      break;
    }
    default:
      break;
  }
  return true;
}

// recursion for thr match of lite node graph and pattern tree's left side, store the mapping of pattern tree node to
// lite graph
bool DfsMatchGraph(const inner::NodePtr &tmp_node, const PatternNodePtr &tmp_pattern,
                   const std::shared_ptr<ParaMap> &para_to_ref, const std::shared_ptr<ConstMap> &const_to_ref,
                   const std::shared_ptr<inner::NodePtrList> &res) {
  std::string tmp_pattern_op = tmp_pattern->op();
  if (!CheckCurNode(tmp_node, tmp_pattern_op, para_to_ref, const_to_ref)) {
    return false;
  }
  std::vector<PatternNodePtr> tmp_pattern_inputs = tmp_pattern->inputs();
  auto tmp_node_inputs = tmp_node->inputs();
  // check if a node meets requiremnet，and DFS check its inputs
  if (tmp_pattern_inputs.size() != 0 && tmp_node_inputs.size() != tmp_pattern_inputs.size()) {
    return false;
  }
  if (PatternNodeType(tmp_pattern_op) == inner::NType::Primitive) {
    // exchange inputs for the node who meets commutative rules
    if (commutative_ops.find(tmp_pattern_op) != commutative_ops.end()) {
      ParaMap para_to_ref_copy = *para_to_ref;
      ConstMap const_to_ref_copy = *const_to_ref;
      bool first_match = DfsMatchGraph(tmp_node_inputs[0], tmp_pattern_inputs[0], para_to_ref, const_to_ref, res) &&
                         DfsMatchGraph(tmp_node_inputs[1], tmp_pattern_inputs[1], para_to_ref, const_to_ref, res);
      if (!first_match) {
        res->clear();
        para_to_ref->clear();
        const_to_ref->clear();
        for (auto &i : para_to_ref_copy) {
          (*para_to_ref)[i.first] = i.second;
        }
        for (auto &i : const_to_ref_copy) {
          (*const_to_ref)[i.first] = i.second;
        }
        bool second_match = DfsMatchGraph(tmp_node_inputs[0], tmp_pattern_inputs[1], para_to_ref, const_to_ref, res) &&
                            DfsMatchGraph(tmp_node_inputs[1], tmp_pattern_inputs[0], para_to_ref, const_to_ref, res);
        if (!second_match) {
          return false;
        }
      }
    } else {
      for (size_t i = 0; i < tmp_pattern_inputs.size(); i++) {
        if (!DfsMatchGraph(tmp_node_inputs[i], tmp_pattern_inputs[i], para_to_ref, const_to_ref, res)) {
          return false;
        }
      }
    }
    res->push_back(tmp_node);
  }
  return true;
}

// traverse pattern tree and return topological order
void PatternTree::DfsTraverse(const std::shared_ptr<PatternNodePtrList> &res, const PatternNodePtr &cur) const {
  if (cur == nullptr) {
    return;
  }
  for (auto &p : cur->inputs()) {
    if (PatternNodeType(p->op()) == inner::NType::Primitive) {
      DfsTraverse(res, p);
    }
  }
  res->push_back(cur);
}

// invoke DfsMatchGraph
inner::NodePtrList PatternTree::MatchGraph(const inner::NodePtr &root, const std::shared_ptr<ParaMap> &para_to_ref,
                                           const std::shared_ptr<ConstMap> &const_to_ref) {
  auto res = std::make_shared<inner::NodePtrList>();
  if (!DfsMatchGraph(root, lhs_root_, para_to_ref, const_to_ref, res)) {
    return {};
  }
  if (CheckAttributes(root)) {
    return *res;
  }
  return {};
}

// leverage pattern tree node and lite node's mapping relation to build new lite node graph from pattern tree's right
// side
inner::NodePtr PatternTree::AlterGraph(const std::shared_ptr<ParaMap> &para_to_ref,
                                       const std::shared_ptr<ConstMap> &const_to_ref,
                                       const inner::NodePtr &origin_root) {
  auto res = std::make_shared<PatternNodePtrList>();
  DfsTraverse(res, rhs_root_);
  auto all_attrs = SetAttributes(origin_root);
  inner::GraphBuilder gb("");
  mindspore::HashMap<PatternNodePtr, inner::NodePtr> pattern_to_ref;
  for (auto &n : (*res)) {
    if (PatternNodeType(n->op()) != inner::NType::Primitive) {
      continue;
    }
    inner::NodePtrList inputs;
    for (auto &i : n->inputs()) {
      if (PatternNodeType(i->op()) == inner::NType::Primitive) {
        inputs.push_back(pattern_to_ref[i]);
      } else if (PatternNodeType(i->op()) == inner::NType::Parameter) {
        inputs.push_back((*para_to_ref)[i->op()[0]]);
      } else {
        if (StartWith(i->op(), "const")) {
          inputs.push_back((*const_to_ref)[i->op()]);
        } else {
          tensor::TensorPtr data = std::make_shared<tensor::Tensor>(static_cast<double>(std::stof(i->op())));
          inputs.push_back(gb.Value(data));
        }
      }
    }
    auto p = gb.Emit(n->op(), inputs, all_attrs[n]);
    pattern_to_ref[n] = p;
  }
  auto &alter_graph = gb.Get()->ops();
  if (alter_graph.empty()) {
    if (PatternNodeType(rhs_root_->op()) == inner::NType::Parameter) {
      return (*para_to_ref)[rhs_root_->op()[0]];
    } else {
      if (StartWith(rhs_root_->op(), "const")) {
        return (*const_to_ref)[rhs_root_->op()];
      } else {
        tensor::TensorPtr data = std::make_shared<tensor::Tensor>(static_cast<double>(std::stof(rhs_root_->op())));
        return gb.Value(data);
      }
    }
  }
  return alter_graph.back();
}

// Reduce(Reduce(A)) = Reduce(A)
class ExtraReduce1PatternTree : public PatternTree {
 public:
  explicit ExtraReduce1PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ExtraReduce1PatternTree() = default;

 protected:
  bool CheckAttributes(const inner::NodePtr &origin_root) const override {
    return (GetValue<bool>((origin_root->inputs()[0])->attrs().find("keep_dims")->second) ==
            GetValue<bool>(origin_root->attrs().find("keep_dims")->second));
  }
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    std::vector<int64_t> axis;
    std::set<int64_t> axis_set;
    auto first_reduce = origin_root->inputs()[0];
    bool keep_dims = GetValue<bool>(origin_root->attrs().find("keep_dims")->second);
    if (keep_dims) {
      for (auto &i : GetValue<std::vector<int64_t>>(origin_root->attrs().find("axis")->second)) {
        (void)axis_set.insert(i);
      }
      for (auto &i : GetValue<std::vector<int64_t>>(first_reduce->attrs().find("axis")->second)) {
        (void)axis_set.insert(i);
      }
    } else {
      auto first_axis = GetValue<std::vector<int64_t>>(first_reduce->attrs().find("axis")->second);
      auto second_axis = GetValue<std::vector<int64_t>>(origin_root->attrs().find("axis")->second);
      std::set<int64_t> st(first_axis.begin(), first_axis.end());
      mindspore::HashMap<int64_t, int64_t> mp;
      int64_t shift = 0;
      for (int64_t n = 0; n < SizeToLong(first_reduce->inputs()[0]->shape.size()); n++) {
        if (st.find(n) != st.end()) {
          shift++;
        } else {
          mp[n - shift] = n;
        }
      }
      (void)std::for_each(first_axis.begin(), first_axis.end(), [&axis_set](auto &i) { (void)axis_set.insert(i); });
      (void)std::for_each(second_axis.begin(), second_axis.end(),
                          [&axis_set, &mp](auto &i) { (void)axis_set.insert(mp[i]); });
    }
    (void)std::copy(axis_set.begin(), axis_set.end(), std::back_inserter(axis));
    attrs_map[this->rhs_root()] = {{"keep_dims", MakeValue(keep_dims)}, {"axis", MakeValue(axis)}};
    return attrs_map;
  }
};

// "ReduceSum(Neg(A))=Neg(ReduceSum(A))"
class ExtraReduce2PatternTree : public PatternTree {
 public:
  explicit ExtraReduce2PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ExtraReduce2PatternTree() = default;

 protected:
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    bool keep_dims = GetValue<bool>(origin_root->attrs().find("keep_dims")->second);
    auto axis = GetValue<std::vector<int64_t>>(origin_root->attrs().find("axis")->second);
    attrs_map[this->rhs_root()->inputs()[0]] = {{"keep_dims", MakeValue(keep_dims)}, {"axis", MakeValue(axis)}};
    return attrs_map;
  }
};

// "LayoutTransform(LayoutTransform(A))=A"
class LayoutTransform1PatternTree : public PatternTree {
 public:
  explicit LayoutTransform1PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~LayoutTransform1PatternTree() = default;

 protected:
  bool CheckAttributes(const inner::NodePtr &origin_root) const override {
    return (GetValue<string>((origin_root->inputs()[0])->attrs().find("src_format")->second) ==
            GetValue<string>(origin_root->attrs().find("dst_format")->second));
  }
};

// "LayoutTransform(LayoutTransform(A))=LayoutTransform(A)"
class LayoutTransform2PatternTree : public PatternTree {
 public:
  explicit LayoutTransform2PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~LayoutTransform2PatternTree() = default;

 protected:
  bool CheckAttributes(const inner::NodePtr &origin_root) const override {
    return (GetValue<string>((origin_root->inputs()[0])->attrs().find("src_format")->second) !=
            GetValue<string>(origin_root->attrs().find("dst_format")->second));
  }
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{"src_format", origin_root->inputs()[0]->attrs().find("src_format")->second},
                                   {"dst_format", origin_root->attrs().find("dst_format")->second}};
    return attrs_map;
  }
};

/*       A
        /
       Neg
       /  \
    Neg     Mul
 Here we cannot transform Neg(Neg(A)) to A because Neg(A) is a input of Mul. OutsideRely is responsible for checking
 this case.
 */
bool OutsideRely(const inner::NodePtrList &nodes, const inner::NodePtr &root) {
  mindspore::HashSet<inner::Node *> nodes_can_simplify;
  (void)std::for_each(nodes.begin(), nodes.end(),
                      [&nodes_can_simplify](auto n) { (void)nodes_can_simplify.insert(n.get()); });
  for (auto &n : nodes) {
    if (n == root) {
      continue;
    }
    for (auto &usr : n->users()) {
      if (nodes_can_simplify.find(usr.first) == nodes_can_simplify.end()) {
        return true;
      }
    }
  }
  return false;
}

struct Expression {
  size_t id;
  std::string math_expr;
  std::function<PatternTreePtr(const std::string &)> func;
};

#define EXPR_PATTERN(cls) [](const std::string &expr) -> PatternTreePtr { return std::make_shared<cls>(expr); }

static std::vector<Expression> expressions = {
  // add
  {1, "Add(A,0)=A", EXPR_PATTERN(PatternTree)},
  {2, "Add(Mul(A,C),Mul(A,B))=Mul(A,Add(B,C))", EXPR_PATTERN(PatternTree)},
  {3, "Add(Add(A,const1),const2)=Add(A,Add(const1,const2))", EXPR_PATTERN(PatternTree)},
  {4, "Add(A,Neg(A))=0", EXPR_PATTERN(PatternTree)},
  {5, "Add(Add(A,B),Neg(A))=B", EXPR_PATTERN(PatternTree)},
  {6, "Add(Add(A,B),Add(Neg(A),C))=Add(B,C)", EXPR_PATTERN(PatternTree)},
  // sub
  {7, "Sub(A,0)=A", EXPR_PATTERN(PatternTree)},
  {8, "Sub(A,const1)=Add(A,Neg(const1))", EXPR_PATTERN(PatternTree)},
  {9, "Sub(Mul(A,C),Mul(A,B))=Mul(A,Sub(B,C))", EXPR_PATTERN(PatternTree)},
  {10, "Sub(Mul(A,C),Mul(B,C))=Mul(Sub(A,B),C)", EXPR_PATTERN(PatternTree)},
  // log
  {11, "Log(Exp(A))=A", EXPR_PATTERN(PatternTree)},
  {12, "Log(Pow(A,B))=Mul(B,Log(Abs(A)))", EXPR_PATTERN(PatternTree)},
  {13, "Log(Sqrt(A))=Mul(0.5,Log(A))", EXPR_PATTERN(PatternTree)},
  {14, "Log(Rsqrt(A))=Mul(-0.5,Log(A))", EXPR_PATTERN(PatternTree)},
  // pow
  {15, "Pow(A,1)=A", EXPR_PATTERN(PatternTree)},
  {16, "Pow(Exp(A),B)=Exp(Mul(A,B))", EXPR_PATTERN(PatternTree)},
  {17, "Pow(A,2)=Mul(A,A)", EXPR_PATTERN(PatternTree)},
  {18, "Pow(A,-1)=Reciprocal(A)", EXPR_PATTERN(PatternTree)},
  // sqrt
  {19, "Sqrt(Mul(A,A))=Abs(A)", EXPR_PATTERN(PatternTree)},
  {20, "Rsqrt(Pow(A,-2))=Abs(A)", EXPR_PATTERN(PatternTree)},
  {21, "Rsqrt(RealDiv(1,A))=Sqrt(A)", EXPR_PATTERN(PatternTree)},
  {22, "Rsqrt(Reciprocal(A))=Sqrt(A)", EXPR_PATTERN(PatternTree)},
  // select
  {23, "Select(A,B,B)=B", EXPR_PATTERN(PatternTree)},
  // Neg
  {24, "Neg(Neg(A))=A", EXPR_PATTERN(PatternTree)},
  // mul
  {25, "Mul(Mul(A,const1),Mul(B,const2))=Mul(Mul(A,B),Mul(const1,const2))", EXPR_PATTERN(PatternTree)},
  {26, "Mul(Mul(A,const1),const2)=Mul(A,Mul(const1,const2))", EXPR_PATTERN(PatternTree)},
  {27, "Mul(Exp(A),Exp(B))=Exp(Add(A,B))", EXPR_PATTERN(PatternTree)},
  {28, "Mul(Mul(Exp(A),C),Exp(B))=Mul(Exp(Add(A,B)),C)", EXPR_PATTERN(PatternTree)},
  {29, "Mul(Mul(Exp(A),C),Mul(Exp(B),D))=Mul(Exp(Add(A,B)),Mul(C,D))", EXPR_PATTERN(PatternTree)},
  {30, "Mul(Sqrt(A),Sqrt(A))=A", EXPR_PATTERN(PatternTree)},
  {31, "Mul(Mul(A,Sqrt(B)),Mul(C,Sqrt(B)))=Mul(Mul(A,B),C)", EXPR_PATTERN(PatternTree)},
  {32, "Mul(Mul(A,Sqrt(B)),Sqrt(B))=Mul(A,B)", EXPR_PATTERN(PatternTree)},
  {33, "Mul(Sqrt(A),Sqrt(B))=Sqrt(Mul(A,B))", EXPR_PATTERN(PatternTree)},
  {34, "Mul(Rsqrt(A),Rsqrt(A))=Reciprocal(A)", EXPR_PATTERN(PatternTree)},
  {35, "Mul(Mul(A,Rsqrt(B)),Rsqrt(B))=RealDiv(A,B)", EXPR_PATTERN(PatternTree)},
  {36, "Mul(Mul(A,Rsqrt(B)),Mul(C,Rsqrt(B)))=RealDiv(Mul(A,C),B)", EXPR_PATTERN(PatternTree)},
  {37, "Mul(Rsqrt(A),Rsqrt(B))=Rsqrt(Mul(A,B))", EXPR_PATTERN(PatternTree)},
  {38, "Mul(A,Rsqrt(A))=Sqrt(A)", EXPR_PATTERN(PatternTree)},
  {39, "Mul(Abs(A),Abs(B))=Abs(Mul(A,B))", EXPR_PATTERN(PatternTree)},
  {40, "Mul(Mul(Abs(A),C),Abs(B))=Mul(Abs(Mul(A,B)),C)", EXPR_PATTERN(PatternTree)},
  {41, "Mul(Mul(Abs(A),C),Mul(Abs(B),D))=Mul(Abs(Mul(A,B)),Mul(C,D))", EXPR_PATTERN(PatternTree)},
  {42, "Mul(Neg(A),const1)=Mul(A,Neg(const1))", EXPR_PATTERN(PatternTree)},
  // realdiv
  {43, "RealDiv(A,1)=A", EXPR_PATTERN(PatternTree)},
  {44, "RealDiv(Exp(A),Exp(B))=Exp(Sub(A,B))", EXPR_PATTERN(PatternTree)},
  {45, "RealDiv(A,Exp(B))=Mul(A,Exp(Neg(B)))", EXPR_PATTERN(PatternTree)},
  {46, "RealDiv(A,Pow(B,const1))=Mul(A,Pow(B,Neg(const1)))", EXPR_PATTERN(PatternTree)},
  {47, "RealDiv(A,Sqrt(A))=Sqrt(A)", EXPR_PATTERN(PatternTree)},
  {48, "RealDiv(A,Sqrt(B))=Mul(A,Rsqrt(B))", EXPR_PATTERN(PatternTree)},
  {49, "RealDiv(A,Rsqrt(B))=Mul(A,Sqrt(B))", EXPR_PATTERN(PatternTree)},
  {50, "RealDiv(A,const1)=Mul(A,Reciprocal(const1))", EXPR_PATTERN(PatternTree)},
  {51, "RealDiv(RealDiv(A,B),RealDiv(C,D))=RealDiv(Mul(A,D),Mul(B,C))", EXPR_PATTERN(PatternTree)},
  {52, "RealDiv(Neg(A),const1)=RealDiv(A,Neg(const1))", EXPR_PATTERN(PatternTree)},
  {53, "RealDiv(RealDiv(A,B),C)=RealDiv(A,Mul(B,C))", EXPR_PATTERN(PatternTree)},
  {54, "RealDiv(A,RealDiv(B,C))=RealDiv(Mul(A,C),B)", EXPR_PATTERN(PatternTree)},
  // reduce1
  {55, "ReduceSum(ReduceSum(A))=ReduceSum(A)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  {56, "ReduceMin(ReduceMin(A))=ReduceMin(A)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  {57, "ReduceMax(ReduceMax(A))=ReduceMax(A)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  // reduce2
  {58, "ReduceSum(Neg(A))=Neg(ReduceSum(A))", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {59, "ReduceSum(RealDiv(A,const1))=RealDiv(ReduceSum(A),const1)", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {60, "ReduceSum(Mul(A,const1))=Mul(ReduceSum(A),const1)", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {61, "CReal(Complex(A,B))=A", EXPR_PATTERN(PatternTree)},
  {62, "CImag(Complex(A,B))=B", EXPR_PATTERN(PatternTree)},
  // lite only
  {63, "LayoutTransform(LayoutTransform(A))=A", EXPR_PATTERN(LayoutTransform1PatternTree)},
  {64, "LayoutTransform(LayoutTransform(A))=LayoutTransform(A)", EXPR_PATTERN(LayoutTransform2PatternTree)},
};

mindspore::HashMap<std::string, std::vector<PatternTreePtr>> GetExpressions() {
  const auto &flags = GraphKernelFlags::GetInstance();
  mindspore::HashMap<std::string, std::vector<PatternTreePtr>> expression_map;
  mindspore::HashSet<std::string> enable_ids{flags.enable_simplify_exprs_only.begin(),
                                             flags.enable_simplify_exprs_only.end()};
  mindspore::HashSet<std::string> disable_ids{flags.disable_simplify_exprs.begin(), flags.disable_simplify_exprs.end()};
  for (auto &e : expressions) {
    if (!enable_ids.empty()) {
      if (enable_ids.count(std::to_string(e.id)) == 0) {
        continue;
      }
    } else {
      if (disable_ids.count(std::to_string(e.id)) > 0) {
        continue;
      }
    }
    PatternTreePtr pt = e.func(e.math_expr);
    expression_map[pt->GetRootOp()].push_back(pt);
  }
  return expression_map;
}

// arithmetic simplify
bool ArithmeticSimplify::DoArithmeticTrans(const inner::LiteGraphPtr &litegraph) {
  auto ops_list = litegraph->ops();
  bool changed = false;
  inner::NodePtrList matched_nodes;
  auto para_to_ref = std::make_shared<ParaMap>();    // A（B，C ...)->Node* mapping
  auto const_to_ref = std::make_shared<ConstMap>();  // const->Node* mapping
  PatternTreePtr cur_pattern;
  auto iter = ops_list.rbegin();
  while (iter != ops_list.rend()) {
    bool can_simplify = false;
    auto this_op = std::static_pointer_cast<inner::PrimOp>(*iter)->op();
    if (expressions_map_.find(this_op) != expressions_map_.end()) {
      for (auto p : expressions_map_[this_op]) {
        cur_pattern = p;
        if (!para_to_ref->empty()) {
          para_to_ref->clear();
        }
        if (!const_to_ref->empty()) {
          const_to_ref->clear();
        }
        // match a pattern;if return is empty,then fails to match
        matched_nodes = p->MatchGraph(*iter, para_to_ref, const_to_ref);
        if (!matched_nodes.empty()) {
          auto right_root_type = PatternNodeType(p->rhs_root()->op());
          if (right_root_type == inner::NType::Primitive && OutsideRely(matched_nodes, *iter)) {
            continue;
          }
          // if no outside rely,then this is a successful match
          can_simplify = true;
          // get the new node to replace
          inner::NodePtr alter_graph_node = cur_pattern->AlterGraph(para_to_ref, const_to_ref, *iter);
          (*iter)->ReplaceWith(alter_graph_node);
          changed = true;
          break;
        }
      }
    }
    if (!can_simplify) {
      ++iter;
    } else {
      break;
    }
  }
  return changed;
}

// constant fold
bool ArithmeticSimplify::DoConstantFold(const inner::LiteGraphPtr &litegraph) {
  auto ops_list = litegraph->GetOrderedNodes();
  bool changed = false;
  auto iter = ops_list.begin();
  while (iter != ops_list.end()) {
    auto this_op = std::static_pointer_cast<inner::PrimOp>(*iter);
    auto value = this_op->InferValue(this_op->inputs(), this_op->attrs());
    if (value != nullptr) {
      (*iter)->ReplaceWith(value);
      ops_list = litegraph->GetOrderedNodes();
      iter = ops_list.begin();
      changed = true;
    } else {
      ++iter;
    }
  }
  return changed;
}

void ReorganizeEmptyGraph(const inner::LiteGraphPtr &litegraph) {
  auto &outputs = litegraph->GetOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->NodeType() == inner::NType::Value) {
      inner::GraphBuilder gb;
      auto op_ptr = gb.Emit("BroadcastTo", {outputs[i]}, {{"shape", MakeValue(outputs[i]->shape)}});
      litegraph->SetOutput(i, op_ptr);
    } else if (outputs[i]->NodeType() == inner::NType::Parameter) {
      inner::GraphBuilder gb;
      auto op_ptr = gb.Emit("Reshape", {outputs[i]}, {{"shape", MakeValue(outputs[i]->shape)}});
      litegraph->SetOutput(i, op_ptr);
    }
  }
  return;
}

bool ArithmeticSimplify::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  bool do_simplify = false;
  expressions_map_ = GetExpressions();
  for (auto node : func_graph->GetOrderedCnodes()) {
    if (AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      inner::LiteGraphPtr lg = GkUtils::AnfGraph2LiteGraph(sub_graph);
      bool find_pattern = true;
      bool change_anf_graph = false;
      while (find_pattern) {
        find_pattern = false;
        find_pattern = DoConstantFold(lg) || find_pattern;
        find_pattern = DoArithmeticTrans(lg) || find_pattern;
        change_anf_graph = change_anf_graph || find_pattern;
      }
      if (!change_anf_graph) {
        continue;
      }
      ReorganizeEmptyGraph(lg);
      auto new_funcgraph = GkUtils::LiteGraph2AnfGraph(lg, Callback::Instance());
      if (new_funcgraph == nullptr) {
        continue;
      }
      new_funcgraph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      auto cnode = node->cast<CNodePtr>();
      AnfNodePtrList inputs(cnode->inputs().begin() + 1, cnode->inputs().end());
      auto new_node = CreateNewFuseCNode(func_graph, new_funcgraph, inputs);
      (void)mng->Replace(node, new_node);
      mng->AddFuncGraph(new_funcgraph);
      do_simplify = true;
    }
  }
  return do_simplify;
}
}  // namespace mindspore::graphkernel
