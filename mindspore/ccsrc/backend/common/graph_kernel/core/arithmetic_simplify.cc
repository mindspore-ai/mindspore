/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <utility>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/model/graph_builder.h"
#include "ops/auto_generate/gen_ops_primitive.h"

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
  // For some patterns, the input parameters may change (e.g.: ReduceSum(ReduceSum(A,B),C)=ReduceSum(A,D)),
  // in this case we need to compute the new axes(D), and update the parameter map.
  virtual std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                                    const std::shared_ptr<ParaMap> &para_to_ref) const {
    (void)origin_root;
    return para_to_ref;
  }
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
  // check whether inputs and attributes meet requirements for certain pattern node if needed;
  virtual bool CheckInputsAndAttrs(const inner::NodePtr &) const { return true; }

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
    return inner::NType::Tensor;
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
    case inner::NType::Tensor: {
      if (tmp_node->NodeType() != inner::NType::Tensor) {
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
  if (CheckInputsAndAttrs(root)) {
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

// Reduce(Reduce(A,B),C) = Reduce(A,D)
class ExtraReduce1PatternTree : public PatternTree {
 public:
  explicit ExtraReduce1PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ExtraReduce1PatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    auto axes1_tensornode = (*para_to_ref)['B']->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(axes1_tensornode);
    auto axes2_tensornode = (*para_to_ref)['C']->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(axes2_tensornode);
    auto axes1 = CheckAndConvertUtils::CheckTensorIntValue("axes", axes1_tensornode->data(), "Reduce");
    auto axes2 = CheckAndConvertUtils::CheckTensorIntValue("axes", axes2_tensornode->data(), "Reduce");
    bool keep_dims = GetValue<bool>(origin_root->attrs().find("keep_dims")->second);
    std::vector<int64_t> axes;
    std::set<int64_t> axis_set;
    if (keep_dims) {
      for (auto &i : axes1) {
        (void)axis_set.insert(i);
      }
      for (auto &i : axes2) {
        (void)axis_set.insert(i);
      }
    } else {
      std::set<int64_t> st(axes1.begin(), axes1.end());
      mindspore::HashMap<int64_t, int64_t> mp;
      int64_t shift = 0;
      auto size = SizeToLong((*para_to_ref)['A']->shape.size());
      for (int64_t n = 0; n < size; n++) {
        if (st.find(n) != st.end()) {
          shift++;
        } else {
          mp[n - shift] = n;
        }
      }
      (void)std::for_each(axes1.begin(), axes1.end(), [&axis_set](auto &i) { (void)axis_set.insert(i); });
      (void)std::for_each(axes2.begin(), axes2.end(), [&axis_set, &mp](auto &i) { (void)axis_set.insert(mp[i]); });
    }
    (void)std::copy(axis_set.begin(), axis_set.end(), std::back_inserter(axes));
    inner::GraphBuilder gb("");
    auto new_axes_tensornode = gb.Tensor(axes);
    (*para_to_ref)['D'] = new_axes_tensornode;
    (void)para_to_ref->erase('B');
    (void)para_to_ref->erase('C');
    return para_to_ref;
  }

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto first_reduce_shape = origin_root->input(0)->shape;
    return (GetValue<bool>((origin_root->inputs()[0])->attrs().find("keep_dims")->second) ==
              GetValue<bool>(origin_root->attrs().find("keep_dims")->second) &&
            !IsDynamicRank(first_reduce_shape));
  }
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    bool keep_dims = GetValue<bool>(origin_root->attrs().find("keep_dims")->second);
    if (GetRootOp() == prim::kPrimReduceSum->name()) {
      auto iter = origin_root->attrs().find("skip_mode");
      if (iter != origin_root->attrs().end()) {
        bool skip_mode = GetValue<bool>(iter->second);
        attrs_map[this->rhs_root()] = {{"keep_dims", MakeValue(keep_dims)}, {"skip_mode", MakeValue(skip_mode)}};
      } else {
        MS_LOG(EXCEPTION) << origin_root->ToString() << "not found skip_mode attrs.";
      }
    } else {
      attrs_map[this->rhs_root()] = {{"keep_dims", MakeValue(keep_dims)}};
    }
    return attrs_map;
  }
};

// "ReduceSum(Neg(A),B)=Neg(ReduceSum(A,B))"
class ExtraReduce2PatternTree : public PatternTree {
 public:
  explicit ExtraReduce2PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ExtraReduce2PatternTree() override = default;

 protected:
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    bool keep_dims = GetValue<bool>(origin_root->attrs().find("keep_dims")->second);
    auto iter = origin_root->attrs().find("skip_mode");
    if (iter != origin_root->attrs().end()) {
      bool skip_mode = GetValue<bool>(iter->second);
      attrs_map[this->rhs_root()->inputs()[0]] = {{"keep_dims", MakeValue(keep_dims)},
                                                  {"skip_mode", MakeValue(skip_mode)}};
    } else {
      MS_LOG(EXCEPTION) << origin_root->ToString() << "not found skip_mode attrs.";
    }
    return attrs_map;
  }
};

// "ReduceSum(A,B)=ReShape(A,C)"
class ReducePatternTree : public PatternTree {
 public:
  explicit ReducePatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ReducePatternTree() override = default;

 protected:
  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    inner::GraphBuilder gb("");
    // Because an empty Tensor cannot be generated, the second input for the reshape function needs to be a Tuple.
    auto shape_node = gb.Tensor(origin_root->shape);
    (*para_to_ref)['C'] = shape_node;
    (void)para_to_ref->erase('B');
    return para_to_ref;
  }
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto reduce_shape = origin_root->input(0)->shape;
    if (IsDynamicShape(reduce_shape)) {
      return false;
    }
    if (reduce_shape.empty()) {
      return true;
    }
    auto reduce_axis = origin_root->input(1)->As<inner::ConstTensorNode>();
    if (reduce_axis == nullptr) {
      return false;
    }
    auto axis = CheckAndConvertUtils::CheckTensorIntValue("axis", reduce_axis->data(), "Reduce");
    for (auto &i : axis) {
      if (i < 0) {
        i += SizeToLong(reduce_shape.size());
      }
      if (reduce_shape[i] != 1) {
        return false;
      }
    }
    return true;
  }
};

class CastPatternTree : public PatternTree {
 public:
  explicit CastPatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~CastPatternTree() = default;

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto dst_type_id = origin_root->type;
    auto src_type_id = origin_root->input(0)->type;
    return dst_type_id == src_type_id;
  }
};

// "LayoutTransform(LayoutTransform(A))=A"
class LayoutTransform1PatternTree : public PatternTree {
 public:
  explicit LayoutTransform1PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~LayoutTransform1PatternTree() override = default;

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    return (GetValue<string>((origin_root->inputs()[0])->attrs().find("src_format")->second) ==
            GetValue<string>(origin_root->attrs().find("dst_format")->second));
  }
};

// "LayoutTransform(LayoutTransform(A))=LayoutTransform(A)"
class LayoutTransform2PatternTree : public PatternTree {
 public:
  explicit LayoutTransform2PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~LayoutTransform2PatternTree() override = default;

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
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

bool IsRedundantTransposePair(const ShapeVector &perm1, const ShapeVector &perm2) {
  auto dim = perm2.size();
  for (size_t i = 0; i < dim; i++) {
    auto index = perm2[i] < 0 ? perm2[i] + static_cast<ShapeValueDType>(dim) : perm2[i];
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(index) < dim, "perm is out of bound");
    auto axis = perm1[index] < 0 ? perm1[index] + static_cast<ShapeValueDType>(dim) : perm1[index];
    if (static_cast<size_t>(axis) != i) {
      return false;
    }
  }
  return true;
}
// Transpose(Transpose(A,B),C)=A
class Transpose1PatternTree : public PatternTree {
 public:
  explicit Transpose1PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~Transpose1PatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    (void)para_to_ref->erase('B');
    (void)para_to_ref->erase('C');
    return para_to_ref;
  }

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto transpose1_node = origin_root->input(0);
    MS_EXCEPTION_IF_NULL(transpose1_node);
    if (transpose1_node->format != origin_root->format) {
      MS_LOG(DEBUG) << "The input format of the first transpose is different from the output format of the second "
                       "transpose, can't remove this transpose pair.";
      return false;
    }
    auto input_shape = transpose1_node->input(0)->shape;
    auto perm2_node = origin_root->input(1);
    MS_EXCEPTION_IF_NULL(perm2_node);
    auto perm1_node = transpose1_node->input(1);
    MS_EXCEPTION_IF_NULL(perm1_node);
    auto perm2_tensornode = perm2_node->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm2_tensornode);
    auto perm1_tensornode = perm1_node->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm1_tensornode);
    auto perm2 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm2_tensornode->data(), "Transpose");
    auto perm1 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm1_tensornode->data(), "Transpose");
    if (perm1.size() != input_shape.size() || perm2.size() != input_shape.size()) {
      MS_LOG(DEBUG) << "The length of input shape and perm is not same";
      return false;
    }
    return IsRedundantTransposePair(perm1, perm2);
  }

  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{"format", MakeValue(origin_root->format)}};
    return attrs_map;
  }
};

// Transpose(A,B)=Reshape(A,C)
class Transpose2PatternTree : public PatternTree {
 public:
  explicit Transpose2PatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~Transpose2PatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    inner::GraphBuilder gb("");
    auto out_shape = origin_root->shape;
    auto out_shape_tensornode = gb.Tensor(out_shape);
    (*para_to_ref)['C'] = out_shape_tensornode;
    (void)para_to_ref->erase('B');
    return para_to_ref;
  }

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto input_shape = origin_root->input(0)->shape;
    if (IsDynamicRank(input_shape)) {
      MS_LOG(DEBUG) << "Skip dynamic rank case";
      return false;
    }
    auto perm_tensornode = origin_root->input(1)->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm_tensornode);
    auto perm = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm_tensornode->data(), "Transpose");
    if (perm.size() != input_shape.size()) {
      MS_LOG(DEBUG) << "The length of input shape " << input_shape << " and perm " << perm << " is not same";
      return false;
    }
    auto rank = SizeToLong(input_shape.size());
    // If the axes which have dimension size greater than 1 keep ascending order in permutation, then this transpose can
    // be replaced by reshape
    ShapeValueDType prev_non_one_axis = -1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (perm[i] < -rank || perm[i] >= rank) {
        MS_LOG(DEBUG) << "perm[" << i << "] is " << perm[i] << ", which is out of range[-" << rank << ", " << rank
                      << ")";
        return false;
      }
      perm[i] = perm[i] < 0 ? (perm[i] + rank) : perm[i];
      if (input_shape[perm[i]] != 1) {
        if (perm[i] < prev_non_one_axis) {
          MS_LOG(DEBUG) << "perm[" << i << "] is axis " << perm[i]
                        << ", which is greater than the previous non-one axis " << prev_non_one_axis
                        << ", replace failed";
          return false;
        }
        prev_non_one_axis = perm[i];
      }
    }
    return true;
  }

  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{"format", MakeValue(origin_root->format)}};
    return attrs_map;
  }
};

// Reshape(Reshape(A,B),C)=Reshape(A,C)
class ReshapePatternTree : public PatternTree {
 public:
  explicit ReshapePatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~ReshapePatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    inner::GraphBuilder gb("");
    auto out_shape = origin_root->shape;
    auto out_shape_tensornode = gb.Tensor(out_shape);
    (*para_to_ref)['C'] = out_shape_tensornode;
    (void)para_to_ref->erase('B');
    return para_to_ref;
  }

 protected:
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{"format", MakeValue(origin_root->format)}};
    return attrs_map;
  }
};

// Transpose(Transpose(Reshape(A,B),C),D)=Reshape(A,E), RTT is the abbreviation for Reshape + Transpose + Transpose
class RTTPatternTree : public PatternTree {
 public:
  explicit RTTPatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~RTTPatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    inner::GraphBuilder gb("");
    auto out_shape = origin_root->shape;
    auto out_shape_tensornode = gb.Tensor(out_shape);
    (*para_to_ref)['E'] = out_shape_tensornode;
    (void)para_to_ref->erase('B');
    (void)para_to_ref->erase('C');
    (void)para_to_ref->erase('D');
    return para_to_ref;
  }

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto perm2_node = origin_root->input(1);
    MS_EXCEPTION_IF_NULL(perm2_node);
    auto transpose1_node = origin_root->input(0);
    MS_EXCEPTION_IF_NULL(transpose1_node);
    auto perm1_node = transpose1_node->input(1);
    MS_EXCEPTION_IF_NULL(perm1_node);
    auto perm2_tensornode = perm2_node->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm2_tensornode);
    auto perm1_tensornode = perm1_node->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm1_tensornode);
    auto perm2 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm2_tensornode->data(), "Transpose");
    auto perm1 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm1_tensornode->data(), "Transpose");
    return IsRedundantTransposePair(perm1, perm2);
  }
  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{"format", MakeValue(origin_root->format)}};
    return attrs_map;
  }
};

// StridedSlice(A,B,C,D)=Reshape(A,E)
class StridedSlicePatternTree : public PatternTree {
 public:
  explicit StridedSlicePatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~StridedSlicePatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    MS_EXCEPTION_IF_NULL(para_to_ref);
    inner::GraphBuilder gb("");
    auto out_shape = origin_root->shape;
    auto out_shape_tensornode = gb.Tensor(out_shape);
    (*para_to_ref)['E'] = out_shape_tensornode;
    (void)para_to_ref->erase('B');
    (void)para_to_ref->erase('C');
    (void)para_to_ref->erase('D');
    return para_to_ref;
  }

 protected:
  const ShapeVector GetInputVec(const inner::NodePtr &origin_root, size_t input_idx, const std::string &node_name,
                                const std::string &input_name) const {
    auto input_node = origin_root->input(input_idx);
    MS_EXCEPTION_IF_NULL(input_node);
    MS_EXCEPTION_IF_CHECK_FAIL(input_node->NodeType() == inner::NType::Tensor, "input must be a Tensor");
    auto input_tensornode = input_node->As<inner::ConstTensorNode>();
    auto input_vec = CheckAndConvertUtils::CheckTensorIntValue(input_name, input_tensornode->data(), node_name);
    return input_vec;
  }

  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto input_node = origin_root->input(0);
    MS_EXCEPTION_IF_NULL(input_node);
    auto input_shape = input_node->shape;
    const ShapeVector &begin_vec = GetInputVec(origin_root, 1, "StridedSlice", "begin");
    if (std::any_of(begin_vec.begin(), begin_vec.end(), [](ShapeValueDType i) { return i != 0; })) {
      return false;
    }
    const ShapeVector &end_vec = GetInputVec(origin_root, 2, "StridedSlice", "end");
    for (size_t i = 0; i < end_vec.size(); i++) {
      if (end_vec[i] != input_shape[i]) {
        return false;
      }
    }
    const ShapeVector &strides_vec = GetInputVec(origin_root, 3, "StridedSlice", "strideds");
    if (std::any_of(strides_vec.begin(), strides_vec.end(), [](ShapeValueDType i) { return i != 1; })) {
      return false;
    }
    auto begin_mask = GetValue<int64_t>(origin_root->attrs().find("begin_mask")->second);
    auto end_mask = GetValue<int64_t>(origin_root->attrs().find("end_mask")->second);
    auto ellipsis_mask = GetValue<int64_t>(origin_root->attrs().find("ellipsis_mask")->second);
    if (begin_mask != 0 || end_mask != 0 || ellipsis_mask != 0) {
      return false;
    }
    auto shrink_axis_mask = LongToSize(GetValue<int64_t>(origin_root->attrs().find("shrink_axis_mask")->second));
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (((shrink_axis_mask >> i) & 1) != 0 && input_shape[i] != 1) {
        return false;
      }
    }
    return true;
  }
};

// Transpose(Transpose(A,B),C)=Transpose(A,D)
class TransposeCombinePatternTree : public PatternTree {
 public:
  explicit TransposeCombinePatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~TransposeCombinePatternTree() override = default;

  std::shared_ptr<ParaMap> UpdateParameters(const inner::NodePtr &origin_root,
                                            const std::shared_ptr<ParaMap> &para_to_ref) const override {
    /* %0 = Transpose(p, (1, 0, 2))
     * %1 = Transpose(%0, (0, 2, 1))
     * --->
     * %0 = Transpose(p, (1, 2, 0))
     */
    MS_EXCEPTION_IF_NULL(para_to_ref);
    auto perm1_node = (*para_to_ref)['B']->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm1_node);
    auto perm2_node = (*para_to_ref)['C']->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm1_node);
    auto perm1 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm1_node->data(), "Transpose");
    auto perm2 = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm2_node->data(), "Transpose");
    auto rank = SizeToLong(origin_root->shape.size());
    (void)std::for_each(perm1.begin(), perm1.end(), [rank](auto &axis) { axis = axis < 0 ? axis + rank : axis; });
    (void)std::for_each(perm2.begin(), perm2.end(), [rank](auto &axis) { axis = axis < 0 ? axis + rank : axis; });
    ShapeVector new_perm(perm2.size());
    for (size_t i = 0; i < perm2.size(); ++i) {
      new_perm[i] = perm1[LongToSize(perm2[i])];
    }
    inner::GraphBuilder gb("");
    (*para_to_ref)['D'] = gb.Tensor(new_perm);
    (void)para_to_ref->erase('B');
    (void)para_to_ref->erase('C');
    return para_to_ref;
  }

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto perm2 = origin_root->input(1);
    MS_EXCEPTION_IF_NULL(perm2);
    auto trans1 = origin_root->input(0);
    MS_EXCEPTION_IF_NULL(trans1);
    auto perm1 = trans1->input(1);
    MS_EXCEPTION_IF_NULL(perm1);
    auto perm2_tensor = perm2->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm2_tensor);
    auto perm1_tensor = perm1->As<inner::ConstTensorNode>();
    MS_EXCEPTION_IF_NULL(perm1_tensor);
    auto perm1_value = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm2_tensor->data(), "Transpose");
    auto perm2_value = CheckAndConvertUtils::CheckTensorIntValue("permutation", perm1_tensor->data(), "Transpose");
    auto shape = origin_root->shape;
    if (perm2_value.size() != shape.size() || perm1_value.size() != shape.size()) {
      MS_LOG(DEBUG) << "perm1, perm2 and shape have different size. perm1: " << perm2_value << " perm2: " << perm1_value
                    << " node shape: " << shape;
      return false;
    }
    return true;
  }

  mindspore::HashMap<PatternNodePtr, inner::DAttrs> SetAttributes(const inner::NodePtr &origin_root) override {
    auto attrs_map = PatternTree::SetAttributes(origin_root);
    attrs_map[this->rhs_root()] = {{kAttrDstFormat, MakeValue(origin_root->format)}};
    return attrs_map;
  }
};

class FloatCheckPatternTree : public PatternTree {
 public:
  explicit FloatCheckPatternTree(const std::string &pattern_str) : PatternTree(pattern_str) {}
  ~FloatCheckPatternTree() override = default;

 protected:
  bool CheckInputsAndAttrs(const inner::NodePtr &origin_root) const override {
    auto type_id = origin_root->type;
    return (type_id == kNumberTypeFloat || type_id == kNumberTypeFloat16 || type_id == kNumberTypeFloat32 ||
            type_id == kNumberTypeFloat64);
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
  {50, "RealDiv(A,const1)=Mul(A,Reciprocal(const1))", EXPR_PATTERN(FloatCheckPatternTree)},
  {51, "RealDiv(RealDiv(A,B),RealDiv(C,D))=RealDiv(Mul(A,D),Mul(B,C))", EXPR_PATTERN(PatternTree)},
  {52, "RealDiv(Neg(A),const1)=RealDiv(A,Neg(const1))", EXPR_PATTERN(PatternTree)},
  {53, "RealDiv(RealDiv(A,B),C)=RealDiv(A,Mul(B,C))", EXPR_PATTERN(PatternTree)},
  {54, "RealDiv(A,RealDiv(B,C))=RealDiv(Mul(A,C),B)", EXPR_PATTERN(PatternTree)},
  // reduce1, B, C, D are all axes input
  {55, "ReduceSum(ReduceSum(A,B),C)=ReduceSum(A,D)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  {56, "ReduceMin(ReduceMin(A,B),C)=ReduceMin(A,D)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  {57, "ReduceMax(ReduceMax(A,B),C)=ReduceMax(A,D)", EXPR_PATTERN(ExtraReduce1PatternTree)},
  // reduce2, B is axes input
  {58, "ReduceSum(Neg(A),B)=Neg(ReduceSum(A,B))", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {59, "ReduceSum(RealDiv(A,const1),B)=RealDiv(ReduceSum(A,B),const1)", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {60, "ReduceSum(Mul(A,const1),B)=Mul(ReduceSum(A,B),const1)", EXPR_PATTERN(ExtraReduce2PatternTree)},
  {61, "CReal(Complex(A,B))=A", EXPR_PATTERN(PatternTree)},
  {62, "CImag(Complex(A,B))=B", EXPR_PATTERN(PatternTree)},
  // lite only
  {63, "LayoutTransform(LayoutTransform(A))=A", EXPR_PATTERN(LayoutTransform1PatternTree)},
  {64, "LayoutTransform(LayoutTransform(A))=LayoutTransform(A)", EXPR_PATTERN(LayoutTransform2PatternTree)},
  // patterns that can be transformed to reshape
  {65, "Transpose(Transpose(A,B),C)=A", EXPR_PATTERN(Transpose1PatternTree)},
  {66, "Transpose(A,B)=Reshape(A,C)", EXPR_PATTERN(Transpose2PatternTree)},
  {67, "Reshape(Reshape(A,B),C)=Reshape(A,C)", EXPR_PATTERN(ReshapePatternTree)},
  {68, "Transpose(Transpose(Reshape(A,B),C),D)=Reshape(A,E)", EXPR_PATTERN(RTTPatternTree)},
  {69, "StridedSlice(A,B,C,D)=Reshape(A,E)", EXPR_PATTERN(StridedSlicePatternTree)},
  // cmp + logical
  {70, "LogicalNot(Greater(A,B))=LessEqual(A,B)", EXPR_PATTERN(PatternTree)},
  {71, "LogicalNot(LessEqual(A,B))=Greater(A,B)", EXPR_PATTERN(PatternTree)},
  {72, "LogicalNot(GreaterEqual(A,B))=Less(A,B)", EXPR_PATTERN(PatternTree)},
  {73, "LogicalNot(Less(A,B))=GreaterEqual(A,B)", EXPR_PATTERN(PatternTree)},
  {74, "LogicalNot(NotEqual(A,B))=Equal(A,B)", EXPR_PATTERN(PatternTree)},
  {75, "LogicalNot(Equal(A,B))=NotEqual(A,B)", EXPR_PATTERN(PatternTree)},
  // reduce -> reshape
  {76, "ReduceSum(A,B)=Reshape(A,C)", EXPR_PATTERN(ReducePatternTree)},
  {77, "ReduceMin(A,B)=Reshape(A,C)", EXPR_PATTERN(ReducePatternTree)},
  {78, "ReduceMax(A,B)=Reshape(A,C)", EXPR_PATTERN(ReducePatternTree)},
  {79, "Cast(A,B)=A", EXPR_PATTERN(CastPatternTree)},
  // transpose
  {80, "Transpose(Transpose(A,B),C)=Transpose(A,D)", EXPR_PATTERN(TransposeCombinePatternTree)},
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
          para_to_ref = cur_pattern->UpdateParameters(*iter, para_to_ref);
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

bool ResetOutputs(const inner::LiteGraphPtr &litegraph) {
  /** If after arithmetic transformation and constant folding, an output of subgraph is just a Tensor or Parameter,
   * insert Reshape/BroadcastTo and reset the output to this op.
   */
  auto &outputs = litegraph->GetOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(outputs[i]);
    auto out_shape = outputs[i]->shape;
    if (outputs[i]->NodeType() == inner::NType::Tensor) {
      if (IsDynamic(out_shape)) {
        return false;
      }
      inner::GraphBuilder gb;
      auto output_shape = outputs[i]->As<inner::ConstTensorNode>()->data()->shape();
      auto op_ptr = gb.BroadcastTo(outputs[i], output_shape);
      litegraph->SetOutput(i, op_ptr);
    } else if (outputs[i]->NodeType() == inner::NType::Parameter) {
      if (IsDynamicRank(out_shape) ||
          std::count_if(out_shape.begin(), out_shape.end(), [](int64_t sh) { return sh < 0; }) > 1) {
        return false;
      }
      inner::GraphBuilder gb;
      auto op_ptr = gb.Reshape(outputs[i], out_shape);
      litegraph->SetOutput(i, op_ptr);
    }
  }
  return true;
}

bool ArithmeticSimplify::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool do_simplify = false;
  expressions_map_ = GetExpressions();
  for (auto node : func_graph->GetOrderedCnodes()) {
    if (AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      if (auto type = sub_graph->get_attr("composite_type")) {
        if (GetValue<std::string>(type) == "inplace_assign_builder") {
          continue;
        }
      }
      auto cnode = node->cast<CNodePtr>();
      AnfNodePtrList inputs = cnode->inputs();
      inner::LiteGraphPtr lg = GkUtils::AnfGraph2LiteGraph(sub_graph);
      bool find_pattern = true;
      bool change_anf_graph = false;
      try {
        MS_LOG_TRY_CATCH_SCOPE;
        while (find_pattern) {
          find_pattern = false;
          find_pattern = DoConstantFold(lg) || find_pattern;
          find_pattern = DoArithmeticTrans(lg) || find_pattern;
          change_anf_graph = change_anf_graph || find_pattern;
        }
      } catch (const std::exception &e) {
        MS_LOG(INFO) << "During arithmetic simplify for node [" << node->fullname_with_scope()
                     << "], an error occurs: " << e.what();
        continue;
      }
      AnfNodePtrList input_nodes{inputs.begin() + 1, inputs.end()};
      if (!change_anf_graph) {
        continue;
      }
      if (!ResetOutputs(lg)) {
        continue;
      }
      auto new_funcgraph = GkUtils::LiteGraph2AnfGraph(lg, Callback::Instance());
      if (new_funcgraph == nullptr) {
        continue;
      }
      (void)ConvertTensorToParameter(new_funcgraph, &input_nodes);
      new_funcgraph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      auto new_node = CreateNewFuseCNode(func_graph, new_funcgraph, input_nodes);
      (void)mng->Replace(node, new_node);
      mng->AddFuncGraph(new_funcgraph);
      do_simplify = true;
    }
  }
  return do_simplify;
}
}  // namespace mindspore::graphkernel
