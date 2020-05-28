/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_

#include <vector>
#include <memory>
#include <algorithm>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "optimizer/irpass/prim_eliminate.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimScalarMul, 0, X}, {prim::kPrimScalarMul, X, 0}
// {prim::kPrimScalarMul, 1, X}, {prim::kPrimScalarMul, X, 1}
class MultiplyByZeroOrOne : public AnfVisitor {
 public:
  MultiplyByZeroOrOne() : zero_(MakeValue(0)), one_(MakeValue(1)) {}
  ~MultiplyByZeroOrOne() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimScalarMul)(node);

    if (is_zero_) {
      return NewValueNode(zero_);
    }
    if (is_one_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_one_ || node->isa<CNode>()) {
      x_ = node;
      return;
    }

    AnfVisitor::Visit(node);
    if (!is_one_) {
      x_ = node;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto value = vnode->value();
    if (*value == *zero_) {
      is_zero_ = true;
    } else if (*value == *one_) {
      is_one_ = true;
    }
  }

  void Reset() {
    x_ = nullptr;
    is_one_ = false;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false}, is_one_{false};
  ValuePtr zero_, one_;
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimScalarAdd, X, 0}
// {prim::kPrimScalarAdd, 0, X}
class AddByZero : public AnfVisitor {
 public:
  AddByZero() : zero_(MakeValue(0)) {}
  ~AddByZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimScalarAdd)(node);

    if (is_zero_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (node->isa<ValueNode>() && *GetValueNode(node) == *zero_) {
      is_zero_ = true;
      return;
    }

    x_ = node;
  }

  void Reset() {
    x_ = nullptr;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false};
  ValuePtr zero_;
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimTensorAdd, {PrimZerosLikeTensor, Y}, X},
// {prim::kPrimTensorAdd, X, {PrimZerosLikeTensor, Y}}
class TensorAddByZero : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTensorAdd)(node);

    if (is_zero_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsPrimitive(node, prim::kPrimZerosLikeTensor)) {
      is_zero_ = true;
      return;
    }

    x_ = node;
  }

  void Reset() {
    x_ = nullptr;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false};
  AnfNodePtr x_{nullptr};
};

// {PrimMomentum, {PrimZerosLikeTensor, X}, Y, Z, Xs}  -> {prim::kPrimMakeTuple, Z, Y}
class OptUpdateZeroTensor : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimMomentum) || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {PrimMomentum, {...}, Y, Z, Xs}
    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs.size() < 4 || !IsPrimitiveCNode(inputs[1], prim::kPrimZerosLikeTensor)) {
      return nullptr;
    }
    auto y = inputs[2];
    auto z = inputs[3];

    // {PrimZerosLikeTensor, X}
    if (inputs[1]->cast<CNodePtr>()->size() != 2) {
      return nullptr;
    }

    // {prim::kPrimMakeTuple, Z, Y}
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimMakeTuple), z, y});
  }
};

// {prim::kPrimMul, Tensor1, {orim::kPrimMul, Tensor2, {...}}} ->
// {prim::kPrimMul, {...}, {prim::kPrimMul, Tensor1, Tensor2}}
class ConstantDuplicateMul : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    // {prim::kPrimMul, Tensor1, {...}}
    AnfVisitor::Match(prim::kPrimMul, {IsNode, IsNode})(node);
    if (vnode_ == nullptr || cnode_ == nullptr) {
      return nullptr;
    }
    auto tensor1 = vnode_;
    auto mul = cnode_;

    Reset();
    // {prim::kPrimMul, Tensor2, {...}}
    AnfVisitor::Match(prim::kPrimMul, {IsNode, IsNode})(mul);
    if (vnode_ == nullptr || cnode_ == nullptr) {
      return nullptr;
    }
    auto tensor2 = vnode_;
    auto cnode = cnode_;

    auto PrimMul = GetValueNode<PrimitivePtr>(mul->input(0));
    auto fg = node->func_graph();
    auto ttmul = NewCNode({NewValueNode(PrimMul), tensor1, tensor2}, fg);
    return NewCNode({NewValueNode(PrimMul), cnode, ttmul}, fg);
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsValueNode<tensor::Tensor>(node)) {
      vnode_ = node;
    }

    if (IsCNode(node)) {
      cnode_ = node->cast<CNodePtr>();
    }
  }

  void Reset() {
    vnode_ = nullptr;
    cnode_ = nullptr;
  }

 private:
  AnfNodePtr vnode_;
  CNodePtr cnode_;
};

// grad = AllReduce(grad) / worker_number
// grad = grad + weight * decy
// ->
// grad = grad + weight * decy
// grad = AllReduce(grad) / worker_number

// {prim::kPrimAddN, {prim::kPrimMakeTuple, {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}, Z}} ->
// {prim::kPrimMul, {prim::kPrimAllReduce, {prim::kPrimAddN,{prim::kPrimMakeTuple, Z, X}}}, Y}
class AdjustAllReduceMulAdd : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    // {prim::kPrimAddN, Zs}
    if (!IsPrimitiveCNode(node, prim::kPrimAddN)) {
      return nullptr;
    }
    auto addn = node->cast<CNodePtr>();
    if (addn->size() != 2) {
      return nullptr;
    }
    AnfVisitor::Match(prim::kPrimMakeTuple, {IsNode, IsNode})(addn->input(1));
    if (x_ == nullptr || y_ == nullptr || z_ == nullptr || all_reduce_fg_ == nullptr) {
      return nullptr;
    }
    auto addn_maketuple = addn->input(1);

    auto fg = all_reduce_fg_;
    // addn inputs cross the graph, make the inputs same as allreduce node.
    if (z_->isa<CNode>() && fg != z_->func_graph()) {
      auto cnode_z = z_->cast<CNodePtr>();
      z_ = NewCNode(cnode_z->inputs(), fg);
    }

    auto addn_op_node = addn->input(0);
    auto make_tuple_op_node = addn->input(1)->cast<CNodePtr>()->input(0);

    AnfNodePtr tuple = NewCNode({make_tuple_op_node, z_, x_}, fg);
    AnfNodePtr add = NewCNode({addn_op_node, tuple}, fg);
    AnfNodePtr all_reduce = NewCNode({all_reduce_, add}, fg);
    AnfNodePtr mul = NewCNode({mul_, all_reduce, y_}, fg);
    ProcessDependEdge(fg, addn_maketuple, all_reduce);
    return mul;
  }
  void ProcessDependEdge(const FuncGraphPtr &fg, const AnfNodePtr &addn_maketuple, const AnfNodePtr &new_node) {
    // If has dynamic loss scale.
    auto &users_map = fg->manager()->node_users();
    auto it = users_map.find(mul_cnode_);
    if (it != users_map.end()) {
      auto users = it->second;
      for (auto &user_pair : users) {
        auto node = user_pair.first;
        if (node != addn_maketuple) {
          if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
            fg->manager()->SetEdge(node, user_pair.second, new_node);
          }
        }
      }
    }
  }
  void Visit(const AnfNodePtr &node) override {
    if (level_ == 0) {
      level_ = 1;
      is_reduce_match_ = false;
      // {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}
      AnfVisitor::Match(prim::kPrimMul)(node);
      level_ = 0;
      if (is_reduce_match_) {
        mul_ = node->cast<CNodePtr>()->input(0);
        mul_cnode_ = node->cast<CNodePtr>();
        y_ = tmp_;
      } else {
        z_ = node;
      }
    }

    if (level_ == 1) {
      // {prim::kPrimAllReduce, X}
      if (IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
        auto cnode = node->cast<CNodePtr>();
        if (cnode->size() > 1) {
          all_reduce_ = cnode->input(0);
          x_ = cnode->input(1);
          is_reduce_match_ = true;
          all_reduce_fg_ = cnode->func_graph();
        }
      } else {
        tmp_ = node;
      }
    }
  }

  void Reset() {
    level_ = 0;
    is_reduce_match_ = false;
    x_ = nullptr;
    y_ = nullptr;
    z_ = nullptr;
    tmp_ = nullptr;
    all_reduce_fg_ = nullptr;
  }

 private:
  int level_{0};
  bool is_reduce_match_{false};
  AnfNodePtr x_{nullptr}, y_{nullptr}, z_{nullptr}, tmp_{nullptr};
  AnfNodePtr all_reduce_{nullptr}, mul_{nullptr}, mul_cnode_{nullptr};
  FuncGraphPtr all_reduce_fg_{nullptr};
};

class ArithmeticSimplify {
 public:
  ArithmeticSimplify()
      : multiply_by_zero_or_one_(),
        add_by_zero_(),
        tensor_add_by_zero_(),
        identity_(prim::kPrimIdentity),
        opt_update_zero_tensor_(),
        constant_duplicate_mul_() {
    eliminaters_.emplace_back(multiply_by_zero_or_one_);
    eliminaters_.emplace_back(add_by_zero_);
    eliminaters_.emplace_back(tensor_add_by_zero_);
    eliminaters_.emplace_back(identity_);
    eliminaters_.emplace_back(opt_update_zero_tensor_);
    eliminaters_.emplace_back(constant_duplicate_mul_);
  }
  ~ArithmeticSimplify() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = eliminater(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  MultiplyByZeroOrOne multiply_by_zero_or_one_;
  AddByZero add_by_zero_;
  TensorAddByZero tensor_add_by_zero_;
  PrimEliminater identity_;
  OptUpdateZeroTensor opt_update_zero_tensor_;
  ConstantDuplicateMul constant_duplicate_mul_;
  std::vector<TransformFuncType> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
