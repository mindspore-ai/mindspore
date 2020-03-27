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

class ArithmeticSimplify {
 public:
  ArithmeticSimplify()
      : multiply_by_zero_or_one_(),
        add_by_zero_(),
        tensor_add_by_zero_(),
        identity_(prim::kPrimIdentity),
        opt_update_zero_tensor_() {
    eliminaters_.emplace_back(multiply_by_zero_or_one_);
    eliminaters_.emplace_back(add_by_zero_);
    eliminaters_.emplace_back(tensor_add_by_zero_);
    eliminaters_.emplace_back(identity_);
    eliminaters_.emplace_back(opt_update_zero_tensor_);
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
  std::vector<TransformFuncType> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
