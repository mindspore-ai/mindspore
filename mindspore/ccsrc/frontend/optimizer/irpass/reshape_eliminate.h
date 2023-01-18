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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_

#include <vector>

#include "ir/func_graph.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace opt {
namespace irpass {
using abstract::Shape;
using abstract::ShapePtr;

// {reshape_op, X, Shape}
class ReshapeSameShapeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimReshape, {IsNode, IsVNode})(node);

    // check pattern match
    if (shape_ == nullptr) {
      return nullptr;
    }

    auto src_shape_abs = x_->abstract();
    if (src_shape_abs == nullptr) {
      return nullptr;
    }

    auto src_shape = src_shape_abs->GetShapeTrack();
    auto tgt_shape_abs = node->abstract();
    if (tgt_shape_abs == nullptr) {
      return nullptr;
    }
    auto tgt_shape = tgt_shape_abs->GetShapeTrack();
    if (src_shape != nullptr && tgt_shape != nullptr && src_shape->isa<Shape>() && tgt_shape->isa<Shape>()) {
      auto elements = tgt_shape->cast<ShapePtr>();
      auto shape = src_shape->cast<ShapePtr>();
      if (shape->shape() == elements->shape()) {
        return x_;
      }
    }

    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (x_ == nullptr) {
      x_ = node;
    } else {
      shape_ = node;
    }
  }

  void Reset() {
    x_ = nullptr;
    shape_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, shape_{nullptr};
};

// {PrimReshape, {PrimReshape, X, Y}, Shape}
class TwoReshapeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimReshape, {IsCNode, IsNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && x_ != nullptr && shape_ != nullptr) {
      auto new_node = fg->NewCNode({NewValueNode(prim_), x_, shape_});
      new_node->set_abstract(node->abstract());
      if (node->scope() != kDefaultScope) {
        new_node->set_scope(node->scope());
      }
      return new_node;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (prim_ == nullptr && x_ == nullptr) {
      if (IsPrimitiveCNode(node, prim::kPrimReshape)) {
        auto &inputs = node->cast<CNodePtr>()->inputs();
        // {PrimReshape, X, Y}
        constexpr auto reshape_input_size = 3;
        if (inputs.size() != reshape_input_size) {
          return;
        }
        prim_ = GetValueNode<PrimitivePtr>(inputs[0]);
        x_ = inputs[1];
      }
    } else {
      shape_ = node;
    }
  }

  void Reset() {
    prim_ = nullptr;
    x_ = nullptr;
    shape_ = nullptr;
  }

 private:
  PrimitivePtr prim_{nullptr};
  AnfNodePtr x_{nullptr}, shape_{nullptr};
};

class ReshapeEliminater : public OptimizerCaller {
 public:
  ReshapeEliminater() : reshape_same_shape_eliminater_(), two_reshape_eliminater_() {}
  ~ReshapeEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    auto new_node = reshape_same_shape_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    new_node = two_reshape_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    return nullptr;
  }

 private:
  ReshapeSameShapeEliminater reshape_same_shape_eliminater_;
  TwoReshapeEliminater two_reshape_eliminater_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_
