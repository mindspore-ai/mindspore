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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/dshape.h"

namespace mindspore {
namespace opt {
namespace irpass {
using abstract::Shape;
using abstract::ShapePtr;

// {ReduceLike, X, axis}
class ReduceOneEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    PrimitivePtr prim;
    if (IsPrimitiveCNode(node, prim::kPrimReduceMean) || IsPrimitiveCNode(node, prim::kPrimReduceAll) ||
        IsPrimitiveCNode(node, prim::kPrimReduceSum) || IsPrimitiveCNode(node, prim::kPrimReduceMax) ||
        IsPrimitiveCNode(node, prim::kPrimReduceMin)) {
      prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
      AnfVisitor::Match(prim, {IsNode, IsVNode})(node);
      if (!is_axis_one_) {
        return nullptr;
      }

      // consider keep_dims
      auto keep_dims = prim->GetAttr("keep_dims");
      auto is_keep_dims = GetValue<bool>(keep_dims);
      // {_Reduce, X, axis} -> X
      if (is_keep_dims) {
        return x_;
      }

      // {_Reduce, Tensor}
      if (is_tensor_) {
        return nullptr;
      }

      // {_Reduce, X, axis} -> {Reshape, X, new_shape}
      std::vector<ValuePtr> elements;
      for (size_t i = 0; i < x_shape_.size(); i++) {
        auto iter = find(axis_.begin(), axis_.end(), i);
        if (iter == axis_.end()) {
          ValuePtr s = MakeValue(x_shape_[i]);
          elements.push_back(s);
        }
      }
      auto new_shape = std::make_shared<ValueTuple>(elements);
      auto reshape_op = prim::GetPythonOps("reshape", "mindspore.ops.functional")->cast<PrimitivePtr>();
      return node->func_graph()->NewCNode({NewValueNode(reshape_op), x_, NewValueNode(new_shape)});
    }

    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (!IsVNode(node) && x_ == nullptr) {
      if (IsValueNode<tensor::Tensor>(node)) {
        is_tensor_ = true;
      }
      // get X's shape
      auto x_shape_abs = node->abstract();
      if (x_shape_abs != nullptr) {
        auto x_track = x_shape_abs->GetShapeTrack()->cast<ShapePtr>();
        if (x_track == nullptr) {
          return;
        }
        auto x_shape = x_track->shape();
        (void)std::copy(x_shape.begin(), x_shape.end(), std::back_inserter(x_shape_));
        x_ = node;
      }
      return;
    }

    // check axis
    AnfVisitor::Visit(node);
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (x_shape_.empty()) {
      return;
    }

    // axis : int
    if (IsValueNode<Int32Imm>(vnode)) {
      auto idx = GetValue<int>(vnode->value());
      // axis could be negative
      if (idx < 0) {
        idx += SizeToInt(x_shape_.size());
      }
      if (SizeToInt(x_shape_.size()) > idx && x_shape_[IntToSize(idx)] == 1) {
        is_axis_one_ = true;
        axis_.push_back(idx);
      }
      return;
    }

    // axis : tuple(int), default ()
    if (IsValueNode<ValueTuple>(vnode)) {
      auto axis = GetValue<std::vector<int>>(vnode->value());
      if (axis.empty()) {
        return;
      }

      auto cmp = std::all_of(axis.cbegin(), axis.cend(), [this](int idx) {
        // axis could be negative
        if (idx < 0) {
          idx += SizeToInt(x_shape_.size());
        }
        return SizeToInt(this->x_shape_.size()) > idx && this->x_shape_[IntToSize(idx)] == 1;
      });
      if (cmp) {
        is_axis_one_ = true;
        (void)std::copy(axis.begin(), axis.end(), std::back_inserter(axis_));
      }
    }
  }

  void Reset() {
    axis_.clear();
    x_shape_.clear();
    x_ = nullptr;
    is_axis_one_ = false;
    is_tensor_ = false;
  }

 private:
  bool is_axis_one_{false}, is_tensor_{false};
  std::vector<int> axis_{}, x_shape_{};
  AnfNodePtr x_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_
