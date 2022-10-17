/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "abstract/dshape.h"
#include "utils/anf_utils.h"

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
    if (node->func_graph() == nullptr) {
      return nullptr;
    }

    PrimitivePtr prim;
    if (IsPrimitiveCNode(node, prim::kPrimReduceMean) || IsPrimitiveCNode(node, prim::kPrimReduceAll) ||
        IsPrimitiveCNode(node, prim::kPrimReduceSum) || IsPrimitiveCNode(node, prim::kPrimReduceMax) ||
        IsPrimitiveCNode(node, prim::kPrimReduceMin)) {
      prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
      AnfVisitor::Match(prim, {IsNode, IsVNode})(node);
      if (!is_axis_one_) {
        return nullptr;
      }

      // if node has keep_alive attr, it would not be eliminated.
      if (IsPrimitiveCNode(node, prim::kPrimReduceMean)) {
        if (prim->HasAttr("keep_alive") && GetValue<bool>(prim->GetAttr("keep_alive"))) {
          MS_LOG(INFO) << "keep node " << node->fullname_with_scope() << " alive";
          return nullptr;
        }
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
      size_t x_shape_size = x_shape_.size();
      std::vector<int64_t> positive_axis;
      std::transform(axis_.begin(), axis_.end(), std::back_inserter(positive_axis),
                     [x_shape_size](int64_t idx) { return idx < 0 ? idx + SizeToLong(x_shape_size) : idx; });

      std::vector<ValuePtr> elements;
      for (size_t i = 0; i < x_shape_size; i++) {
        auto iter = find(positive_axis.begin(), positive_axis.end(), i);
        if (iter == positive_axis.end()) {
          ValuePtr s = MakeValue(x_shape_[i]);
          elements.push_back(s);
        }
      }

      auto new_shape = std::make_shared<ValueTuple>(elements);
      py::gil_scoped_acquire gil;
      auto reshape_op = prim::GetPythonOps("reshape_", "mindspore.ops.functional")->cast<PrimitivePtr>();
      auto node_abstract = node->abstract();
      // handle auto_parallel get nullptr abstract
      if (node_abstract != nullptr) {
        auto new_base_shape = std::make_shared<abstract::Shape>(GetValue<std::vector<int64_t>>(new_shape));
        node_abstract->set_shape(new_base_shape);
        auto new_node = node->func_graph()->NewCNode({NewValueNode(reshape_op), x_, NewValueNode(new_shape)});
        new_node->set_abstract(node_abstract);
        return new_node;
      }
      auto new_node = node->func_graph()->NewCNode({NewValueNode(reshape_op), x_, NewValueNode(new_shape)});
      if (AnfUtils::GetDumpFlag(node)) {
        AnfUtils::SetDumpFlag(new_node);
      }
      return new_node;
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

    // axis : int64_t
    if (IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      // axis could be negative
      if (idx < 0) {
        idx += SizeToLong(x_shape_.size());
      }
      if (SizeToLong(x_shape_.size()) > idx && x_shape_[LongToSize(idx)] == 1) {
        is_axis_one_ = true;
        axis_.push_back(idx);
      }
      return;
    }

    // axis : tuple(int64_t), default ()
    if (IsValueNode<ValueTuple>(vnode)) {
      auto axis = GetValue<std::vector<int64_t>>(vnode->value());
      if (axis.empty()) {
        return;
      }

      auto cmp = std::all_of(axis.cbegin(), axis.cend(), [this](int64_t idx) {
        // axis could be negative
        if (idx < 0) {
          idx += SizeToLong(x_shape_.size());
        }
        return SizeToLong(this->x_shape_.size()) > idx && this->x_shape_[LongToSize(idx)] == 1;
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
  std::vector<int64_t> axis_{}, x_shape_{};
  AnfNodePtr x_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REDUCE_ELIMINATE_H_
