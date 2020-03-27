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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "operator/ops.h"
#include "ir/func_graph.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/parse/python_adapter.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimCast, X, T}
class CastSameTypeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimCast, {IsNode, IsVNode})(node);

    // check pattern match
    if (tgt_ == nullptr) {
      return nullptr;
    }

    // src type check
    auto src_type = src_->Type();
    if (src_type == nullptr) {
      return nullptr;
    }

    if (src_type->isa<TensorType>()) {
      src_type = src_type->cast<TensorTypePtr>()->element();
    }

    // tgt type check
    auto tgt_type = GetValueNode<TypePtr>(tgt_);
    if (tgt_type->isa<TensorType>()) {
      tgt_type = tgt_type->cast<TensorTypePtr>()->element();
    }

    if (src_type->type_id() == tgt_type->type_id()) {
      return src_;
    }

    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (src_ == nullptr) {
      src_ = node;
    } else {
      tgt_ = node;
    }
  }

  void Reset() {
    src_ = nullptr;
    tgt_ = nullptr;
  }

 private:
  AnfNodePtr src_{nullptr}, tgt_{nullptr};
};

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}
class TwoCastEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimCast, {IsCNode, IsNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && x_ != nullptr && t_ != nullptr) {
      auto cast_op = parse::python_adapter::GetPyFn("mindspore.ops.operations", "Cast")();
      ValuePtr cast = parse::data_converter::PyDataToValue(cast_op);
      auto cnode = fg->NewCNode({NewValueNode(cast), x_, t_});
      cnode->set_abstract(node->abstract());
      return cnode;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      // {prim::kPrimCast, X, Y}
      if (inputs.size() != 3) {
        return;
      }
      x_ = inputs[1];
    } else {
      t_ = node;
    }
  }

  void Reset() {
    x_ = nullptr;
    t_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, t_{nullptr};
};

class CastEliminater {
 public:
  CastEliminater() : cast_same_type_eliminater_(), two_cast_eliminater_() {}
  ~CastEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    auto new_node = cast_same_type_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    new_node = two_cast_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    return nullptr;
  }

 private:
  CastSameTypeEliminater cast_same_type_eliminater_;
  TwoCastEliminater two_cast_eliminater_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
