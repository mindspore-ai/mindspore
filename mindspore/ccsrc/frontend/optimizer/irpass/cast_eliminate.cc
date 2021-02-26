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

#include "frontend/optimizer/irpass/cast_eliminate.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "ir/func_graph.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/python_adapter.h"

namespace mindspore {
namespace opt {
namespace irpass {
AnfNodePtr TransThroughDepend(const AnfNodePtr &node) {
  auto cur_node = node;
  while (IsPrimitiveCNode(cur_node, prim::kPrimDepend)) {
    cur_node = cur_node->cast<CNodePtr>()->input(1);
  }
  return cur_node;
}

bool IsValueNode(const AnfNodePtr &node) { return IsVNode(TransThroughDepend(node)); }

// {prim::kPrimCast, X, T}
AnfNodePtr CastSameTypeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimCast, {IsNode, IsValueNode})(node);

  // check pattern match
  if (tgt_ == nullptr) {
    return nullptr;
  }

  // src type check
  auto src_type = src_->Type();
  if (src_type == nullptr || !src_type->isa<TensorType>()) {
    return nullptr;
  }

  src_type = src_type->cast<TensorTypePtr>()->element();

  // tgt type check
  auto tgt_type = GetValueNode<TypePtr>(tgt_);
  if (tgt_type->isa<TensorType>()) {
    tgt_type = tgt_type->cast<TensorTypePtr>()->element();
  }

  if (src_type->type_id() == tgt_type->type_id()) {
    // If 2nd input of cast is a depend, can't erase cast directly, but should replace cast with a new depend.
    if (IsPrimitiveCNode(node->cast<CNodePtr>()->input(2), prim::kPrimDepend)) {
      auto new_depend =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), src_, node->cast<CNodePtr>()->input(2)});
      return new_depend;
    }
    return src_;
  }

  return nullptr;
}

void CastSameTypeEliminater::Visit(const AnfNodePtr &node) {
  if (src_ == nullptr) {
    src_ = node;
  } else {
    tgt_ = TransThroughDepend(node);
  }
}

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}
AnfNodePtr TwoCastEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimCast, {IsCNode, IsNode})(node);

  if (x_ != nullptr && t_ != nullptr) {
    auto cast_op = parse::python_adapter::GetPyFn("mindspore.ops.operations", "Cast")();
    ValuePtr cast = parse::data_converter::PyDataToValue(cast_op);
    auto cnode = NewCNode({NewValueNode(cast), x_, t_}, node->func_graph());
    cnode->set_abstract(node->abstract());
    return cnode;
  }
  return nullptr;
}

void TwoCastEliminater::Visit(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimCast)) {
    auto cnode = node->cast<CNodePtr>();
    // {prim::kPrimCast, X, Y}
    if (cnode->size() != 3) {
      return;
    }
    x_ = cnode->input(1);
  } else {
    t_ = node;
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
