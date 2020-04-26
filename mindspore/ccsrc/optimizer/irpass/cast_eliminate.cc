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

#include "optimizer/irpass/cast_eliminate.h"
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
AnfNodePtr CastSameTypeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimCast, {IsNode, IsVNode})(node);

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
    return src_;
  }

  return nullptr;
}

void CastSameTypeEliminater::Visit(const AnfNodePtr &node) {
  if (src_ == nullptr) {
    src_ = node;
  } else {
    tgt_ = node;
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
