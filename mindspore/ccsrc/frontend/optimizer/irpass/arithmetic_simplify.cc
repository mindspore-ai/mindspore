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

#include "frontend/optimizer/irpass/arithmetic_simplify.h"

namespace mindspore {
namespace opt {
namespace irpass {
AnfNodePtr ArithmeticSimplify::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x, y, z;
  PConstant one_(node, false, 1);
  PConstant one_scalar_(node, false, 1, true);
  PConstant zero_(node, false, 0);
  PConstant zero_scalar_(node, false, 0, true);
  PConstant const_(node);
  PConstant const_2(node);
  PConstant any_const(node);
  // if node has keep_alive attr, it would not be eliminated.
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->HasAttr("keep_alive") && GetValue<bool>(prim->GetAttr("keep_alive"))) {
      MS_LOG(INFO) << "keep node " << node->fullname_with_scope() << " alive";
      return nullptr;
    }
  }
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    auto IsAddByZeroSimplifiable = [node](const AnfNodePtr &real_x) {
      // If real_x is Load CNode, We should not simplify it as Load is a no-op at backend, after simplication, the
      // result of the Load may be incorrect.
      if (IsPrimitiveCNode(real_x, prim::kPrimLoad)) {
        MS_LOG(DEBUG) << "Cannot simplify as real_x is CNode Load: " << real_x->ToString();
        return false;
      }

      if (real_x->abstract() != nullptr && real_x->abstract()->GetShapeTrack() != nullptr &&
          node->abstract() != nullptr && node->abstract()->GetShapeTrack() != nullptr &&
          *real_x->abstract()->GetShapeTrack() == *node->abstract()->GetShapeTrack()) {
        MS_LOG(DEBUG) << "Can simplify when their shapes are same: real_x shape:"
                      << real_x->abstract()->GetShapeTrack()->ToString()
                      << ", node shape: " << node->abstract()->GetShapeTrack()->ToString();
        return true;
      }
      MS_LOG(DEBUG) << "Cannot simplify when their shapes are not same: real_x shape:"
                    << real_x->abstract()->GetShapeTrack()->ToString()
                    << ", node shape: " << node->abstract()->GetShapeTrack()->ToString();
      return false;
    };
    MATCH_REPLACE_IF(node, x + zero_, x, x.CheckFunc(IsAddByZeroSimplifiable, node));  // Add by zero

    MATCH_REPLACE(node, PBinOperation(prim::kPrimScalarAdd, x, zero_scalar_, true), x);          // Scalar Add by zero
    MATCH_REPLACE_IF(node, x * one_, any_const.WithValueOf(x), !one_.CheckFunc(IsParam, node));  // Multiply by one
    MATCH_REPLACE(node, PBinOperation(prim::kPrimScalarMul, x, one_scalar_, true), x);           // Scalar Mul by one

    // Scalar Mul by zero
    MATCH_REPLACE(node, PBinOperation(prim::kPrimScalarMul, x, zero_scalar_, true), zero_scalar_.NewValue());
  }
  // Prim Eliminate (identity)
  MATCH_REPLACE(node, PPrimitive(prim::kPrimIdentity, x), x);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return nullptr;
  }

  // ConstantDuplicateMul
  auto const_dup_lambda = [&node, &x, &const_, &const_2]() -> AnfNodePtr {
    auto new_mul_tensor = const_.MulByPatternConst(const_2, x.GetNode(node));
    auto mul_node = node->cast<CNodePtr>()->inputs()[0];
    if (new_mul_tensor == nullptr) {
      auto ttmul = NewCNode({mul_node, const_.GetNode(node), const_2.GetNode(node)}, node->func_graph());
      return NewCNode({mul_node, x.GetNode(node), ttmul}, node->func_graph());
    }
    auto new_cnode = NewCNode({mul_node, x.GetNode(node), new_mul_tensor}, node->func_graph());
    new_cnode->set_abstract(node->abstract());
    return new_cnode;
  };
  MATCH_REPLACE_LAMBDA(node, const_ * (const_2 * x), const_dup_lambda);

  if (node->func_graph() == nullptr) {
    return nullptr;
  }

  // OptUpdateZeroTensor: {kPrimMomentum, {kPrimZerosLike, x}, y, z, xs} -> {kPrimMakeTuple, z, y}
  MATCH_REPLACE(node, PPrimitive(prim::kPrimMomentum, PPrimitive(prim::kPrimZerosLike, x), y, z).MinExtraNodes(0),
                PPrimitive(prim::kPrimMakeTuple, z, y));

  // PowerOneEliminate
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimPow, x, one_scalar_), x,
                   one_scalar_.CheckFunc(IsValueNode<Scalar>, node));

  return nullptr;
}

AnfNodePtr ArithmeticSimplify2::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return nullptr;
  }
  PatternNode x, y;
  PConstant zero_(node, false, 0);

  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->IsDynamic()) {
    return nullptr;
  }

  MATCH_REPLACE_IF(node, x * zero_, zero_.WithShapeAs(node),
                   !zero_.CheckFunc(IsParam, node) && !x.CheckFunc(IsLoad, node) &&
                     x.GetNode(node)->func_graph() == node->func_graph());
  auto zero_prim = PPrimitive(prim::kPrimZerosLike, y);
  MATCH_REPLACE_IF(node, x * zero_prim, zero_.WithShapeAs(node),
                   !zero_prim.CheckFunc(IsParam, node) && x.GetNode(node)->func_graph() == node->func_graph());

  return nullptr;
}

// grad = AllReduce(grad) / worker_number
// grad = grad + weight * decy
// ->
// grad = grad + weight * decy
// grad = AllReduce(grad) / worker_number
// {prim::kPrimAddN, {prim::kPrimMakeTuple, {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}, Z}} ->
// {prim::kPrimMul, {prim::kPrimAllReduce, {prim::kPrimAddN,{prim::kPrimMakeTuple, Z, X}}}, Y}
AnfNodePtr AdjustAllReduceMulAdd::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x, y, z;
  auto all_reduce_pat = PPrimitive(prim::kPrimAllReduce, x);
  auto mul_pat = PBinOperation(prim::kPrimMul, all_reduce_pat, y, true);
  auto admktup_pat = PBinOperation(prim::kPrimMakeTuple, mul_pat, z, true);
  auto addn_pat = PPrimitive(prim::kPrimAddN, admktup_pat);
  auto adjust_lambda = [&node, &x, &y, &z, &addn_pat, &all_reduce_pat, &admktup_pat, &mul_pat, this]() -> AnfNodePtr {
    auto fg = all_reduce_pat.GetFuncGraph();
    auto z_ = z.GetNode(node);
    auto x_ = x.GetNode(node);

    // If addn inputs cross the graph, make the inputs same as allreduce node.
    if (z_->isa<CNode>() && fg != z_->func_graph()) {
      auto cnode_z = z_->cast<CNodePtr>();
      z_ = NewCNode(cnode_z->inputs(), fg);
    }

    auto addn_cnode = addn_pat.GetOriginalNode()->cast<CNodePtr>();
    auto addn_op_node = addn_cnode->input(0);
    auto make_tuple_op_node = addn_cnode->input(1)->cast<CNodePtr>()->input(0);
    auto all_reduce_prim = all_reduce_pat.GetOriginalNode()->cast<CNodePtr>()->input(0);
    mul_cnode_ = mul_pat.GetOriginalNode();
    auto mul_prim = mul_cnode_->cast<CNodePtr>()->input(0);
    auto addn_maketuple = admktup_pat.GetOriginalNode();

    ShapeVector x_shape, z_shape;
    if (!x_->isa<ValueNode>()) {
      if ((x_->abstract() == nullptr) || !x_->abstract()->isa<abstract::AbstractTensor>()) {
        return nullptr;
      }
      auto x_abstract = x_->abstract()->cast<abstract::AbstractTensorPtr>();
      x_shape = x_abstract->shape()->shape();
    } else {
      ValuePtr x_value = x_->cast<ValueNodePtr>()->value();
      if (!x_value->isa<tensor::Tensor>()) {
        return nullptr;
      }
      auto x_tensor = GetValueNode<tensor::TensorPtr>(x_->cast<ValueNodePtr>());
      x_shape = x_tensor->shape();
    }
    if (!z_->isa<ValueNode>()) {
      if ((z_->abstract() == nullptr) || !z_->abstract()->isa<abstract::AbstractTensor>()) {
        return nullptr;
      }
      auto z_abstract = z_->abstract()->cast<abstract::AbstractTensorPtr>();
      z_shape = z_abstract->shape()->shape();
    } else {
      ValuePtr z_value = z_->cast<ValueNodePtr>()->value();
      if (!z_value->isa<tensor::Tensor>()) {
        return nullptr;
      }
      auto z_tensor = GetValueNode<tensor::TensorPtr>(z_->cast<ValueNodePtr>());
      z_shape = z_tensor->shape();
    }

    if (x_shape != z_shape) {
      // AddN requires x_ and z_ have the same shape.
      // If broadcasting TensorAdd is supported then can use this
      return nullptr;
    }
    AnfNodePtr tuple = NewCNode({make_tuple_op_node, z_, x_}, fg);
    AnfNodePtr add = NewCNode({addn_op_node, tuple}, fg);
    AnfNodePtr all_reduce = NewCNode({all_reduce_prim, add}, fg);
    AnfNodePtr mul = NewCNode({mul_prim, all_reduce, y.GetNode(node)}, fg);
    ProcessDependEdge(fg, addn_maketuple, all_reduce);
    return mul;
  };
  MATCH_REPLACE_LAMBDA(node, addn_pat, adjust_lambda);
  return nullptr;
}

void AdjustAllReduceMulAdd::ProcessDependEdge(const FuncGraphPtr &fg, const AnfNodePtr &addn_maketuple,
                                              const AnfNodePtr &new_node) {
  // If has dynamic loss scale.
  MS_EXCEPTION_IF_NULL(fg);
  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &users_map = manager->node_users();
  auto it = users_map.find(mul_cnode_);
  if (it != users_map.end()) {
    auto users = it->second;
    for (auto &user_pair : users) {
      auto node = user_pair.first;
      if (node != addn_maketuple && IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
        manager->SetEdge(node, user_pair.second, new_node);
      }
    }
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
