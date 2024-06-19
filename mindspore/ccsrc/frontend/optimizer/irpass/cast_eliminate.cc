/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "ir/func_graph.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace opt {
namespace irpass {
AnfNodePtr TransThroughDepend(const AnfNodePtr &node) {
  auto cur_node = node;
  while (IsPrimitiveCNode(cur_node, prim::kPrimDepend)) {
    cur_node = cur_node->cast<CNodePtr>()->input(1);
    const auto &abs = node->abstract();
    if (abs != nullptr) {
      cur_node->set_abstract(abs);
    }
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
  MS_EXCEPTION_IF_NULL(src_type);

  // tgt type check
  auto tgt_type_id_value = GetValueNode<Int64ImmPtr>(tgt_);
  MS_EXCEPTION_IF_NULL(tgt_type_id_value);
  int64_t tgt_type_id = tgt_type_id_value->value();

  if (src_type->type_id() == tgt_type_id) {
    // If 2nd input of cast is a depend, can't erase cast directly, but should replace cast with a new depend.
    if (IsPrimitiveCNode(node->cast<CNodePtr>()->input(2), prim::kPrimDepend)) {
      auto new_depend =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), src_, node->cast<CNodePtr>()->input(2)});
      const auto &abs = src_->abstract();
      if (abs != nullptr) {
        new_depend->set_abstract(abs);
      }
      return new_depend;
    }
    // Temporary patch for the output dtype mismatch, ResizeBilinear on Ascend always return Float32 tensor.
    if (IsPrimitiveCNode(node->cast<CNodePtr>()->input(1), prim::kPrimResizeBilinearV2)) {
      return nullptr;
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

bool TwoCastEliminater::CheckTwoTypes(const std::map<TypeId, int> &type_map, TypeId type1, TypeId type2) const {
  auto type1_iter = type_map.find(type1);
  auto type2_iter = type_map.find(type2);
  if (type1_iter != type_map.end() && type2_iter != type_map.end()) {
    return type1_iter->second <= type2_iter->second;
  }
  return false;
}

bool TwoCastEliminater::CheckThreeTypes(const std::map<TypeId, int> &type_map, TypeId type1, TypeId type2,
                                        TypeId type3) const {
  auto type1_iter = type_map.find(type1);
  auto type2_iter = type_map.find(type2);
  auto type3_iter = type_map.find(type3);
  if (type1_iter != type_map.end() && type2_iter != type_map.end() && type3_iter != type_map.end()) {
    return type1_iter->second <= type2_iter->second && type2_iter->second <= type3_iter->second;
  }
  return false;
}

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}  -> {prim::kPrimCast, X, T}
// y_type == t_type or x_type <= y_type or x_type >= y_type >= t_type
bool TwoCastEliminater::CheckTypesIsIncreasingOrDecreasing() {
  auto x_type = x_->Type();
  if (x_type->isa<TensorType>()) {
    x_type = x_type->cast<TensorTypePtr>()->element();
  }

  auto y_dtype_id_value = GetValueNode<Int64ImmPtr>(y_);
  if (y_dtype_id_value == nullptr) {
    return false;
  }
  auto y_type_id = static_cast<TypeId>(y_dtype_id_value->value());

  auto t_dtype_id_value = GetValueNode<Int64ImmPtr>(t_);
  if (t_dtype_id_value == nullptr) {
    return false;
  }
  auto t_type_id = static_cast<TypeId>(t_dtype_id_value->value());

  auto x_type_id = x_type->type_id();
  // y_type == t_type
  if (y_type_id == t_type_id) {
    return true;
  }
  // If the precision is increasing or decreasing, the cast can be eliminated.
  // x_type <= y_type
  bool increasing = CheckTwoTypes(int_map_, x_type_id, y_type_id) || CheckTwoTypes(uint_map_, x_type_id, y_type_id) ||
                    CheckTwoTypes(float_map_, x_type_id, y_type_id);
  if (increasing) {
    return true;
  }
  //  x_type >= y_type >= t_type
  return CheckThreeTypes(int_map_, t_type_id, y_type_id, x_type_id) ||
         CheckThreeTypes(uint_map_, t_type_id, y_type_id, x_type_id) ||
         CheckThreeTypes(float_map_, t_type_id, y_type_id, x_type_id);
}

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}
AnfNodePtr TwoCastEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimCast, {IsCNode, IsNode})(node);

  if (x_ == nullptr || t_ == nullptr || y_ == nullptr) {
    return nullptr;
  }
  // Sometimes the abstract information of the Depend node has not been derived.
  // the type of X is nullptr, {prim::kPrimCast, {prim::kPrimCast, Depend(W, Z), Y}, T}
  // In this case, we postpone the elimination of the two casts after the next renormalize.
  auto x_type = x_->Type();
  if (x_type == nullptr) {
    return nullptr;
  }
  if (CheckTypesIsIncreasingOrDecreasing()) {
    auto cnode = NewCNode({NewValueNode(prim::kPrimCast), x_, t_}, node->func_graph());
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_abstract(node->abstract());
    return cnode;
  }
  return nullptr;
}

void TwoCastEliminater::Visit(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimCast)) {
    auto cnode = node->cast<CNodePtr>();
    // {prim::kPrimCast, X, Y}
    constexpr size_t cast_size = 3;
    constexpr size_t cast_data_index = 1;
    constexpr size_t cast_type_index = 2;
    if (cnode->size() != cast_size) {
      return;
    }
    x_ = cnode->input(cast_data_index);
    y_ = cnode->input(cast_type_index);
  } else {
    t_ = node;
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
