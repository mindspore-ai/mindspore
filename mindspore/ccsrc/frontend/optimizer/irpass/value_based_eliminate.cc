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

#include "frontend/optimizer/irpass/value_based_eliminate.h"

namespace mindspore {
namespace opt {
namespace irpass {
#define UPPER_FLT_LIMIT (FLT_MAX / 2.0)
#define LOWER_FLT_LIMIT (-FLT_MAX / 2.0)
// Define the checking mode
enum class ScalarCheckingMode : int64_t { GREATER_EQUAL = 0, LESS };

bool IsNodeScalarTrueWith(const AnfNodePtr &node, const ScalarCheckingMode &checking_mode, const float &check_value) {
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return false;
  }

  auto value = value_node->value();
  if (value == nullptr) {
    return false;
  }

  auto scalar = value->cast<ScalarPtr>();
  if (scalar != nullptr) {
    if (scalar->isa<FloatImm>()) {
      if (checking_mode == ScalarCheckingMode::GREATER_EQUAL) {
        return GetValue<float>(scalar) >= check_value;
      }
      return GetValue<float>(scalar) < check_value;
    }
  }
  // Check for Tensor [] or Tensor [1]
  auto tensor_ptr = value->cast<tensor::TensorPtr>();
  if (tensor_ptr == nullptr) {
    return false;
  }
  if (tensor_ptr->DataSize() > 1) {
    return false;
  }

  TypeId tensor_type = tensor_ptr->Dtype()->type_id();
  if ((tensor_type == TypeId::kNumberTypeFloat32) || (tensor_type == TypeId::kNumberTypeFloat)) {
    float *data = reinterpret_cast<float *>(tensor_ptr->data_c());
    if (checking_mode == ScalarCheckingMode::GREATER_EQUAL) {
      return data[0] >= check_value;
    }
    return data[0] < check_value;
  }

  return false;
}

// check if a value is greater or equal 0.0
bool IsNodeScalarPositive(const AnfNodePtr &node) {
  return IsNodeScalarTrueWith(node, ScalarCheckingMode::GREATER_EQUAL, 0.0);
}

bool IsCNodePositive(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimReduceSum) || IsPrimitiveCNode(node, prim::kPrimSqueeze)) {
    return IsCNodePositive(node->cast<CNodePtr>()->input(1));
  }
  if (IsPrimitiveCNode(node, prim::kPrimSquare) || IsPrimitiveCNode(node, prim::kPrimSqrt)) {
    return true;
  }
  if (IsPrimitiveCNode(node, prim::kPrimMinimum) || IsPrimitiveCNode(node, prim::kPrimRealDiv)) {
    auto first_node_positive =
      IsCNodePositive(node->cast<CNodePtr>()->input(1)) || IsNodeScalarPositive(node->cast<CNodePtr>()->input(1));
    auto second_node_positive =
      IsCNodePositive(node->cast<CNodePtr>()->input(2)) || IsNodeScalarPositive(node->cast<CNodePtr>()->input(2));
    return first_node_positive && second_node_positive;
  }

  return false;
}

// check if a value is greater or equal UPPER_FLT_LIMIT
bool IsNodeScalarMaxFLT(const AnfNodePtr &node) {
  return IsNodeScalarTrueWith(node, ScalarCheckingMode::GREATER_EQUAL, UPPER_FLT_LIMIT);
}

// check if a value is smaller than LOWER_FLT_LIMIT
bool IsNodeScalarMinFLT(const AnfNodePtr &node) {
  return IsNodeScalarTrueWith(node, ScalarCheckingMode::LESS, LOWER_FLT_LIMIT);
}

AnfNodePtr ValueBasedEliminate::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x, y, z;
  PConstant zero_(node, false, 0);
  PConstant zero_scalar_(node, false, 0, true);

  // {prim::kPrimSelect, {prim::kPrimGreater, X, 0}, Y, Z}} -> Y when X is always greater than 0
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreaterEqual, x, zero_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreaterEqual, x, zero_scalar_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  // {prim::kPrimMaximum, X, LOWER_FLT_LIMIT}} -> X
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimMaximum, x, y), x, IsNodeScalarMinFLT(y.GetNode(node)));

  // {prim::kPrimMinimum, X, UPPER_FLT_LIMIT}} -> X
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimMinimum, x, y), x, IsNodeScalarMaxFLT(y.GetNode(node)));

  // {prim::kPrimMaximum, X, 0}} -> X when X is always greater or equal 0
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimMaximum, x, zero_), x, IsCNodePositive(x.GetNode(node)));

  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
