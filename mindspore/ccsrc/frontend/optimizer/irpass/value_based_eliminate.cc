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

bool IsCNodePositive(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimReduceSum) || IsPrimitiveCNode(node, prim::kPrimSqueeze)) {
    return IsCNodePositive(node->cast<CNodePtr>()->input(1));
  }
  if (IsPrimitiveCNode(node, prim::kPrimSquare) || IsPrimitiveCNode(node, prim::kPrimSqrt)) {
    return true;
  }
  return false;
}

// check if a value is bigger than UPPER_FLT_LIMIT
bool IsNodeScalarMaxFLT(const AnfNodePtr &node) {
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
      return GetValue<float>(scalar) > UPPER_FLT_LIMIT;
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
    return data[0] > UPPER_FLT_LIMIT;
  }

  return false;
}

// check if a value is smaller than LOWER_FLT_LIMIT
bool IsNodeScalarMinFLT(const AnfNodePtr &node) {
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
      return GetValue<float>(scalar) < LOWER_FLT_LIMIT;
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
    return data[0] < LOWER_FLT_LIMIT;
  }

  return false;
}

AnfNodePtr ValueBasedEliminate::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x, y, z;
  PConstant zero_(node, false, 0);
  PConstant zero_scalar_(node, false, 0, true);

  // {prim::kPrimSelect, {prim::kPrimGreater, X, 0}, Y, Z}} -> Y when X is always greater than 0
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreater, x, zero_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreater, x, zero_scalar_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimMaximum, x, y), x, IsNodeScalarMinFLT(y.GetNode(node)));

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimMinimum, x, y), x, IsNodeScalarMaxFLT(y.GetNode(node)));

  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
