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
bool IsCNodePositive(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimReduceSum) || IsPrimitiveCNode(node, prim::kPrimSqueeze)) {
    return IsCNodePositive(node->cast<CNodePtr>()->input(1));
  }
  if (IsPrimitiveCNode(node, prim::kPrimSquare) || IsPrimitiveCNode(node, prim::kPrimSqrt)) {
    return true;
  }
  return false;
}

AnfNodePtr ValueBasedEliminate::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode x, y, z;
  PConstant zero_(node, false, 0);
  PConstant zero_scalar_(node, false, 0, true);

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreater, x, zero_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, PPrimitive(prim::kPrimGreater, x, zero_scalar_), y, z), y,
                   IsCNodePositive(x.GetNode(node)));

  return nullptr;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
