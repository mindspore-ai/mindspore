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

#include "backend/optimizer/ascend/ir_fusion/lamb_update_with_lr_v2.h"
#include <memory>
#include <algorithm>
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
const BaseRef LambUpdateWithLrV2::DefinePattern() const {
  const auto prim_greater = std::make_shared<Primitive>(kGreaterOpName);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  const size_t kZeroIndex = 0;
  const size_t kFirstIndex = 1;
  const size_t kSecondIndex = 2;
  const size_t kThirdIndex = 3;
  const size_t kFourthIndex = 4;
  const size_t kFifthIndex = 5;
  const size_t kSixthIndex = 6;

  VectorRef greater0({prim_greater, input_varptr_[kZeroIndex], input_varptr_[kFifthIndex]});
  VectorRef greater1({prim_greater, input_varptr_[kFirstIndex], input_varptr_[kFifthIndex]});
  VectorRef real_div0({prim_deal_div, input_varptr_[kZeroIndex], input_varptr_[kFirstIndex]});
  VectorRef select0({prim::kPrimSelect, greater1, real_div0, input_varptr_[kSixthIndex]});
  VectorRef select1({prim::kPrimSelect, greater0, select0, input_varptr_[kSixthIndex]});
  VectorRef mul0({prim::kPrimMul, select1, input_varptr_[kSecondIndex]});
  VectorRef mul1({prim::kPrimMul, mul0, input_varptr_[kThirdIndex]});

  return VectorRef({prim::kPrimSub, input_varptr_[kFourthIndex], mul1});
}

const AnfNodePtr LambUpdateWithLrV2::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  auto prim = std::make_shared<Primitive>(kLambUpdateWithLrV2OpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  (void)std::transform(input_varptr_.begin(), input_varptr_.end(), std::back_inserter(inputs),
                       [&equiv](const VarPtr &in) { return utils::cast<AnfNodePtr>((*equiv)[in]); });
  auto lamb_update_with_lr_v2 = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(lamb_update_with_lr_v2);
  lamb_update_with_lr_v2->set_abstract(node->abstract());

  return lamb_update_with_lr_v2;
}
}  // namespace opt
}  // namespace mindspore
