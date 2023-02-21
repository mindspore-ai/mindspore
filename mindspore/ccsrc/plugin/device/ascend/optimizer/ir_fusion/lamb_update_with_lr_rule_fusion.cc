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
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_update_with_lr_rule_fusion.h"

#include <memory>
#include <vector>
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
const BaseRef LambUpdateWithLRRuleFusion::DefinePattern() const {
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(real_div);
  auto greater = std::make_shared<Primitive>(kGreaterOpName);
  MS_EXCEPTION_IF_NULL(greater);

  VectorRef pattern_real_div0({real_div, input1_, input2_});
  VectorRef pattern_greater0({greater, input0_, constant_greater_max_});
  VectorRef pattern_greater1({greater, input1_, constant_greater_max_});
  VectorRef pattern_select0({prim::kPrimSelect, pattern_greater0, pattern_real_div0, constant_select_});
  VectorRef pattern_select1({prim::kPrimSelect, pattern_greater1, pattern_select0, constant_select_});
  VectorRef pattern_minimum0({prim::kPrimMinimum, pattern_select1, constant_minimum_});
  VectorRef pattern_maximum0({prim::kPrimMaximum, pattern_minimum0, constant_greater_max_});
  VectorRef pattern_mul0({prim::kPrimMul, pattern_maximum0, input3_});
  VectorRef pattern_mul1({prim::kPrimMul, pattern_mul0, input4_});
  VectorRef pattern({prim::kPrimSub, input5_, pattern_mul1});
  return pattern;
}

const AnfNodePtr LambUpdateWithLRRuleFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  auto input0 = utils::cast<AnfNodePtr>((*equiv)[input0_]);
  auto input1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  auto input2 = utils::cast<AnfNodePtr>((*equiv)[input2_]);
  auto input3 = utils::cast<AnfNodePtr>((*equiv)[input3_]);
  auto input4 = utils::cast<AnfNodePtr>((*equiv)[input4_]);
  auto input5 = utils::cast<AnfNodePtr>((*equiv)[input5_]);
  auto input6 = utils::cast<AnfNodePtr>((*equiv)[constant_greater_max_]);
  auto input7 = utils::cast<AnfNodePtr>((*equiv)[constant_select_]);
  auto input8 = utils::cast<AnfNodePtr>((*equiv)[constant_minimum_]);

  auto prim = std::make_shared<Primitive>(kLambUpdateWithLROpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {
    NewValueNode(prim), input0, input1, input2, input3, input4, input5, input6, input7, input8};
  auto lamb_update_with_lr = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(lamb_update_with_lr);

  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, lamb_update_with_lr.get());
  lamb_update_with_lr->set_scope(node->scope());
  return lamb_update_with_lr;
}
}  // namespace opt
}  // namespace mindspore
