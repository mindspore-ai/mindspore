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
#include "plugin/device/ascend/optimizer/ir_fusion/clip_by_norm_no_div_square_sum_fusion.h"

#include <memory>
#include <vector>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
const BaseRef ClipByNormNoDivSquareSumFusion::DefinePattern() const {
  auto greater = std::make_shared<Primitive>(kGreaterOpName);
  MS_EXCEPTION_IF_NULL(greater);
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(sqrt);

  VectorRef greater_pattern({greater, input_, constant_greater_});
  VectorRef pattern(
    {prim::kPrimMaximum,
     VectorRef({prim::kPrimSelect, greater_pattern,
                VectorRef({sqrt, VectorRef({prim::kPrimSelect, greater_pattern, input_, constant_select_})}), input_}),
     constant_maximum_});
  return pattern;
}

const AnfNodePtr ClipByNormNoDivSquareSumFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  BaseRef &input_gnode = (*equiv)[input_];
  BaseRef &constant_select_gnode = (*equiv)[constant_select_];
  BaseRef &constant_greater_gnode = (*equiv)[constant_greater_];
  BaseRef &constant_maximum_gnode = (*equiv)[constant_maximum_];
  auto input = utils::cast<AnfNodePtr>(input_gnode);
  auto constant_select = utils::cast<AnfNodePtr>(constant_select_gnode);
  auto constant_greater = utils::cast<AnfNodePtr>(constant_greater_gnode);
  auto constant_maximum = utils::cast<AnfNodePtr>(constant_maximum_gnode);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(constant_select);
  MS_EXCEPTION_IF_NULL(constant_greater);
  MS_EXCEPTION_IF_NULL(constant_maximum);

  auto prim = std::make_shared<Primitive>(kClipByNormNoDivSumOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input, constant_greater, constant_select, constant_maximum};
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, fusion_node.get());
  fusion_node->set_scope(node->scope());
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
