/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/prelu_fusion.h"

#include <memory>
#include <vector>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef PReluFusion::DefinePattern() const {
  VectorRef x_pattern({prim::kPrimRelu, VectorRef({prim::kPrimNeg, x_})});
  VectorRef mul_pattern({prim::kPrimMul, VectorRef({prim::kPrimNeg, weight_}), x_pattern});
  VectorRef pattern({prim::kPrimAdd, VectorRef({prim::kPrimRelu, x_}), mul_pattern});
  return pattern;
}

const AnfNodePtr PReluFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  BaseRef &x_gnode = (*equiv)[x_];
  BaseRef &weight_gnode = (*equiv)[weight_];

  auto x = utils::cast<AnfNodePtr>(x_gnode);
  auto weight = utils::cast<AnfNodePtr>(weight_gnode);

  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(weight);

  auto prim = std::make_shared<Primitive>(kPReluOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x, weight};
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_abstract(node->abstract());
  fusion_node->set_scope(node->scope());
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
