/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/bias_dropout_add_fusion.h"

#include <memory>
#include <vector>

#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef BiasDropoutAddFusion::DefinePattern() const {
  auto dropout = VectorRef({prim::kPrimDropout, VectorRef({prim::kPrimAdd, x_, bias_})});
  auto get_item = VectorRef({prim::kPrimTupleGetItem, dropout, index_});
  auto output = VectorRef({prim::kPrimAdd, residual_, get_item});
  return output;
}

const AnfNodePtr BiasDropoutAddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto x = utils::cast<AnfNodePtr>((*equiv)[x_]);
  auto bias = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  auto residual = utils::cast<AnfNodePtr>((*equiv)[residual_]);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(bias);
  MS_EXCEPTION_IF_NULL(residual);

  auto get_item = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 1);
  MS_EXCEPTION_IF_NULL(get_item);
  auto dropout = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(get_item), 0);
  MS_EXCEPTION_IF_NULL(dropout);
  auto bias_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(dropout), 0);
  MS_EXCEPTION_IF_NULL(bias_add);

  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(bias_add, 0);
  auto bias_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(bias_add, 1);
  if (bias_shape.size() > 1 || x_shape.size() <= 1 || bias_shape[0] != x_shape[1]) {
    return nullptr;
  }
  auto residual_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  if (residual_shape != x_shape) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(prim::kPrimBiasDropoutAdd->name());
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x, bias, residual};
  auto fused_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_node);

  fused_node->set_scope(dropout->scope());
  fused_node->set_abstract(dropout->abstract());
  common::AnfAlgo::CopyNodeAttrs(dropout, fused_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(dropout, fused_node);
  return get_item;
}
}  // namespace opt
}  // namespace mindspore
