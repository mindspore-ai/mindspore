/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/gather_fusion.h"

#include <memory>

#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/op_utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef GatherFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimGather, Xs});
}

const AnfNodePtr GatherFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  size_t idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "batch_dims");
  ValuePtr batch_dim = MakeValue(static_cast<int64_t>(0));
  if (idx != SIZE_MAX) {
    auto batch_dim_node = common::AnfAlgo::GetInputNode(cnode, idx);
    if (utils::isa<ValueNodePtr>(batch_dim_node)) {
      auto batch_dim_v = ops::GetScalarValue<int64_t>(batch_dim_node->cast<ValueNodePtr>()->value());
      if (batch_dim_v.has_value()) {
        batch_dim = batch_dim_node->cast<ValueNodePtr>()->value();
      }
    }
  }

  if (!common::AnfAlgo::HasNodeAttr("batch_dims", cnode)) {
    common::AnfAlgo::SetNodeAttr("batch_dims", batch_dim, cnode);
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
