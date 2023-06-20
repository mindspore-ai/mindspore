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

#include "backend/common/pass/broadcast_to_fusion.h"

#include <vector>
#include <memory>

#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef BroadcastToFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBroadcastTo, Xs});
}

const AnfNodePtr BroadcastToFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  const auto &origin_prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(origin_prim);
  const auto &origin_attrs = origin_prim->attrs();

  if (origin_attrs.count(kShape) == 0) {
    MS_LOG(DEBUG) << "Origin primitive: " << origin_prim->name() << "has no attr : " << kShape;
    return node;
  }
  auto attr_value = origin_prim->GetAttr(kShape);
  MS_EXCEPTION_IF_NULL(attr_value);

  auto input_x = GetValue<std::vector<int64_t>>(attr_value);
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex0);
  auto outer_dim_offset = input_x.size() - x_shape.size();
  bool flag = true;
  if (input_x.end() == find(input_x.begin(), input_x.end(), -1)) {
    flag = false;
  } else {
    flag = true;
  }

  if (flag) {
    for (size_t i = 0; i < input_x.size(); i++) {
      if (input_x[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << "For '" << origin_prim
                                   << "', -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
        input_x[i] = x_shape[i - outer_dim_offset];
      }
    }
  }
  common::AnfAlgo::SetNodeAttr(kShape, MakeValue(input_x), cnode);
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
