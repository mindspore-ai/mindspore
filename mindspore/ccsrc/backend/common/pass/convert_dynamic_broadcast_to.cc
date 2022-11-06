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

#include <memory>
#include "backend/common/pass/convert_dynamic_broadcast_to.h"
#include "ir/anf.h"
#include "backend/common/optimizer/optimizer.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
const AnfNodePtr ConvertDynamicBroadcastTo::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (node_name == prim::kPrimDynamicBroadcastTo->name() && !common::AnfAlgo::IsDynamicShape(node)) {
    auto broadcast_to_op_name = prim::kPrimBroadcastTo->name();
    auto ori_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(ori_cnode);
    auto input_x = common::AnfAlgo::GetInputNode(ori_cnode, 0);
    CNodePtr broadcast_to_node =
      opt::NewCNode({NewValueNode(std::make_shared<Primitive>(broadcast_to_op_name)), input_x}, func_graph, {node});
    MS_EXCEPTION_IF_NULL(broadcast_to_node);
    broadcast_to_node->set_abstract(node->abstract());
    auto shape_ptr = node->abstract()->BuildShape()->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(shape_ptr->shape()), broadcast_to_node);
    return broadcast_to_node;
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
