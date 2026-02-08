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
#include "backend/common/pass/add_attr_to_node/add_attr_to_node.h"
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AddAttrToNode::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto impl = AddAttrToNodeImplRegistry::GetInstance().GetImplByOpName(op_name);
  if (impl == nullptr) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "add_attr_to_node matched op " << op_name << ", extended attr will add to this node, origin node is"
                << node->DebugString();
  return impl(graph, node);
}
}  // namespace opt
}  // namespace mindspore
