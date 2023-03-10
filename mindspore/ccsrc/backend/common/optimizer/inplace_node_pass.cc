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

#include "include/backend/optimizer/inplace_node_pass.h"

namespace mindspore {
namespace opt {
AnfNodePtr InplaceNodePass::Run(const FuncGraphPtr &, const AnfNodePtr &node) {
  std::vector<AnfNodePtr> pre_inputs;
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto inputs = cnode->inputs();
    pre_inputs.insert(pre_inputs.end(), inputs.begin(), inputs.end());
  }
  bool ret = Process(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto inputs = cnode->inputs();
    if (inputs.size() != pre_inputs.size()) {
      MS_LOG(EXCEPTION) << "InplaceNodePass ERROR, the pass modify node: " << node->DebugString()
                        << ", pass name: " << name();
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      MS_EXCEPTION_IF_NULL(inputs[i]);
      MS_EXCEPTION_IF_NULL(pre_inputs[i]);
      if (!opt::AnfEqual(inputs[i], pre_inputs[i])) {
        MS_LOG(EXCEPTION) << "InplaceNodePass ERROR, the pass modify node: " << node->DebugString()
                          << ", pass name: " << name() << ", before node " << i << ":" << inputs[i]->DebugString()
                          << ", after node " << i << ":" << pre_inputs[i]->DebugString();
      }
    }
  }
  if (ret) {
    return node;
  } else {
    return nullptr;
  }
}
}  // namespace opt
}  // namespace mindspore
