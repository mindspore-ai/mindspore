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
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"

namespace mindspore {
namespace opt {
const AnfNodePtr HShrinkModifyLambd(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Modify HShrink lambd from negative to 0.";
  auto lambd_index = cnode->size() - 1;
  auto lambd_node = cnode->input(lambd_index);
  if (lambd_node->isa<ValueNode>()) {
    auto lambd_value_node = lambd_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(lambd_value_node);
    auto lambd = GetValue<float>(lambd_value_node->value());
    if (lambd < 0) {
      lambd_value_node->set_value(MakeValue(0.0f));
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
