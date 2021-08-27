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

#include "tools/optimizer/graph/reduce_same_act_pass.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "ops/fusion/activation.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMinUsersSize = 2;
}
bool ReduceSameActPass::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  // this pass handle this: multi output ops with >2 same relu.
  // after pass become: multi output ops with exactly 1 relu.
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cur_node_users = func_graph->manager()->node_users()[node];
    if (cur_node_users.size() < kMinUsersSize) {
      continue;
    }

    // OrderedMap<AnfNodePtr, AnfNodeIndexSet>
    // OrderedSet<std::pair<AnfNodePtr, int>
    int relu_count = 0;
    AnfNodePtr relu_anf_node_ptr = nullptr;
    for (const auto &node_user : cur_node_users) {
      if (!CheckPrimitiveType(node_user.first, prim::kPrimActivation)) {
        continue;
      }
      auto post_cnode = node_user.first->cast<CNodePtr>();
      if (post_cnode == nullptr) {
        return false;
      }
      auto primitive_c = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(post_cnode->input(0));
      if (primitive_c == nullptr) {
        return false;
      }
      if (primitive_c->get_activation_type() != mindspore::RELU) {
        continue;
      }
      relu_count++;
      if (relu_count == 1) {
        relu_anf_node_ptr = node_user.first;
      } else {
        func_graph->manager()->Replace(node_user.first, relu_anf_node_ptr);
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
