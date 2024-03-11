/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/remove_cast_before_assign_add.h"

#include <memory>
#include <vector>
#include <list>
#include <string>
#include <utility>
#include <unordered_map>
#include <algorithm>

#include "mindspore/core/ops/framework_ops.h"

#include "utils/ms_context.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"

#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {
void RemoveCastBeforeAssignAdd(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend != kAscendDevice) {
    return;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAssignAdd)) {
        continue;
      }
      if (!IsPrimitiveCNode(node->cast<CNodePtr>()->input(kIndex2), prim::kPrimCast)) {
        continue;
      }
      auto assign_add_node = node->cast<CNodePtr>();
      auto cast_node = assign_add_node->input(kIndex2)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cast_node);
      auto src_data = cast_node->input(kIndex1);
      MS_EXCEPTION_IF_NULL(src_data);
      auto dst_data = assign_add_node->input(kIndex1);
      MS_EXCEPTION_IF_NULL(dst_data);
      auto src_type = src_data->Type();
      if (src_type->isa<TensorType>()) {
        src_type = src_type->cast<TensorTypePtr>()->element();
      }
      auto dst_type = dst_data->Type();
      if (dst_type->isa<TensorType>()) {
        dst_type = dst_type->cast<TensorTypePtr>()->element();
      }
      auto x_type_id = src_type->type_id();
      auto y_type_id = dst_type->type_id();
      if ((x_type_id == kNumberTypeFloat16 || x_type_id == kNumberTypeBFloat16) && y_type_id == kNumberTypeFloat32) {
        MS_LOG(INFO) << "Remove cast node:" << cast_node->fullname_with_scope();
        std::vector<AnfNodePtr> assign_add_cast{NewValueNode(prim::kPrimAssignAdd->Clone()), node->input(kIndex1),
                                                cast_node->input(kIndex1)};
        auto assign_add_cnode = each_graph->NewCNode(assign_add_cast);
        assign_add_cnode->set_abstract(node->abstract()->Clone());
        (void)manager->Replace(node, assign_add_cnode);
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
