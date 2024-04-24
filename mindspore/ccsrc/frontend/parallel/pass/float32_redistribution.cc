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

#include "frontend/parallel/pass/float32_redistribution.h"
#include <memory>
#include <vector>
#include <string>
#include <list>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"

namespace mindspore {
namespace parallel {
namespace {
AnfNodePtr InsertCastCNode(const FuncGraphPtr &graph, const AnfNodePtr &pre_node, TypeId data_type) {
  auto type_value = std::make_shared<Int64Imm>(static_cast<int64_t>(data_type));
  auto type_node = NewValueNode(type_value);
  type_node->set_abstract(type_value->ToAbstract());
  const std::vector<std::string> &input_names = {"x", "dst_type"};
  const std::vector<std::string> &output_names = {"output"};
  auto prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  prim->SetAttrs({{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
  auto cast_node = graph->NewCNode({NewValueNode(prim), pre_node, type_node});
  auto cast_abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(data_type), pre_node->abstract()->GetShape());
  cast_node->set_abstract(cast_abstract);
  return cast_node;
}
}  // namespace
void Float32Redistribution(const FuncGraphPtr &graph) {
  if (!ParallelContext::GetInstance()->force_fp32_communication()) {
    return;
  }
  auto manager = graph->manager();
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!(IsPrimitiveCNode(node, prim::kPrimReduceScatter) || IsPrimitiveCNode(node, prim::kPrimAllReduce))) {
        continue;
      }
      auto tensor_type = node->abstract()->BuildType();
      if (tensor_type->isa<TensorType>()) {
        tensor_type = tensor_type->cast<TensorTypePtr>()->element();
      }
      if (tensor_type->type_id() == kNumberTypeFloat32) {
        MS_LOG(WARNING) << node->fullname_with_scope() << " is fp32 comm, no need to transfer.";
        continue;
      }
      MS_LOG(WARNING) << "Force " << node->fullname_with_scope() << " to fp32 comm.";
      auto pre_cast_node = InsertCastCNode(each_graph, node->input(1), kNumberTypeFloat32);
      std::vector<AnfNodePtr> reduce_scatter_inputs{node->input(0), pre_cast_node};
      auto new_reduce_scatter = each_graph->NewCNode(reduce_scatter_inputs);
      auto new_reduce_scatter_abstract =
        std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeFloat32), node->abstract()->GetShapeTrack());
      node->set_abstract(new_reduce_scatter_abstract);
      new_reduce_scatter->set_abstract(node->abstract());
      auto post_cast_node = InsertCastCNode(each_graph, new_reduce_scatter, tensor_type->type_id());
      manager->Replace(node, post_cast_node);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
