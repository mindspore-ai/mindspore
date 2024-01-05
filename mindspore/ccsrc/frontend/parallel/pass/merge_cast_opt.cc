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

#include "frontend/parallel/pass/merge_cast_opt.h"
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/pass/pass_utils.h"

namespace mindspore {
namespace parallel {
namespace {
bool InsertMakeTupleInput(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node)) {
    return true;
  }
  if (!parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode->HasAttr(parallel::FINE_GRAINED_INTERLEAVED_TAG)) {
    auto tag = GetValue<size_t>(cnode->GetAttr(parallel::FINE_GRAINED_INTERLEAVED_TAG));
    if (tag != kIndex0) {
      return false;
    }
  }
  return true;
}

}  // namespace

void MergeCastOpt(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  if (!parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved()) {
    return;
  }
  auto manager = graph->manager();
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    std::unordered_map<AnfNodePtr, std::vector<CNodePtr>> cast_map;
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimCast) || !IsPrimitiveCNode(node->input(kIndex1), prim::kPrimDepend)) {
        continue;
      }
      auto depend_node = node->input(kIndex1)->cast<CNodePtr>();
      auto depend_node_first_input = GetInputNodeWithFilter(depend_node->input(kIndex1), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
        return std::make_pair(filter, 1);
      });
      if (depend_node_first_input->isa<ValueNode>()) {
        continue;
      }
      cast_map[depend_node_first_input].push_back(node);
    }
    for (const auto &pair : cast_map) {
      if (pair.second.size() <= 1) {
        continue;
      }
      std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
      std::vector<AbstractBasePtr> maketuple_abs_inputs;
      MS_LOG(INFO) << "Merged cast node input:" << pair.first->fullname_with_scope()
                   << ", unique_id:" << AnfNodeInfo(pair.first);
      for (const auto &cast_node : pair.second) {
        MS_LOG(INFO) << "Merged cast node:" << cast_node->fullname_with_scope()
                     << ", unique_id:" << AnfNodeInfo(cast_node);
        auto depend_node = cast_node->input(kIndex1)->cast<CNodePtr>();
        auto depend_second_input = depend_node->input(kIndex2);
        if (IsPrimitiveCNode(depend_second_input, prim::kPrimMakeTuple)) {
          auto make_tuple_cnode = depend_second_input->cast<CNodePtr>();
          for (size_t i = 1; i < make_tuple_cnode->size(); ++i) {
            if (!InsertMakeTupleInput(make_tuple_cnode->input(i))) {
              continue;
            }
            make_tuple_inputs.push_back(make_tuple_cnode->input(i));
            maketuple_abs_inputs.push_back(make_tuple_cnode->input(i)->abstract()->Clone());
          }
          continue;
        }
        if (!InsertMakeTupleInput(depend_second_input)) {
          continue;
        }
        make_tuple_inputs.push_back(depend_second_input);
        maketuple_abs_inputs.push_back(depend_second_input->abstract()->Clone());
      }
      auto new_make_tuple_node = each_graph->NewCNode(make_tuple_inputs);
      new_make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple_abs_inputs));
      std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), pair.first, new_make_tuple_node};
      auto new_depend_node = each_graph->NewCNode(depend_inputs);
      new_depend_node->set_abstract(pair.second.front()->input(kIndex1)->cast<CNodePtr>()->abstract()->Clone());
      std::vector<AnfNodePtr> cast_inputs{pair.second.front()->input(kIndex0), new_depend_node,
                                          pair.second.front()->input(kIndex2)};
      auto new_cast_node = each_graph->NewCNode(cast_inputs);
      new_cast_node->set_abstract(pair.second.front()->abstract()->Clone());
      for (const auto &cast_node : pair.second) {
        manager->Replace(cast_node, new_cast_node);
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
