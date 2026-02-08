/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/parallel/pass/overlap_param_gather.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>

#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/utils/parallel_context.h"
#include "ir/graph_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/utils.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace parallel {
namespace {
typedef struct ParamGatherUser {
 public:
  ParamGatherUser(const CNodePtr &param_gather_cnode_, const std::vector<std::pair<AnfNodePtr, size_t>> &users_,
                  const std::vector<size_t> &user_indexes_)
      : param_gather_cnode(param_gather_cnode_), users(users_), user_indexes(user_indexes_) {}
  CNodePtr param_gather_cnode;
  std::vector<std::pair<AnfNodePtr, size_t>> users;
  std::vector<size_t> user_indexes;
} ParamGatherUser;

bool ParamGatherTopoSort(const ParamGatherUser &a, const ParamGatherUser &b) {
  auto user_size = std::min(a.users.size(), b.users.size());
  for (size_t i = 0; i < user_size; ++i) {
    if (a.users[i].second == b.users[i].second) {
      continue;
    }
    return a.users[i].second < b.users[i].second;
  }
  return a.users.size() < b.users.size();
}

FuncGraphPtr GetNextFuncGraph(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || cnode->inputs().empty()) {
    return nullptr;
  }
  auto value_node = cnode->input(kIndex0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  return value->cast<FuncGraphPtr>();
}
}  // namespace

std::vector<ParamGatherUser> ExtractOrderedParamGatherNodes(const FuncGraphPtr &graph) {
  auto ret = graph->get_return();
  auto param_gather_filter = [](const AnfNodePtr &node) {
    return !IsPrimitiveCNode(node, prim::kPrimAllGather) ||
           !common::AnfAlgo::IsFromParallelOptimizer(node->cast<CNodePtr>());
  };
  auto param_gather_nodes = DeepScopedGraphSearchWithFilter(ret, AlwaysInclude, param_gather_filter);
  std::vector<ParamGatherUser> param_gather_user_list;
  for (auto param_gather_node : param_gather_nodes) {
    auto param_gather_cnode = param_gather_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(param_gather_cnode);

    std::vector<std::pair<AnfNodePtr, size_t>> users;
    std::vector<size_t> user_indexes;
    AnfNodePtr cur_node = param_gather_cnode;
    FuncGraphPtr cur_func_graph = graph;
    do {
      const auto &all_nodes = TopoSort(cur_func_graph->get_return());
      cur_func_graph = nullptr;
      auto param_gather_cnode_user_list = GetOutputNodesWithFilter(cur_node, [](const AnfNodePtr &node) {
        return IsOneOfPrimitiveCNode(node, {prim::kPrimLoad, prim::kPrimDepend, prim::kPrimMakeTuple, prim::kPrimCast});
      });
      for (size_t topo_order = 0; topo_order < all_nodes.size(); ++topo_order) {
        auto node = all_nodes[topo_order];
        auto iter =
          std::find_if(param_gather_cnode_user_list.begin(), param_gather_cnode_user_list.end(),
                       [&node](const std::pair<AnfNodePtr, int> &user_pair) { return node == user_pair.first; });
        if (iter != param_gather_cnode_user_list.end()) {
          (void)users.emplace_back(std::make_pair(node, topo_order));
          user_indexes.push_back(IntToSize(iter->second));
          cur_func_graph = GetNextFuncGraph(iter->first);
          if (cur_func_graph) {
            cur_node = cur_func_graph->parameters().at(iter->second - 1);
          }
          break;
        }
      }
    } while (cur_func_graph);
    (void)param_gather_user_list.emplace_back(ParamGatherUser(param_gather_cnode, users, user_indexes));
  }
  std::sort(param_gather_user_list.begin(), param_gather_user_list.end(), ParamGatherTopoSort);
  return param_gather_user_list;
}

void InsertDependByOrder(const std::vector<ParamGatherUser> &ordered_param_gather_nodes) {
  if (ordered_param_gather_nodes.empty()) {
    return;
  }
  auto pre_param_gather_node = ordered_param_gather_nodes.front();
  auto func_graph = pre_param_gather_node.param_gather_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto iter = ordered_param_gather_nodes.begin() + 1; iter != ordered_param_gather_nodes.end(); ++iter) {
    auto cur_param_gather_node = *iter;
    InsertDepend(pre_param_gather_node.param_gather_cnode, cur_param_gather_node.param_gather_cnode, manager,
                 func_graph, "param_gather_in_order_depend");
    pre_param_gather_node = cur_param_gather_node;
  }
  // Insert depend to control last param gather execute before first param gather node user
  InsertDepend(ordered_param_gather_nodes.back().param_gather_cnode,
               ordered_param_gather_nodes.front().users.front().first, manager, func_graph,
               "param_gather_compute_depend", ordered_param_gather_nodes.front().user_indexes.front());
}

void OverlapParamGather(const FuncGraphPtr &func_graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_OPT_SHARD_COMM_OPT);
  if (!is_enable) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  circle_handler::SetAttrToDepend(func_graph);
  const auto &node_list = TopoSort(func_graph->get_return());

  // Solve for cell_reuse
  // find user across subgraph
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }

  auto ordered_param_gather_nodes = ExtractOrderedParamGatherNodes(func_graph);
  if (ordered_param_gather_nodes.empty()) {
    MS_LOG(DEBUG) << "Cannot find any parameter gather nodes, no need to overlap param gather.";
    return;
  }
  InsertDependByOrder(ordered_param_gather_nodes);

  circle_handler::DetectAndRevertGraphCircle(func_graph, func_graph->manager(), "OverlapParamGather",
                                             "enable_opt_shard_comm_opt");
}
}  // namespace parallel
}  // namespace mindspore
