/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/optimizer/node_pass.h"

#include <deque>
#include <utility>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const size_t kSwitchBranchIndex = 2;
const size_t kCallArgsIndex = 1;
const size_t kPartialArgsIndex = 1;

void UpdateCallerAbstract(const AnfNodePtr &call_node, const FuncGraphPtr &call_node_fg,
                          const FuncGraphPtr &sub_graph) {
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(call_node_fg);
  call_node->set_abstract(sub_graph->output()->abstract());
  auto manager = call_node_fg->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // need update TupleGetItem abstract after call node
  auto &node_users = manager->node_users();
  auto iter = node_users.find(call_node);
  if (iter == node_users.end()) {
    return;
  }
  for (auto &node_index : iter->second) {
    auto used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (!common::AnfAlgo::CheckPrimitiveType(used_node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto idx = common::AnfAlgo::GetTupleGetItemOutIndex(used_node->cast<CNodePtr>());
    std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(call_node, idx)};
    auto shapes = {common::AnfAlgo::GetOutputInferShape(call_node, idx)};
    common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, used_node.get());
  }
}

void AddOutputAndCallerToMap(
  const CNodePtr &cnode, const FuncGraphPtr &fg,
  mindspore::HashMap<AnfNodePtr, std::vector<std::pair<AnfNodePtr, FuncGraphPtr>>> *out_caller_map) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(out_caller_map);
  auto inputs = cnode->inputs();
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    auto partial_node = dyn_cast<CNode>(inputs.at(kSwitchBranchIndex));
    MS_EXCEPTION_IF_NULL(partial_node);
    const auto &partial_inputs = partial_node->inputs();
    if (!IsPrimitive(partial_inputs.at(0), prim::kPrimPartial)) {
      MS_LOG(EXCEPTION) << "Invalid switch node: " << cnode->DebugString();
    }
    auto switch_subgraph = GetValueNode<FuncGraphPtr>(partial_inputs.at(kPartialArgsIndex));
    MS_EXCEPTION_IF_NULL(switch_subgraph);
    (*out_caller_map)[switch_subgraph->output()].emplace_back(cnode, fg);
    UpdateCallerAbstract(cnode, fg, switch_subgraph);
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    auto call_subgraph = GetValueNode<FuncGraphPtr>(inputs.at(kCallArgsIndex));
    MS_EXCEPTION_IF_NULL(call_subgraph);
    (*out_caller_map)[call_subgraph->output()].emplace_back(cnode, fg);
    UpdateCallerAbstract(cnode, fg, call_subgraph);
  }
}

void UpdateSubGraphCaller(
  const AnfNodePtr &origin_output, const FuncGraphPtr &fg,
  mindspore::HashMap<AnfNodePtr, std::vector<std::pair<AnfNodePtr, FuncGraphPtr>>> *out_caller_map) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(fg->output());
  auto find_iter = (*out_caller_map).find(origin_output);
  if (find_iter != (*out_caller_map).end()) {
    auto call_node_list = find_iter->second;
    (*out_caller_map).erase(find_iter);
    for (auto &call_node_pair : call_node_list) {
      auto call_node = call_node_pair.first;
      auto call_node_fg = call_node_pair.second;
      UpdateCallerAbstract(call_node, call_node_fg, fg);
    }
    (*out_caller_map)[fg->output()] = call_node_list;
  }
}

void SkipSameOp(const AnfNodePtr &old_node, const AnfNodePtr &new_node, mindspore::HashSet<AnfNodePtr> *seen_node) {
  MS_EXCEPTION_IF_NULL(seen_node);
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  if (old_node->isa<CNode>() && new_node->isa<CNode>() &&
      (common::AnfAlgo::GetCNodeName(old_node) == common::AnfAlgo::GetCNodeName(new_node))) {
    (void)seen_node->insert(new_node);
  }
}

bool NodePass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);

  // maybe call subgraph many times
  mindspore::HashMap<AnfNodePtr, std::vector<std::pair<AnfNodePtr, FuncGraphPtr>>> subgraph_out_caller_map = {};
  mindspore::HashSet<AnfNodePtr> seen_node;
  std::deque<std::pair<AnfNodePtr, FuncGraphPtr>> todo{{func_graph->output(), func_graph}};
  bool changes = false;
  while (!todo.empty()) {
    AnfNodePtr node = todo.front().first;
    auto fg = todo.front().second;
    manager->AddFuncGraph(fg);
    todo.pop_front();
    if (seen_node.count(node) > 0 || !manager->all_nodes().contains(node)) {
      continue;
    }
    (void)seen_node.insert(node);
    TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));
    // we may update return value in some pass.
    MS_EXCEPTION_IF_NULL(fg);
    auto origin_output = fg->output();
    MS_EXCEPTION_IF_NULL(origin_output);
    auto origin_abstract = origin_output->abstract();
    AnfNodePtr new_node = Run(fg, node);
    bool change = (new_node != nullptr);
    if (origin_abstract != fg->output()->abstract()) {
      UpdateSubGraphCaller(origin_output, fg, &subgraph_out_caller_map);
    }
    if (new_node != nullptr && new_node != node) {
      SkipSameOp(node, new_node, &seen_node);
      (void)manager->Replace(node, new_node);
      // if replaced node is end_goto, refresh relative params in kernel graph
      auto kernel_graph = fg->cast<std::shared_ptr<session::KernelGraph>>();
      if (kernel_graph != nullptr && node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        auto end_label = kernel_graph->get_end_goto();
        if (cnode == end_label && common::AnfAlgo::GetCNodeName(cnode) == kLabelSwitchOpName) {
          kernel_graph->set_end_goto(new_node->cast<CNodePtr>());
        }
      }
      (void)seen_node.erase(node);
    } else if (new_node == nullptr) {
      new_node = node;
    }
    if (new_node && IsValueNode<FuncGraph>(new_node)) {
      auto const_func_graph = GetValueNode<FuncGraphPtr>(new_node);
      MS_EXCEPTION_IF_NULL(const_func_graph);
      if (!const_func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
        (void)todo.emplace_back(const_func_graph->output(), const_func_graph);
      }
    } else if (new_node && new_node->isa<CNode>()) {
      if (common::AnfAlgo::IsGraphKernel(new_node)) {
        (void)todo.emplace_back(new_node, func_graph);
      }
      auto cnode = new_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      AddOutputAndCallerToMap(cnode, fg, &subgraph_out_caller_map);
      auto inputs = cnode->inputs();
      (void)std::for_each(inputs.begin(), inputs.end(), [&fg, &todo](AnfNodePtr &node) {
        (void)todo.emplace_back(std::pair<AnfNodePtr, FuncGraphPtr>(node, fg));
      });
    }
    changes = changes || change;
  }
  return changes;
}
}  // namespace opt
}  // namespace mindspore
