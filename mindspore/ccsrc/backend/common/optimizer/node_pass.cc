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
#include <vector>
#include <set>
#include <algorithm>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
const size_t kSwitchBranchIndex = 2;
const size_t kCallArgsIndex = 1;
const size_t kPartialArgsIndex = 1;
}  // namespace

void UpdateCallerAbstract(const AnfNodePtr &call_node, const FuncGraphPtr &call_node_fg,
                          const FuncGraphPtr &sub_graph) {
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(call_node_fg);
  MS_EXCEPTION_IF_NULL(sub_graph);
  MS_EXCEPTION_IF_NULL(sub_graph->output());
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

void ModifyOutputAndCallerToMap(const CNodePtr &cnode, const FuncGraphPtr &fg,
                                mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> *out_caller_map, bool is_add) {
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
    if (is_add) {
      (*out_caller_map)[switch_subgraph->output()].insert(cnode);
      UpdateCallerAbstract(cnode, fg, switch_subgraph);
    } else {
      (*out_caller_map)[switch_subgraph->output()].erase(cnode);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    auto call_subgraph = GetValueNode<FuncGraphPtr>(inputs.at(kCallArgsIndex));
    MS_EXCEPTION_IF_NULL(call_subgraph);
    if (is_add) {
      (*out_caller_map)[call_subgraph->output()].insert(cnode);
      UpdateCallerAbstract(cnode, fg, call_subgraph);
    } else {
      (*out_caller_map)[call_subgraph->output()].erase(cnode);
    }
  }
}

void UpdateSubGraphCaller(const AnfNodePtr &origin_output, const FuncGraphPtr &fg,
                          mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> *out_caller_map,
                          const mindspore::HashMap<AnfNodePtr, FuncGraphWeakPtr> &node_to_fg) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(fg->output());
  auto find_iter = (*out_caller_map).find(origin_output);
  if (find_iter != (*out_caller_map).end()) {
    auto call_node_list = find_iter->second;
    (*out_caller_map).erase(find_iter);
    for (auto &call_node : call_node_list) {
      auto fg_iter = node_to_fg.find(call_node);
      if (fg_iter == node_to_fg.end()) {
        MS_LOG(EXCEPTION) << "Node to Funcgraph find failed: " << call_node->fullname_with_scope();
      }
      auto call_node_fg = fg_iter->second.lock();
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

std::string GetCNodeKey(const AnfNodePtr &node) {
  auto primitive = GetCNodePrimitive(node);
  if (primitive != nullptr) {
    return primitive->name();
  } else {
    return "";
  }
}

void GenIndex(const FuncGraphPtr &func_graph, const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  if (func_graph_index->has_gen_index()) {
    return;
  }

  func_graph_index->set_has_gen_index(true);
  func_graph_index->node_to_fg_.clear();
  func_graph_index->node_degree_.clear();
  func_graph_index->name_to_cnode_.clear();
  func_graph_index->subgraph_out_caller_map_.clear();

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  mindspore::HashSet<AnfNodePtr> seen_node;
  std::deque<std::pair<AnfNodePtr, FuncGraphPtr>> todo{{func_graph->output(), func_graph}};

  while (!todo.empty()) {
    AnfNodePtr node = todo.front().first;
    MS_EXCEPTION_IF_NULL(node);
    auto fg = todo.front().second;
    manager->AddFuncGraph(fg);
    todo.pop_front();

    func_graph_index->node_to_fg_[node] = fg;
    auto degree_iter = func_graph_index->node_degree_.find(node);
    if (degree_iter == func_graph_index->node_degree_.end()) {
      func_graph_index->node_degree_[node] = 1;
    } else {
      degree_iter->second++;
    }
    if (node->isa<CNode>()) {
      func_graph_index->name_to_cnode_[GetCNodeKey(node)].insert(node);
    }

    if (seen_node.count(node) > 0 || !manager->all_nodes().contains(node)) {
      continue;
    }
    (void)seen_node.insert(node);
    TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));

    if (IsValueNode<FuncGraph>(node)) {
      auto const_func_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(const_func_graph);
      if (!const_func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
        (void)todo.emplace_back(const_func_graph->output(), const_func_graph);
      }
    } else if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      ModifyOutputAndCallerToMap(cnode, fg, &func_graph_index->subgraph_out_caller_map_);
      auto inputs = cnode->inputs();
      (void)std::for_each(inputs.begin(), inputs.end(),
                          [&fg, &todo](AnfNodePtr &node) { (void)todo.emplace_back(node, fg); });
    }
  }
}

bool NodePass::ProcessFastPassNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                   const FuncGraphIndexPtr &func_graph_index, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = func_graph_index->node_to_fg_.find(node);
  if (iter == func_graph_index->node_to_fg_.end()) {
    MS_LOG(EXCEPTION) << "Node to Funcgraph map can't find node: " << node->fullname_with_scope();
  }
  auto fg = iter->second.lock();
  TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));
  auto degree_iter = func_graph_index->node_degree_.find(node);
  if (degree_iter == func_graph_index->node_degree_.end()) {
    MS_LOG(EXCEPTION) << "Node degree map can't find node: " << node->fullname_with_scope();
  }
  auto degree = degree_iter->second;
  if (degree == 0 && node != func_graph->output()) {
    return false;
  }
  // we may update return value in some pass.
  MS_EXCEPTION_IF_NULL(fg);
  auto origin_output = fg->output();
  MS_EXCEPTION_IF_NULL(origin_output);
  auto origin_abstract = origin_output->abstract();
  AnfNodePtr new_node = Run(fg, node);
  bool change = (new_node != nullptr);
  MS_EXCEPTION_IF_NULL(fg->output());
  if (origin_abstract != fg->output()->abstract()) {
    UpdateSubGraphCaller(origin_output, fg, &func_graph_index->subgraph_out_caller_map_, func_graph_index->node_to_fg_);
  }
  if (new_node != nullptr && new_node != node) {
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
    AfterProcess(node, new_node, fg, func_graph_index);
  }
  return change;
}

bool NodePass::ProcessFastPass(const FuncGraphPtr &func_graph, const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  if (!func_graph_index->has_gen_index()) {
    MS_LOG(EXCEPTION) << "ProcessFastPass Error, func graph has not gen index, pass name: " << name();
  }
  auto src_pattern_root_name = GetPatternRootPrimitiveName();
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changes = false;

  std::vector<AnfNodePtr> cand_node;
  if (!src_pattern_root_name.empty()) {
    auto cnode_iter = func_graph_index->name_to_cnode_.find(src_pattern_root_name);
    if (cnode_iter == func_graph_index->name_to_cnode_.end()) {
      return false;
    }
    std::copy(cnode_iter->second.begin(), cnode_iter->second.end(), std::back_inserter(cand_node));
  } else {
    for (const auto &kv : func_graph_index->name_to_cnode_) {
      std::copy(kv.second.begin(), kv.second.end(), std::back_inserter(cand_node));
    }
  }
  for (const auto &node : cand_node) {
    auto change = ProcessFastPassNode(node, func_graph, func_graph_index, manager);
    changes = changes || change;
  }
  return changes;
}

bool NodePass::ProcessPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(manager);
  bool changes = false;

  // maybe call subgraph many times
  mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> subgraph_out_caller_map = {};
  mindspore::HashMap<AnfNodePtr, FuncGraphWeakPtr> node_to_fg = {};
  mindspore::HashSet<AnfNodePtr> seen_node;
  std::deque<std::pair<AnfNodePtr, FuncGraphPtr>> todo{{func_graph->output(), func_graph}};
  while (!todo.empty()) {
    AnfNodePtr node = todo.front().first;
    auto fg = todo.front().second;
    MS_EXCEPTION_IF_NULL(node);
    manager->AddFuncGraph(fg);
    todo.pop_front();
    node_to_fg[node] = fg;
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
      UpdateSubGraphCaller(origin_output, fg, &subgraph_out_caller_map, node_to_fg);
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
      ModifyOutputAndCallerToMap(cnode, fg, &subgraph_out_caller_map);
      auto inputs = cnode->inputs();
      (void)std::for_each(inputs.begin(), inputs.end(),
                          [&fg, &todo](AnfNodePtr &node) { (void)todo.emplace_back(node, fg); });
    }
    changes = changes || change;
  }
  return changes;
}

bool NodePass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  auto func_graph_index = manager->func_graph_index(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);

  if (IsFastPass()) {
    MS_LOG(INFO) << "Run fast pass: " << name();
    GenIndex(func_graph, func_graph_index);
    return ProcessFastPass(func_graph, func_graph_index);
  }
  if (func_graph_index->has_gen_index()) {
    auto ret = MustExistPrimitiveName();
    for (const auto &primtive_name : ret) {
      auto cnode_iter = func_graph_index->name_to_cnode_.find(primtive_name);
      if (cnode_iter == func_graph_index->name_to_cnode_.end()) {
        return false;
      }
    }
    if (!ret.empty()) {
      MS_LOG(INFO) << "Skip pass fail, run pass: " << name();
    }
  }
  func_graph_index->set_has_gen_index(false);

  return ProcessPass(func_graph, manager);
}
}  // namespace opt
}  // namespace mindspore
