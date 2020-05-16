/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/buffer_fusion/pass/fusion_type_fusion_pass.h"

#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <memory>
#include <algorithm>

#include "kernel/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
namespace {
const int8_t MAX_PATTERN_SIZE = 7;
const int8_t MIN_PATTERN_SIZE = 2;
const int8_t ELTWISE_INPUT_SIZE = 2;
const int8_t ELTWISE_USE = 1;
const int8_t MULTI_ELTWISE_USE = 2;
const int8_t MAX_MULTI_ELTWISE_SIZE = 4;
const int8_t MAX_PURE_BUFFER_SUCC_SIZE = 3;
constexpr auto kOpAttrFusionId = "fusion_id";

bool CheckEltWiseNode(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *record, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(record);
  auto user_nodes = manager->node_users()[node];
  return (AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
          AnfAlgo::GetFusionType(node) == kernel::FusionType::ELEMWISE &&
          (user_nodes.size() <= ELTWISE_USE || record->size() == 0));
}

// Common method to check for predecessors and successors in a fusion pattern
std::tuple<bool, CNodePtr> FindPredAndSuccEltWiseNodes(const int8_t &max_size, FuncGraphManager *manager,
                                                       std::unordered_set<AnfNodePtr> *visited_set,
                                                       std::deque<AnfNodePtr> *todo,
                                                       std::unordered_set<AnfNodePtr> *record, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(todo);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr new_node = node;
  if (new_node->inputs().size() < ELTWISE_INPUT_SIZE) {
    return std::make_tuple(false, new_node);
  }
  int8_t index = 1;
  auto &users = manager->node_users();
  while (CheckEltWiseNode(manager, record, new_node)) {
    (void)record->insert(new_node);
    (void)visited_set->insert(new_node);
    (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());

    auto cnode = new_node->input(1);
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->isa<CNode>()) {
      return std::make_tuple(false, new_node);
    }
    new_node = cnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(new_node);

    if (!AnfAlgo::IsRealKernel(new_node) || new_node->inputs().size() < ELTWISE_INPUT_SIZE ||
        users[(new_node)].size() >= MULTI_ELTWISE_USE || visited_set->find(new_node) != visited_set->end()) {
      return std::make_tuple(false, new_node);
    }

    if (index >= max_size) {
      break;
    }
    index++;
  }
  return std::make_tuple(true, new_node);
}

std::tuple<bool, CNodePtr> MatchGeneralPattern(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *record,
                                               std::unordered_set<AnfNodePtr> *visited_set,
                                               std::deque<AnfNodePtr> *todo, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(todo);
  CNodePtr new_node = node;
  auto &users = manager->node_users();
  if (users[(new_node)].size() >= MULTI_ELTWISE_USE) {
    return std::make_tuple(false, new_node);
  }

  (void)record->insert(node);
  (void)visited_set->insert(node);
  (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());

  if (node->inputs().size() < 2) {
    return std::make_tuple(false, new_node);
  }
  // only check the first real input, will check all
  auto cnode = node->input(1);
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->isa<CNode>()) {
    return std::make_tuple(false, new_node);
  }
  new_node = cnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node);

  if (!AnfAlgo::IsRealKernel(new_node) || users[(new_node)].size() >= MULTI_ELTWISE_USE ||
      visited_set->find(new_node) != visited_set->end()) {
    return std::make_tuple(false, new_node);
  }
  return std::make_tuple(true, new_node);
}

CNodePtr FindFusionAnfNode(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *visited_set,
                           std::unordered_set<AnfNodePtr> *record, std::deque<AnfNodePtr> *todo, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(todo);
  // find fusion pattern predecessor nodes
  auto ret = FindPredAndSuccEltWiseNodes(MAX_MULTI_ELTWISE_SIZE, manager, visited_set, todo, record, node);
  auto new_node = std::get<1>(ret);
  auto node_use_size = manager->node_users()[new_node].size();
  if (!std::get<0>(ret) || (record->size() > 1 && node_use_size > 1) || record->size() >= MAX_MULTI_ELTWISE_SIZE ||
      AnfAlgo::GetKernelType(new_node) != KernelType::TBE_KERNEL) {
    return new_node;
  }

  // key of fusion precessor
  auto node_fusion_type = AnfAlgo::GetFusionType(new_node);
  switch (node_fusion_type) {
    case kernel::FusionType::COMMREDUCE:
    case kernel::FusionType::SEGMENT:
      ret = MatchGeneralPattern(manager, record, visited_set, todo, new_node);
      new_node = std::get<1>(ret);
      if (!std::get<0>(ret)) {
        return new_node;
      }
      break;
    case kernel::FusionType::ELEMWISE:
      return new_node;
      // -fallthrough to default and return
    case kernel::FusionType::CONVLUTION:
      (void)record->insert(new_node);
    default:
      (void)visited_set->insert(new_node);
      if (new_node != nullptr) {
        (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());
      }
      return new_node;
  }
  // find fusion pattern successor nodes
  ret = FindPredAndSuccEltWiseNodes(MAX_PURE_BUFFER_SUCC_SIZE, manager, visited_set, todo, record, new_node);
  return std::get<1>(ret);
}
}  // namespace

void FusionTypeFusionPass::MatchFusionTypePattern(const session::KernelGraph &kernel_graph,
                                                  FusedNodeRecord *candidate_fusion) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(candidate_fusion);

  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return;
  }
  std::deque<AnfNodePtr> todo;
  todo.push_back(return_node->input(1));
  std::unordered_set<AnfNodePtr> visited_set;

  while (!todo.empty()) {
    auto node = todo.front();
    MS_EXCEPTION_IF_NULL(node);
    todo.pop_front();
    std::unordered_set<AnfNodePtr> record;
    if (visited_set.find(node) != visited_set.end() || fusion_id_allocator->HasFusionIdAttr(node)) {
      continue;
    }
    // Only fuse real cnode
    if (!AnfAlgo::IsRealCNodeKernel(node)) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode != nullptr) {
        (void)todo.insert(todo.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
      }
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // cnode maybe updated
    cnode = FindFusionAnfNode(manager.get(), &visited_set, &record, &todo, cnode);
    if (record.size() >= MIN_PATTERN_SIZE && record.size() <= MAX_PATTERN_SIZE) {
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
    if (record.find(cnode) == record.end()) {
      todo.push_back(cnode);
    }
    // no node matched
    if (record.size() == 0) {
      (void)visited_set.insert(node);
    }
    (void)todo.insert(todo.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  }
}

bool FusionTypeFusionPass::MatchUBFusionPattern(const session::KernelGraph &kernel_graph) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern start...";
  FusedNodeRecord candidate_fusion;
  MatchFusionTypePattern(kernel_graph, &candidate_fusion);
  if (candidate_fusion.empty()) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern Success...";
  return true;
}
}  // namespace opt
}  // namespace mindspore
