/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/pass/eliminate_redundant_op.h"
#include <memory>
#include <utility>
#include <unordered_map>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"
#include "base/core_ops.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
using KernelWithIndex = std::pair<CNodePtr, size_t>;
namespace {
CNodePtr GetRealPrevCNode(const AnfNodePtr &node, size_t index, std::vector<KernelWithIndex> *pass_vector) {
  MS_EXCEPTION_IF_NULL(pass_vector);
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::IsRealCNodeKernel(cnode)) {
    pass_vector->push_back(make_pair(cnode, IntToSize(1)));
    return cnode;
  }

  auto input0 = cnode->input(0);
  MS_EXCEPTION_IF_NULL(input0);
  if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
    auto temp_node = cnode->input(index + IntToSize(1));
    MS_EXCEPTION_IF_NULL(temp_node);
    pass_vector->push_back(make_pair(cnode, index + IntToSize(1)));
    return GetRealPrevCNode(temp_node, 0, pass_vector);
  } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
    auto input2 = cnode->input(2);
    MS_EXCEPTION_IF_NULL(input2);
    auto value_node = input2->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto item_idx = GetValue<int64_t>(value_node->value());
    pass_vector->push_back(make_pair(cnode, IntToSize(1)));
    return GetRealPrevCNode(cnode->input(1), LongToSize(item_idx), pass_vector);
  } else if (IsPrimitive(input0, prim::kPrimDepend) || IsPrimitive(input0, prim::kPrimControlDepend)) {
    pass_vector->push_back(make_pair(cnode, IntToSize(1)));
    return GetRealPrevCNode(cnode->input(1), 0, pass_vector);
  } else if (IsPrimitive(input0, prim::kPrimUpdateState)) {
    pass_vector->push_back(make_pair(cnode, IntToSize(kUpdateStateRealInput)));
    return GetRealPrevCNode(cnode->input(kUpdateStateRealInput), 0, pass_vector);
  } else {
    return nullptr;
  }
}

bool TransOpEliminateCondition(const CNodePtr &, const CNodePtr &) { return true; }

bool CastEliminateCondition(const CNodePtr &node1, const CNodePtr &node2) {
  return HasSymmetricalKernelInfo(node1, node2);
}

bool TransDataOpEliminateCondition(const CNodePtr &node1, const CNodePtr &node2) {
  return AnfAlgo::GetInputFormat(node1, 0) == AnfAlgo::GetOutputFormat(node2, 0) &&
         AnfAlgo::GetOutputFormat(node1, 0) == AnfAlgo::GetInputFormat(node2, 0) &&
         kernel::IsSameShape(AnfAlgo::GetInputDeviceShape(node2, 0), AnfAlgo::GetOutputDeviceShape(node1, 0));
}

const AnfNodePtr ProcessMatchedNodes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const CNodePtr &prev_cnode,
                                     std::vector<KernelWithIndex> *pass_vector) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(pass_vector);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  bool has_depend_node = false;
  bool has_node_used_more_than_once = false;
  auto &users = manager->node_users();

  auto pass_size = pass_vector->size();
  for (size_t idx = 1; idx <= pass_size - 1; ++idx) {
    auto nd = (*pass_vector)[idx].first;
    if (AnfAlgo::CheckPrimitiveType(nd, prim::kPrimDepend) ||
        AnfAlgo::CheckPrimitiveType(nd, prim::kPrimControlDepend)) {
      has_depend_node = true;
    }
    if (users[nd].size() >= 2) {
      has_node_used_more_than_once = true;
    }
  }

  // when no depend node and no node used more than once, no need to rebuild the pass nodes
  if (!has_depend_node) {
    return prev_cnode->input(1);
  } else if (!has_node_used_more_than_once) {
    (void)manager->Replace(prev_cnode, prev_cnode->input(1));
    return cnode->input(1);
  } else {  // rebuild the pass nodes
    for (size_t idx = pass_size - 2; idx > 0; --idx) {
      auto new_node = func_graph->NewCNode((*pass_vector)[idx].first->inputs());
      if (idx == pass_size - 2) {
        new_node->set_input((*pass_vector)[idx].second,
                            (*pass_vector)[idx + 1].first->input((*pass_vector)[idx + 1].second));
      } else {
        new_node->set_input((*pass_vector)[idx].second, (*pass_vector)[idx + 1].first);
      }
      (*pass_vector)[idx].first = new_node;
    }
    return (*pass_vector)[1].first;
  }
}
}  // namespace

void EliminateRedundantOp::Init() {
  (void)redundant_process_map_.emplace(std::pair<std::string, RedundantOpPair>(
    kFour2FiveOpName, std::pair<std::string, ConditionFunc>(kFive2FourOpName, TransOpEliminateCondition)));
  (void)redundant_process_map_.emplace(std::pair<std::string, RedundantOpPair>(
    kFive2FourOpName, std::pair<std::string, ConditionFunc>(kFour2FiveOpName, TransOpEliminateCondition)));
  (void)redundant_process_map_.emplace(std::pair<std::string, RedundantOpPair>(
    prim::kPrimCast->name(), std::pair<std::string, ConditionFunc>(prim::kPrimCast->name(), CastEliminateCondition)));
  (void)redundant_process_map_.emplace(std::pair<std::string, RedundantOpPair>(
    kTransDataOpName, std::pair<std::string, ConditionFunc>(kTransDataOpName, TransDataOpEliminateCondition)));
}

const AnfNodePtr EliminateRedundantOp::DoEliminate(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const {
  // match the first name
  auto name1 = AnfAlgo::GetCNodeName(cnode);
  auto it = redundant_process_map_.find(name1);
  if (it == redundant_process_map_.end()) {
    return nullptr;
  }
  std::vector<KernelWithIndex> pass_vector;
  pass_vector.push_back(make_pair(cnode, 1));
  auto prev_cnode = GetRealPrevCNode(cnode->input(1), 0, &pass_vector);
  if (prev_cnode == nullptr) {
    return nullptr;
  }
  // match the second name
  auto name2 = AnfAlgo::GetCNodeName(prev_cnode);
  if (name2 != it->second.first) {
    return nullptr;
  }
  // match condition
  auto condition_func = it->second.second;
  if (condition_func == nullptr) {
    return nullptr;
  }
  if (!condition_func(cnode, prev_cnode)) {
    return nullptr;
  }

  return ProcessMatchedNodes(func_graph, cnode, prev_cnode, &pass_vector);
}

const AnfNodePtr EliminateRedundantOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || func_graph == nullptr) {
    return nullptr;
  }

  if (AnfAlgo::IsGraphKernel(node)) {
    // do eliminate for ops in graph kernel.
    auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    auto mng = sub_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    std::vector<AnfNodePtr> todo;
    kernel::GetValidKernelNodes(sub_graph, &todo);
    for (auto &t : todo) {
      CNodePtr t_cnode = t->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(t_cnode);
      auto t_new_node = DoEliminate(sub_graph, t_cnode);
      if (t_new_node != nullptr && t_new_node != t) {
        (void)mng->Replace(t, t_new_node);
      }
    }
    return node;
  }
  // do eliminate for single op.
  return DoEliminate(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
