/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/merge_comm.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kCastType = "cast_type";
constexpr auto kGetItemIndex = "get_item_index";
static Shape GetMakeTupleValue(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() == kSizeThree, "Input size of Reshape is not 3.");
  auto make_tuple = cnode->input(kIndex2);
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  Shape ret;
  for (size_t i = 1; i < make_tuple_cnode->size(); ++i) {
    auto input_node = make_tuple_cnode->input(i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto value_node = GetValueNode(input_node);
    if (value_node != nullptr && value_node->isa<Int64Imm>()) {
      auto shape_ele = GetValue<int64_t>(value_node);
      ret.push_back(shape_ele);
    } else {
      ret.push_back(-1);
    }
  }
  return ret;
}

static bool IsSameTargetDynamicShape(const CNodePtr &reshape_node_a, const CNodePtr &reshape_node_b) {
  MS_EXCEPTION_IF_NULL(reshape_node_a);
  MS_EXCEPTION_IF_NULL(reshape_node_b);
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_node_a->inputs().size() == kSizeThree, "Input size of Reshape is not 3.");
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_node_b->inputs().size() == kSizeThree, "Input size of Reshape is not 3.");
  if (!IsPrimitiveCNode(reshape_node_a->input(kIndex2), prim::kPrimMakeTuple)) {
    MS_LOG(WARNING) << "the dst shape of reshape node a is not make_tuple for dynamic shape";
    return false;
  }

  if (!IsPrimitiveCNode(reshape_node_b->input(kIndex2), prim::kPrimMakeTuple)) {
    MS_LOG(WARNING) << "the dst shape of reshape node b is not make_tuple for dynamic shape";
    return false;
  }

  Shape node_a_shape = GetMakeTupleValue(reshape_node_a);
  Shape node_b_shape = GetMakeTupleValue(reshape_node_b);
  MS_LOG(INFO) << "the node a shape is " << node_a_shape << ", the node b shape is " << node_b_shape;
  if (std::count(node_a_shape.cbegin(), node_a_shape.cend(), -1) > 1) {
    return false;
  }
  if (std::count(node_b_shape.cbegin(), node_b_shape.cend(), -1) > 1) {
    return false;
  }

  return (node_a_shape == node_b_shape);
}

bool IsSameTargetShape(const CNodePtr &reshape_node_a, const CNodePtr &reshape_node_b) {
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_node_a->inputs().size() == kSizeThree, "Input size of Reshape is not 3.");
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_node_b->inputs().size() == kSizeThree, "Input size of Reshape is not 3.");
  if (!reshape_node_a->input(kIndex2)->isa<ValueNode>() || !reshape_node_b->input(kIndex2)->isa<ValueNode>()) {
    return IsSameTargetDynamicShape(reshape_node_a, reshape_node_b);
  }
  MS_EXCEPTION_IF_NULL(reshape_node_a->input(kIndex2)->cast<ValueNodePtr>());
  MS_EXCEPTION_IF_NULL(reshape_node_b->input(kIndex2)->cast<ValueNodePtr>());
  auto value_ptr_a = reshape_node_a->input(kIndex2)->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  auto value_ptr_b = reshape_node_b->input(kIndex2)->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  if (value_ptr_a.size() != value_ptr_b.size()) {
    return false;
  }
  for (size_t i = 0; i < value_ptr_a.size(); i++) {
    int64_t cur_shape_a = GetValue<int64_t>(value_ptr_a.at(i));
    int64_t cur_shape_b = GetValue<int64_t>(value_ptr_b.at(i));
    if (cur_shape_a != cur_shape_b) {
      return false;
    }
  }
  return true;
}

std::unordered_map<CNodePtr, std::vector<CNodePtr>> AllGatherInputMap(const std::vector<AnfNodePtr> &all_nodes) {
  std::unordered_map<CNodePtr, std::vector<CNodePtr>> allgather_input_map;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      continue;
    }
    auto allgather_cnode = node->cast<CNodePtr>();
    auto pre_node = GetInputNodeWithFilter(allgather_cnode->input(kIndex1), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimReshape);
      return std::make_pair(filter, 1);
    });
    if (IsPrimitiveCNode(pre_node, prim::kPrimCast)) {
      MS_EXCEPTION_IF_NULL(pre_node->cast<CNodePtr>()->input(kIndex2));
      auto attr_v = pre_node->cast<CNodePtr>()->input(kIndex2)->cast<ValueNodePtr>();
      if (!IsValueNode<Int64Imm>(attr_v)) {
        continue;
      }
      auto cast_value = attr_v->value();
      allgather_cnode->AddAttr(kCastType, cast_value);
      pre_node = pre_node->cast<CNodePtr>()->input(kIndex1);
      MS_EXCEPTION_IF_NULL(pre_node);
    }
    if (IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
      MS_EXCEPTION_IF_NULL(pre_node->cast<CNodePtr>()->input(kIndex2));
      auto index_value = pre_node->cast<CNodePtr>()->input(kIndex2)->cast<ValueNodePtr>()->value();
      allgather_cnode->AddAttr(kGetItemIndex, index_value);
      pre_node = pre_node->cast<CNodePtr>()->input(kIndex1);
    }
    if (!IsPrimitiveCNode(pre_node)) {
      continue;
    }
    auto pre_cnode = pre_node->cast<CNodePtr>();
    allgather_input_map[pre_cnode].push_back(allgather_cnode);
  }
  return allgather_input_map;
}

bool CheckAttr(const CNodePtr &allgather_cnode1, const CNodePtr &allgather_cnode2, const std::string &attr) {
  if (allgather_cnode1->HasAttr(attr) != allgather_cnode2->HasAttr(attr)) {
    return false;
  }
  if (allgather_cnode1->HasAttr(attr)) {
    auto attr1 = GetValue<int64_t>(allgather_cnode1->GetAttr(attr));
    auto attr2 = GetValue<int64_t>(allgather_cnode2->GetAttr(attr));
    if (attr1 != attr2) {
      return false;
    }
  }
  return true;
}

void MergeAllGather(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  auto allgather_input_map = AllGatherInputMap(all_nodes);
  for (const auto &allgather_pairs : allgather_input_map) {
    if (allgather_pairs.second.size() <= 1) {
      continue;
    }
    auto allgather_list = allgather_pairs.second;
    auto allgather_cnode1 = allgather_list.front();
    auto is_same_allgather =
      std::all_of(allgather_list.begin(), allgather_list.end(), [&allgather_cnode1](const CNodePtr &allgather_cnode2) {
        auto ag1_prim = GetCNodePrimitive(allgather_cnode1);
        MS_EXCEPTION_IF_NULL(ag1_prim);
        auto ag2_prim = GetCNodePrimitive(allgather_cnode2);
        MS_EXCEPTION_IF_NULL(ag2_prim);
        auto group1 = ag1_prim->GetAttr(GROUP);
        auto group2 = ag2_prim->GetAttr(GROUP);
        if (!group1 || !group2) {
          return false;
        }
        if (GetValue<std::string>(group1) != GetValue<std::string>(group2)) {
          return false;
        }
        if (IsPrimitiveCNode(allgather_cnode1->input(kIndex1), prim::kPrimReshape) !=
            IsPrimitiveCNode(allgather_cnode2->input(kIndex1), prim::kPrimReshape)) {
          return false;
        }
        if (IsPrimitiveCNode(allgather_cnode1->input(kIndex1), prim::kPrimReshape) &&
            IsPrimitiveCNode(allgather_cnode2->input(kIndex1), prim::kPrimReshape)) {
          if (!IsSameTargetShape(allgather_cnode1->input(kIndex1)->cast<CNodePtr>(),
                                 allgather_cnode2->input(kIndex1)->cast<CNodePtr>())) {
            return false;
          }
        }
        if (allgather_cnode1->func_graph() != allgather_cnode2->func_graph()) {
          return false;
        }
        if (!CheckAttr(allgather_cnode1, allgather_cnode2, kCastType)) {
          return false;
        }
        if (!CheckAttr(allgather_cnode1, allgather_cnode2, kGetItemIndex)) {
          return false;
        }
        return true;
      });
    if (!is_same_allgather) {
      MS_LOG(INFO) << "allgather nodes share the same input node:" << allgather_pairs.first->DebugString()
                   << " is not equal.";
      continue;
    }
    auto ag0 = allgather_list.front();
    for (const auto &ag : allgather_list) {
      manager->Replace(ag, ag0);
    }
  }
}
}  // namespace

bool MergeComm(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto graph_set = ForwardGraph(root);
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root) || (root->has_flag(MERGE_COMM_RUN_ONCE_ONLY)) || graph_set.size() < 1) {
    return changes;
  }
  FuncGraphManagerPtr manager;
  pipeline::ResourceBasePtr res;
  if (optimizer == nullptr) {
    manager = root->manager();
    res = std::make_shared<pipeline::Resource>();
    res->set_manager(manager);
  } else {
    res = optimizer->resource();
    MS_EXCEPTION_IF_NULL(res);
    manager = res->manager();
  }

  MS_EXCEPTION_IF_NULL(manager);
  CNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  MergeAllGather(all_nodes, manager);
  DumpGraph(root, std::string("merge_comm"));

  // allreduce fusion only run once
  root->set_flag(MERGE_COMM_RUN_ONCE_ONLY, true);
  // Keep all func graph for parallel before save result.
  SetReserved(root);
  res->SetResult(pipeline::kStepParallelGraph, root);
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
