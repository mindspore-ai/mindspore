/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/merge_send_recv.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
std::unordered_map<AnfNodePtr, std::vector<CNodePtr>> ReceiveInputMap(const std::vector<AnfNodePtr> &all_nodes) {
  std::unordered_map<AnfNodePtr, std::vector<CNodePtr>> recv_input_map;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto recv_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(recv_cnode);
    auto send_node = GetInputNodeWithFilter(recv_cnode->input(kIndex1), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
      return std::make_pair(filter, 1);
    });
    if (!IsPrimitiveCNode(send_node, prim::kPrimSend)) {
      continue;
    }
    auto send_cnode = send_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(send_cnode);
    auto pre_node = GetInputNodeWithFilter(send_cnode->input(kIndex1), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
      return std::make_pair(filter, 1);
    });
    recv_input_map[pre_node].push_back(recv_cnode);
  }
  return recv_input_map;
}

template <typename S>
bool CheckPrimAttr(const PrimitivePtr &prim1, const PrimitivePtr &prim2, const std::string &attr) {
  MS_EXCEPTION_IF_NULL(prim1);
  MS_EXCEPTION_IF_NULL(prim2);
  auto attr_v1 = prim1->GetAttr(attr);
  auto attr_v2 = prim2->GetAttr(attr);
  if (!attr_v1 || !attr_v2) {
    return false;
  }
  return GetValue<S>(attr_v1) == GetValue<S>(attr_v2);
}

template <typename S>
bool CheckPrimalAttr(const CNodePtr &cnode1, const CNodePtr &cnode2, const std::string &attr) {
  MS_EXCEPTION_IF_NULL(cnode1);
  MS_EXCEPTION_IF_NULL(cnode2);
  auto attr_v1 = cnode1->GetPrimalAttr(attr);
  auto attr_v2 = cnode2->GetPrimalAttr(attr);
  if (!attr_v1 && !attr_v2) {
    return true;
  }
  if (!attr_v1 || !attr_v2) {
    return false;
  }
  return GetValue<S>(attr_v1) == GetValue<S>(attr_v2);
}

bool CheckSendReceive(const CNodePtr &cnode1, const CNodePtr &cnode2) {
  auto prim1 = GetCNodePrimitive(cnode1);
  auto prim2 = GetCNodePrimitive(cnode2);
  MS_EXCEPTION_IF_NULL(cnode1);
  MS_EXCEPTION_IF_NULL(cnode2);
  if (cnode1->func_graph() != cnode2->func_graph()) {
    return false;
  }
  if (!CheckPrimAttr<std::string>(prim1, prim2, GROUP)) {
    return false;
  }

  if (!CheckPrimAttr<int64_t>(prim1, prim2, SR_TAG)) {
    return false;
  }

  if (IsPrimitiveCNode(cnode1, prim::kPrimReceive) && !CheckPrimAttr<int64_t>(prim1, prim2, SRC_RANK)) {
    return false;
  }
  if (IsPrimitiveCNode(cnode1, prim::kPrimSend) && !CheckPrimAttr<int64_t>(prim1, prim2, DEST_RANK)) {
    return false;
  }
  if (!CheckPrimalAttr<int64_t>(cnode1, cnode2, STAGE)) {
    return false;
  }
  if (!CheckPrimalAttr<int64_t>(cnode1, cnode2, CHUNK)) {
    return false;
  }
  return true;
}

void MergeSR(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  const auto &recv_input_map = ReceiveInputMap(all_nodes);
  for (const auto &recv_pairs : recv_input_map) {
    if (recv_pairs.second.size() <= 1) {
      continue;
    }
    auto recv_list = recv_pairs.second;
    auto recv_cnode1 = recv_list.front();
    auto is_same_recv = std::all_of(recv_list.begin(), recv_list.end(), [&recv_cnode1](const CNodePtr &recv_cnode2) {
      if (!CheckSendReceive(recv_cnode1, recv_cnode2)) {
        return false;
      }
      auto send_node1 = GetInputNodeWithFilter(recv_cnode1->input(kIndex1), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
        return std::make_pair(filter, 1);
      });
      if (!IsPrimitiveCNode(send_node1, prim::kPrimSend)) {
        return false;
      }

      auto send_node2 = GetInputNodeWithFilter(recv_cnode2->input(kIndex1), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
        return std::make_pair(filter, 1);
      });
      if (!IsPrimitiveCNode(send_node2, prim::kPrimSend)) {
        return false;
      }

      if (!CheckSendReceive(send_node1->cast<CNodePtr>(), send_node2->cast<CNodePtr>())) {
        return false;
      }

      return true;
    });
    if (!is_same_recv) {
      MS_LOG(INFO) << "recv nodes share the same input node:" << recv_pairs.first->DebugString() << " is not equal.";
      continue;
    }
    auto recv0 = recv_list.front();
    for (const auto &recv : recv_list) {
      (void)manager->Replace(recv, recv0);
    }
  }
}
}  // namespace

bool MergeSendReceive(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto graph_set = ForwardGraph(root);
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root) || (root->has_flag(MERGE_SEND_RECEIVE_RUN_ONCE_ONLY)) || graph_set.size() < 1) {
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
  const auto &all_nodes = TopoSort(ret, SuccDeeperSimple);
  MergeSR(all_nodes, manager);
  DumpGraph(root, std::string("merge_send_receive"));

  root->set_flag(MERGE_SEND_RECEIVE_RUN_ONCE_ONLY, true);
  SetReserved(root);
  res->SetResult(pipeline::kStepParallelGraph, root);
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
