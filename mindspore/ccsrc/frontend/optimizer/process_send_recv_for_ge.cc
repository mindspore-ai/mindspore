/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/process_send_recv_for_ge.h"

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include "include/common/utils/anfalgo.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "ops/other_ops.h"
#include "ops/framework_ops.h"

namespace mindspore::opt {
namespace {
AnfNodePtr last_need_depend = nullptr;
bool IsSendRecvOps(const AnfNodePtr &node) {
  static const PrimitiveSet kSendRecvOpsPrim = {prim::kPrimSend, prim::kPrimReceive};
  return IsOneOfPrimitiveCNode(node, kSendRecvOpsPrim);
}

bool IsCommOps(const AnfNodePtr &node) {
  static const PrimitiveSet kCommunicationOpsPrim = {prim::kPrimSend,
                                                     prim::kPrimReceive,
                                                     prim::kPrimAllReduce,
                                                     prim::kPrimAllGather,
                                                     prim::kPrimReduceScatter,
                                                     prim::kPrimAllToAll,
                                                     prim::kPrimAllSwap,
                                                     prim::kPrimAllToAllv,
                                                     prim::kPrimNeighborExchange,
                                                     prim::kPrimNeighborExchangeV2,
                                                     prim::kPrimNeighborExchangeV2Grad};
  return IsOneOfPrimitiveCNode(node, kCommunicationOpsPrim);
}

std::tuple<FuncGraphPtr, CNodePtr> CreateNewCNode(const FuncGraphManagerPtr &manager, const CNodePtr &old_node,
                                                  bool is_send) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> params;

  auto old_prim = GetValueNode<PrimitivePtr>(old_node->input(0));
  auto prim_name = old_prim->name();
  params.push_back(NewValueNode(std::make_shared<Primitive>(prim_name)));

  for (size_t i = 1; i < old_node->inputs().size(); i++) {
    auto param = fg->add_parameter();
    params.push_back(param);
    param->set_abstract(old_node->input(i)->abstract());
  }

  CNodePtr new_node = fg->NewCNode(params);
  new_node->set_abstract(old_node->abstract());

  std::ostringstream ss;
  if (IsSendRecvOps(old_node)) {
    ss << old_prim->name() << old_prim->GetAttr(parallel::SR_TAG)->ToString();
  } else {
    ss << old_prim->name();
  }
  fg->debug_info()->set_name(ss.str());

  for (auto &kv : old_prim->attrs()) {
    common::AnfAlgo::SetNodeAttr(kv.first, kv.second, new_node);
  }
  new_node->set_primal_attrs(old_node->primal_attrs());
  return {fg, new_node};
}

void ProcessSend(const FuncGraphPtr &graph, const CNodePtr &node) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto [fg, new_send] = CreateNewCNode(manager, node, true);

  auto value_node = NewValueNode(MakeValue<int32_t>(1));
  value_node->set_abstract(value_node->value()->ToAbstract());
  auto value_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1));
  auto depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), value_node, new_send});
  depend->set_abstract(value_abs);
  fg->set_output(depend);

  std::vector<AnfNodePtr> call_params;
  call_params.push_back(NewValueNode(fg));
  for (size_t i = 1; i < node->inputs().size(); i++) {
    if (last_need_depend != nullptr) {
      auto new_depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), node->input(i), last_need_depend});
      new_depend->set_abstract(node->input(i)->abstract());
      call_params.push_back(new_depend);
    } else {
      call_params.push_back(node->input(i));
    }
  }
  auto call = graph->NewCNode(call_params);
  call->set_abstract(value_abs);
  last_need_depend = call;

  manager->AddFuncGraph(fg);
  manager->Replace(node, call);
}

void ProcessRecv(const FuncGraphPtr &graph, const CNodePtr &node) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto [fg, new_node] = CreateNewCNode(manager, node, false);
  fg->set_output(new_node);

  std::vector<AnfNodePtr> call_params;
  call_params.push_back(NewValueNode(fg));
  for (size_t i = 1; i < node->inputs().size(); i++) {
    if (last_need_depend != nullptr) {
      auto new_depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), node->input(i), last_need_depend});
      new_depend->set_abstract(node->input(i)->abstract());
      call_params.push_back(new_depend);
    } else {
      call_params.push_back(node->input(i));
    }
  }
  auto call = graph->NewCNode(call_params);
  call->set_abstract(node->abstract());
  last_need_depend = call;

  manager->AddFuncGraph(fg);
  manager->Replace(node, call);
}

void ProcessClearFloatStatus(const FuncGraphPtr &graph, const CNodePtr &node) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto value_node = NewValueNode(MakeValue<int32_t>(1));
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  value_node->set_abstract(value_node->value()->ToAbstract());
  auto value_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1));
  MS_EXCEPTION_IF_NULL(value_abs);
  auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), value_node, node});
  MS_EXCEPTION_IF_NULL(depend);
  depend->set_abstract(value_abs);
  auto tensor_move = graph->NewCNode({NewValueNode(prim::kPrimTensorMove), depend});
  MS_EXCEPTION_IF_NULL(tensor_move);
  tensor_move->set_abstract(value_abs);
  manager->Replace(node, tensor_move);
}
}  // namespace
void ProcessSendRecvForGE(const FuncGraphPtr &graph) {
  static const bool is_enable_ge = (common::GetEnv("MS_ENABLE_GE") == "1");
  static const bool is_cell_reuse =
    (common::GetEnv("MS_DEV_CELL_REUSE") == "1" || common::GetEnv("MS_DEV_CELL_REUSE") == "2");
  if (!is_enable_ge || !is_cell_reuse) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->func_graph() != graph) {
      continue;
    }
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      ProcessSend(graph, node->cast<CNodePtr>());
    } else if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      ProcessRecv(graph, node->cast<CNodePtr>());
    } else if (IsPrimitiveCNode(node, prim::kPrimNPUClearFloatStatusV2)) {
      ProcessClearFloatStatus(graph, node->cast<CNodePtr>());
    } else if (IsCommOps(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (last_need_depend != nullptr && cnode->inputs().size() > 1) {
        auto before_input = node->cast<CNodePtr>()->input(1);
        auto new_depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), before_input, last_need_depend});
        new_depend->set_abstract(before_input->abstract());
        node->cast<CNodePtr>()->set_input(1, new_depend);
      }
      last_need_depend = node;
    } else if (IsValueNode<FuncGraph>(node->cast<CNodePtr>()->input(0))) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (last_need_depend != nullptr) {
        for (size_t i = 1; i < cnode->inputs().size(); i++) {
          auto before_input = node->cast<CNodePtr>()->input(i);
          if (!utils::isa<CNodePtr>(before_input)) {
            // Prevents cutting out a subgraph with only a Depend node.
            continue;
          }
          auto new_depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), before_input, last_need_depend});
          new_depend->set_abstract(before_input->abstract());
          node->cast<CNodePtr>()->set_input(i, new_depend);
        }
      }
      last_need_depend = node;
    }
  }
}
}  // namespace mindspore::opt
