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

#include "frontend/parallel/pass/overlap_grad_ring_attention.h"
#include <map>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <queue>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace parallel {
namespace {

std::string GetNewStr(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) + 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

void FindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological, std::map<std::string, AnfNodePtr> *grad_fa_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_map, std::map<std::string, AnfNodePtr> *grad_send_map,
                    CNodePtr *loss_node) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimReceive) && !IsPrimitiveCNode(node, prim::kPrimSend) &&
        !IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_fa_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_recv_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_send_map).insert({flash_index, node});
    }
  }
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const CNodePtr &node) {
  if (former_node == nullptr) {
    return latter_node->cast<CNodePtr>();
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), latter_node, former_node};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(latter_node->abstract()->Clone());
  return depend_node;
}

void GetPreNode(CNodePtr *pre_grad_recv_node, CNodePtr *pre_grad_send_node, const std::string &new_str,
                std::map<std::string, AnfNodePtr> *grad_recv_map, std::map<std::string, AnfNodePtr> *grad_send_map) {
  if ((*grad_recv_map).find(new_str) != (*grad_recv_map).end() &&
      (*grad_send_map).find(new_str) != (*grad_send_map).end()) {
    (*pre_grad_recv_node) = (*grad_recv_map).at(new_str)->cast<CNodePtr>();
    (*pre_grad_send_node) = (*grad_send_map).at(new_str)->cast<CNodePtr>();
  }
}
}  // namespace
void OverlapGradRingAttention(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> grad_fa_map;
  std::map<std::string, AnfNodePtr> grad_send_map;
  std::map<std::string, AnfNodePtr> grad_recv_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;

  FindTargetNode(&origin_nodes_topological, &grad_fa_map, &grad_recv_map, &grad_send_map, &loss_node);
  for (auto it = grad_send_map.begin(); it != grad_send_map.end(); ++it) {
    // if (grad_fa_map.find(it->first) == grad_fa_map.end() || grad_recv_map.find(it->first) == grad_recv_map.end()) {
    //   continue;
    // }
    auto grad_fa_node = grad_fa_map.at(it->first)->cast<CNodePtr>();
    auto grad_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
    auto grad_send_node = it->second->cast<CNodePtr>();

    CNodePtr pre_grad_recv_node;
    CNodePtr pre_grad_send_node;
    auto new_str = GetNewStr(it->first);
    GetPreNode(&pre_grad_recv_node, &pre_grad_send_node, new_str, &grad_recv_map, &grad_send_map);

    auto grad_recv_pos = GetValue<int64_t>(grad_recv_node->GetPrimalAttr(RING_ATTENTION_POS));
    // auto grad_send_pos = GetValue<int64_t>(grad_send_node->GetPrimalAttr(RING_ATTENTION_POS));
    if (grad_recv_pos % kIndex2 == 0) {
      auto grad_send_input = grad_send_node->input(1);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_send_input = CreateDepend(grad_send_input, grad_fa_input_node, grad_send_node);
        }
      }
      grad_send_input = CreateDepend(grad_send_input, loss_node, grad_send_node);
      manager->SetEdge(grad_send_node, 1, grad_send_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_send_node->size(); i++) {
        auto grad_send_input_node = grad_send_node->input(i);
        if (grad_send_input_node != nullptr && grad_send_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_send_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_recv_node != nullptr) {
        auto grad_send_input_new = grad_send_node->input(1);
        manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input_new, pre_grad_recv_node, grad_send_node));
      }

      auto grad_recv_input = grad_recv_node->input(1);
      manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input, grad_send_node, grad_recv_node));

      manager->Replace(grad_recv_node, CreateDepend(grad_recv_node, grad_fa_node, grad_recv_node));

      std::vector<AnfNodePtr> depend3_inputs{NewValueNode(prim::kPrimDepend), grad_fa_node, grad_recv_node};
      auto depend_node3 = grad_fa_node->func_graph()->NewCNode(depend3_inputs);
      MS_EXCEPTION_IF_NULL(depend_node3);
      depend_node3->set_abstract(grad_fa_node->abstract()->Clone());
      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_recv_node, grad_fa_node));
    } else {
      auto grad_recv_input = grad_recv_node->input(1);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_recv_input = CreateDepend(grad_recv_input, grad_fa_input_node, grad_recv_node);
        }
      }
      grad_recv_input = CreateDepend(grad_recv_input, loss_node, grad_recv_node);
      manager->SetEdge(grad_recv_node, 1, grad_recv_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_recv_node->size(); i++) {
        auto grad_recv_input_node = grad_recv_node->input(i);
        if (grad_recv_input_node != nullptr && grad_recv_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_recv_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_send_node != nullptr) {
        auto grad_recv_input_new = grad_recv_node->input(1);
        manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input_new, pre_grad_send_node, grad_recv_node));
      }

      auto grad_send_input = grad_send_node->input(1);
      manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input, grad_recv_node, grad_send_node));

      manager->Replace(grad_send_node, CreateDepend(grad_send_node, grad_fa_node, grad_send_node));

      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_send_node, grad_fa_node));
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
