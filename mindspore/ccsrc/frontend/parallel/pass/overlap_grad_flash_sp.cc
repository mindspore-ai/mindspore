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

#include "frontend/parallel/pass/overlap_grad_flash_sp.h"
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
std::vector<int64_t> GetCommOrder(std::string comm_order_str) {
  std::istringstream iss(comm_order_str);
  std::string token;
  std::vector<int64_t> res;

  while (std::getline(iss, token, '_')) {
    res.push_back(std::stoi(token));
  }
  return res;
}

std::string NextFlashIndex(std::string flash_index, int64_t sp_num) {
  size_t underscore_pos = flash_index.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "FLASH_INDEX ERROR";
  }

  std::string first_number_str = flash_index.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = flash_index.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) + 1;

  if (second_number > sp_num) {
    return "";
  }

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

void FindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological, std::map<std::string, AnfNodePtr> *grad_fa_map,
                    std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_oml_map, CNodePtr *loss_node) {
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE)) {
      (*loss_node) = node;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive) &&
        !IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      (*grad_fa_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      if (GetValue<std::string>(node->GetPrimalAttr(FLASH_SP_COMM_TYPE)) == FLASH_SP_COMM_QKV) {
        (*grad_recv_qkv_map).insert({flash_index, node});
      } else {
        (*grad_recv_oml_map).insert({flash_index, node});
      }
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      if (GetValue<std::string>(node->GetPrimalAttr(FLASH_SP_COMM_TYPE)) == FLASH_SP_COMM_QKV) {
        (*grad_send_qkv_map).insert({flash_index, node});
      } else {
        (*grad_send_oml_map).insert({flash_index, node});
      }
    }
  }
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const CNodePtr &node) {
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), latter_node, former_node};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(latter_node->abstract()->Clone());
  return depend_node;
}

void GetGradNode(std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                 std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                 std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                 std::map<std::string, AnfNodePtr> *grad_recv_oml_map, CNodePtr *grad_send_qkv_node,
                 CNodePtr *grad_recv_qkv_node, CNodePtr *grad_send_oml_node, CNodePtr *grad_recv_oml_node,
                 const std::string &new_str) {
  if ((*grad_send_qkv_map).find(new_str) != (*grad_send_qkv_map).end()) {
    (*grad_send_qkv_node) = (*grad_send_qkv_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_recv_qkv_map).find(new_str) != (*grad_recv_qkv_map).end()) {
    (*grad_recv_qkv_node) = (*grad_recv_qkv_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_send_oml_map).find(new_str) != (*grad_send_oml_map).end()) {
    (*grad_send_oml_node) = (*grad_send_oml_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_recv_oml_map).find(new_str) != (*grad_recv_oml_map).end()) {
    (*grad_recv_oml_node) = (*grad_recv_oml_map).at(new_str)->cast<CNodePtr>();
  }
}

void GetPreCommNode(const std::string &new_str, int64_t index, std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_oml_map, CNodePtr *pre_grad_comm_node) {
  if (index == 0 && (*grad_send_qkv_map).find(new_str) != (*grad_send_qkv_map).end()) {
    (*pre_grad_comm_node) = (*grad_send_qkv_map).at(new_str)->cast<CNodePtr>();
  } else if (index == 1 && (*grad_recv_qkv_map).find(new_str) != (*grad_recv_qkv_map).end()) {
    (*pre_grad_comm_node) = (*grad_recv_qkv_map).at(new_str)->cast<CNodePtr>();
  } else if (index == 2 && (*grad_send_oml_map).find(new_str) != (*grad_send_oml_map).end()) {
    (*pre_grad_comm_node) = (*grad_send_oml_map).at(new_str)->cast<CNodePtr>();
  } else if (index == 3 && (*grad_recv_oml_map).find(new_str) != (*grad_recv_oml_map).end()) {
    (*pre_grad_comm_node) = (*grad_recv_oml_map).at(new_str)->cast<CNodePtr>();
  }
}
}  // namespace
void OverlapGradFlashSP(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> grad_fa_map, grad_send_qkv_map, grad_recv_qkv_map, grad_send_oml_map,
    grad_recv_oml_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;
  FindTargetNode(&origin_nodes_topological, &grad_fa_map, &grad_send_qkv_map, &grad_recv_qkv_map, &grad_send_oml_map,
                 &grad_recv_oml_map, &loss_node);

  for (auto it = grad_fa_map.begin(); it != grad_fa_map.end(); ++it) {
    if (grad_send_qkv_map.find(it->first) == grad_send_qkv_map.end() &&
        grad_recv_qkv_map.find(it->first) == grad_recv_qkv_map.end() &&
        grad_send_oml_map.find(it->first) == grad_send_oml_map.end() &&
        grad_recv_oml_map.find(it->first) == grad_recv_oml_map.end()) {
      continue;
    }
    CNodePtr grad_fa_node, grad_send_qkv_node, grad_recv_qkv_node, grad_send_oml_node, grad_recv_oml_node;
    grad_fa_node = it->second->cast<CNodePtr>();
    auto sp_num = GetValue<int64_t>(grad_fa_node->GetPrimalAttr("sp_num"));
    GetGradNode(&grad_send_qkv_map, &grad_recv_qkv_map, &grad_send_oml_map, &grad_recv_oml_map, &grad_send_qkv_node,
                &grad_recv_qkv_node, &grad_send_oml_node, &grad_recv_oml_node, it->first);
    std::map<int64_t, CNodePtr> grad_comm_map;
    grad_comm_map.insert({0, grad_send_qkv_node});
    grad_comm_map.insert({1, grad_recv_qkv_node});
    grad_comm_map.insert({2, grad_send_oml_node});
    grad_comm_map.insert({3, grad_recv_oml_node});

    auto new_str = NextFlashIndex(it->first, sp_num);
    CNodePtr pre_grad_comm_node;
    while (new_str != "") {
      if (grad_fa_map.find(new_str) != grad_fa_map.end()) {
        auto pre_grad_fa_node = grad_fa_map.at(new_str)->cast<CNodePtr>();
        if (GetValue<std::string>(pre_grad_fa_node->GetPrimalAttr("comm_order")) != "") {
          auto pre_comm_order = GetValue<std::string>(pre_grad_fa_node->GetPrimalAttr("comm_order"));
          auto pre_comm_order_list = GetCommOrder(pre_comm_order);
          // if (pre_comm_order_list[0] == 0 && grad_send_qkv_map.find(new_str) != grad_send_qkv_map.end()) {
          //   pre_grad_comm_node = grad_send_qkv_map.at(new_str)->cast<CNodePtr>();
          // } else if (pre_comm_order_list[0] == 1 && grad_recv_qkv_map.find(new_str) != grad_recv_qkv_map.end()) {
          //   pre_grad_comm_node = grad_recv_qkv_map.at(new_str)->cast<CNodePtr>();
          // } else if (pre_comm_order_list[0] == 2 && grad_send_oml_map.find(new_str) != grad_send_oml_map.end()) {
          //   pre_grad_comm_node = grad_send_oml_map.at(new_str)->cast<CNodePtr>();
          // } else if (pre_comm_order_list[0] == 3 && grad_recv_oml_map.find(new_str) != grad_recv_oml_map.end()) {
          //   pre_grad_comm_node = grad_recv_oml_map.at(new_str)->cast<CNodePtr>();
          // }
          GetPreCommNode(new_str, pre_comm_order_list[0], &grad_send_qkv_map, &grad_recv_qkv_map, &grad_send_oml_map,
                         &grad_recv_oml_map, &pre_grad_comm_node);
          break;
        }
      }
      new_str = NextFlashIndex(new_str, sp_num);
    }

    auto comm_order = GetValue<std::string>(grad_fa_node->GetPrimalAttr("comm_order"));
    auto comm_order_list = GetCommOrder(comm_order);
    auto grad_first_comm_node = grad_comm_map.at(comm_order_list[comm_order_list.size() - 1]);
    auto grad_last_comm_node = grad_comm_map.at(comm_order_list[0]);
    auto grad_first_comm_node_input = grad_first_comm_node->input(1);
    for (size_t i = 0; i < grad_fa_node->size(); i++) {
      auto grad_fa_input_node = grad_fa_node->input(i);
      if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
        grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, grad_fa_input_node, grad_first_comm_node);
      }
    }
    if (loss_node != nullptr) {
      grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, loss_node, grad_first_comm_node);
    }
    manager->SetEdge(grad_first_comm_node, 1, grad_first_comm_node_input);

    auto grad_fa_node_input = grad_fa_node->input(1);
    for (size_t i = 0; i < grad_first_comm_node->size(); i++) {
      auto grad_first_comm_input_node = grad_first_comm_node->input(i);
      if (grad_first_comm_input_node != nullptr && grad_first_comm_input_node->func_graph() != nullptr) {
        grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_first_comm_input_node, grad_fa_node);
      }
    }
    manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

    if (pre_grad_comm_node != nullptr) {
      auto grad_first_comm_node_input_new = grad_first_comm_node->input(1);
      manager->SetEdge(grad_first_comm_node, 1,
                       CreateDepend(grad_first_comm_node_input_new, pre_grad_comm_node, grad_first_comm_node));
    }

    for (size_t idx = comm_order_list.size() - 1; idx > 0; idx--) {
      auto grad_pre_idx_input = grad_comm_map.at(comm_order_list[idx - 1])->input(1);
      manager->SetEdge(grad_comm_map.at(comm_order_list[idx - 1]), 1,
                       CreateDepend(grad_pre_idx_input, grad_comm_map.at(comm_order_list[idx]),
                                    grad_comm_map.at(comm_order_list[idx - 1])));
    }
    manager->Replace(grad_last_comm_node, CreateDepend(grad_last_comm_node, grad_fa_node, grad_last_comm_node));

    manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_last_comm_node, grad_fa_node));
  }
}
}  // namespace parallel
}  // namespace mindspore
