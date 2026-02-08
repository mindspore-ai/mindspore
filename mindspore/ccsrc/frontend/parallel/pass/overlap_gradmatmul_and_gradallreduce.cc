/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/overlap_gradmatmul_and_gradallreduce.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>
#include <sstream>
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/utils.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace parallel {
namespace {
PrimitiveSet has_dw_prim_set = {prim::kPrimMatMul, prim::kPrimBatchMatMul, prim::kPrimMatMulExt,
                                prim::kPrimBatchMatMulExt, prim::kPrimGroupedMatmul};
std::unordered_map<std::string, size_t> match_prim_level = {{prim::kPrimMatMul->name(), 0},
                                                            {prim::kPrimMatMulExt->name(), 0},
                                                            {prim::kPrimBatchMatMul->name(), 1},
                                                            {prim::kPrimBatchMatMulExt->name(), 1},
                                                            {prim::kPrimGroupedMatmul->name(), 2}};
const size_t count_ten = 10;

std::set<std::string> split(const std::string &str, char delimiter) {
  std::set<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.insert(token);
  }

  return tokens;
}

void ExtractForwardMatMul(const std::vector<CNodePtr> &origin_nodes_topological,
                          std::vector<std::string> *forward_matmul_unique_id_list) {
  for (auto &node : origin_nodes_topological) {
    if (!IsForwardNode(node) || !IsOneOfPrimitiveCNode(node, has_dw_prim_set)) {
      continue;
    }
    auto matmul_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul_cnode);
    if (!matmul_cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      continue;
    }
    auto matmul_unique_id = GetValue<std::string>(matmul_cnode->GetPrimalAttr(kPrimalAttrUniqueId));
    (*forward_matmul_unique_id_list).push_back(matmul_unique_id);
  }
}

int64_t GetMatMulFlops(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimGroupedMatmul)) {
    // GroupedMatmul cannot calculate flops, return max int
    return INT64_MAX;
  }

  auto full_a_shape = cnode->input(kIndex1)->abstract()->GetShapeTrack()->GetShapeVector();
  auto full_b_shape = cnode->input(kIndex2)->abstract()->GetShapeTrack()->GetShapeVector();
  auto transpose_b = false;
  if (kIndex4 < cnode->inputs().size()) {
    // MatmulExt dose not has transpose inputs
    transpose_b = GetValue<bool>(GetValueNode(cnode->input(kIndex4)));
  }
  auto a_dim_index = full_a_shape.size() - kIndex2;
  auto b_dim_index = full_a_shape.size() - 1;
  int64_t flops = 1;
  auto pre_shape = full_a_shape.size() > full_b_shape.size() ? full_a_shape : full_b_shape;
  for (size_t i = 0; i < pre_shape.size() - kIndex2; i++) {
    flops *= pre_shape[i];
  }
  // [N, C]*[C, M]
  auto M = transpose_b ? *(full_b_shape.end() - 2) : *(full_b_shape.end() - 1);
  const int64_t coefficient = 2;
  flops *= coefficient * full_a_shape[a_dim_index] * full_a_shape[b_dim_index] * M;
  return flops;
}

std::vector<CNodePtr> GetCommInputMatMulNode(const AnfNodePtr &node,
                                             const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                                             size_t count_num, size_t loop_max = 150) {
  std::vector<CNodePtr> result;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  size_t loop = 0;
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    if (result.size() == count_num || loop >= loop_max) {
      return result;
    }
    if (!queue_end->isa<CNode>()) {
      continue;
    }
    auto cnode_queue_end = queue_end->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_queue_end);
    if (cnode_queue_end->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (IsOneOfPrimitiveCNode(cnode_queue_end, has_dw_prim_set) &&
        backward_matmul_dx_dw_map.count(cnode_queue_end->cast<CNodePtr>()) > 0) {
      result.push_back(queue_end->cast<CNodePtr>());
    }
    auto input_size = cnode_queue_end->size();
    if (IsPrimitiveCNode(cnode_queue_end, prim::kPrimDepend)) {
      input_size = 2;
    }
    for (size_t i = 1; i < input_size; ++i) {
      anf_queue.push(cnode_queue_end->input(i));
    }
    loop++;
  }
  return result;
}

void InsertDepend(const FuncGraphManagerPtr &manager, const CNodePtr &comm_i1, const CNodePtr &matmul_i) {
  // In some cases, GroupedMatmul requires its first input node to be a TransposeView, so insert the depend node
  // according to its second input.
  int64_t matmul_input_index = IsPrimitiveCNode(matmul_i, prim::kPrimGroupedMatmul) ? kIndex2 : kIndex1;
  auto comm_i1_input = comm_i1->input(kIndex1);
  auto matmul_i_input = matmul_i->input(matmul_input_index);
  std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), matmul_i_input, comm_i1_input};
  auto depend_node1 = matmul_i_input->func_graph()->NewCNode(depend1_inputs);
  MS_EXCEPTION_IF_NULL(depend_node1);
  depend_node1->set_abstract(matmul_i_input->abstract()->Clone());
  depend_node1->AddAttr("matmul_grad_depend1", MakeValue(true));
  depend_node1->AddAttr(kAttrCommInputDepend, MakeValue(true));
  manager->SetEdge(matmul_i, matmul_input_index, depend_node1);

  auto comm_i1_output = manager->node_users()[comm_i1].front().first;
  std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), comm_i1, matmul_i};
  auto depend_node2 = comm_i1->func_graph()->NewCNode(depend2_inputs);
  MS_EXCEPTION_IF_NULL(depend_node2);
  depend_node2->set_abstract(comm_i1->abstract()->Clone());
  depend_node2->AddAttr("matmul_grad_depend2", MakeValue(true));
  auto comm_id = comm_i1->UniqueId();
  comm_i1->AddAttr(GRAD_OVERLAP_MATMUL, MakeValue(comm_id));
  matmul_i->AddAttr(GRAD_OVERLAP_MATMUL, MakeValue(comm_id));
  manager->SetEdge(comm_i1_output, manager->node_users()[comm_i1].front().second, depend_node2);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->GetBackend() != "GE") {
    std::vector<AnfNodePtr> depend3_inputs{NewValueNode(prim::kPrimDepend), matmul_i_input, comm_i1};
    auto depend_node3 = matmul_i_input->func_graph()->NewCNode(depend3_inputs);
    MS_EXCEPTION_IF_NULL(depend_node3);
    depend_node3->set_abstract(matmul_i_input->abstract()->Clone());
    depend_node3->AddAttr("matmul_grad_depend3", MakeValue(true));
    depend_node3->AddAttr(kAttrCommInputDepend, MakeValue(true));
    manager->SetEdge(matmul_i, matmul_input_index, depend_node3);
  }
}

void OverLapGradMatMul(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                       const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                       const std::vector<std::string> &forward_matmul_unique_id_list) {
  std::set<CNodePtr> matched_matmul_list;
  CNodePtrList communicate_cnode_list;
  auto grad_comm_value = MsContext::GetInstance()->get_param<std::string>(MS_CTX_GRAD_COMM_OVERLAP);
  auto grad_comm_name_list = split(grad_comm_value, ',');
  for (const auto &node : origin_nodes_topological) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (!IsSomePrimitiveList(cnode, grad_comm_name_list)) {
      continue;
    }
    if (IsForwardNode(node) || node->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (!node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    if (node->HasAttr(INTERLEAVED_OVERLAP_MATMUL)) {
      continue;
    }
    communicate_cnode_list.push_back(node);
  }

  // Priority is given to masking AlltoAll communications
  std::stable_sort(communicate_cnode_list.begin(), communicate_cnode_list.end(),
                   [](const CNodePtr &cnode1, const CNodePtr &cnode2) {
                     return IsOneOfPrimitiveCNode(cnode1, {prim::kPrimAlltoAll, prim::kPrimAlltoAllV}) ||
                            !IsOneOfPrimitiveCNode(cnode2, {prim::kPrimAlltoAll, prim::kPrimAlltoAllV});
                   });
  for (const auto &communicate_cnode : communicate_cnode_list) {
    auto input_matmul_dx_nodes = GetCommInputMatMulNode(communicate_cnode, backward_matmul_dx_dw_map, count_ten);
    if (input_matmul_dx_nodes.empty()) {
      MS_LOG(DEBUG) << "comm node:" << communicate_cnode->fullname_with_scope()
                    << ", unique_id:" << AnfNodeInfo(communicate_cnode) << " cannot find input matmuls";
      continue;
    }
    std::sort(
      input_matmul_dx_nodes.begin(), input_matmul_dx_nodes.end(), [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
        auto cnode1_prim_name = GetCNodePrimitive(cnode1)->name();
        auto cnode2_prim_name = GetCNodePrimitive(cnode2)->name();
        if (match_prim_level[cnode1_prim_name] != match_prim_level[cnode2_prim_name]) {
          return match_prim_level[cnode1_prim_name] > match_prim_level[cnode2_prim_name];
        }
        auto flops1 = GetMatMulFlops(cnode1);
        auto flops2 = GetMatMulFlops(cnode2);
        if (flops1 != flops2) {
          return flops1 > flops2;
        }

        auto id1 = GetValue<std::string>(cnode1->GetPrimalAttr(kPrimalAttrForwardUniqueId));
        auto id2 = GetValue<std::string>(cnode2->GetPrimalAttr(kPrimalAttrForwardUniqueId));
        size_t index1 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id1) -
                        forward_matmul_unique_id_list.begin();
        size_t index2 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id2) -
                        forward_matmul_unique_id_list.begin();
        return index1 < index2;
      });
    for (const auto &matmul : input_matmul_dx_nodes) {
      if (matched_matmul_list.count(backward_matmul_dx_dw_map.at(matmul)) > 0) {
        continue;
      }
      // insert depend
      MS_LOG(DEBUG) << "insert depend for comm node:" << communicate_cnode->fullname_with_scope()
                    << ", unique id:" << AnfNodeInfo(communicate_cnode) << " and "
                    << backward_matmul_dx_dw_map.at(matmul)->fullname_with_scope()
                    << ", unique id:" << AnfNodeInfo(backward_matmul_dx_dw_map.at(matmul));
      InsertDepend(manager, communicate_cnode, backward_matmul_dx_dw_map.at(matmul));
      matched_matmul_list.insert(backward_matmul_dx_dw_map.at(matmul));
      break;
    }
  }
}

void DoOverLapWay(const FuncGraphManagerPtr &manager, const FuncGraphPtr &forward_graph,
                  const FuncGraphPtr &backward_graph) {
  const auto &forward_orders = forward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> forward_origin_nodes_topological(forward_orders.cbegin(), forward_orders.cend());
  const auto &backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  std::vector<std::string> forward_matmul_unique_id_list;
  ExtractForwardMatMul(forward_origin_nodes_topological, &forward_matmul_unique_id_list);
  std::unordered_map<CNodePtr, CNodePtr> backward_matmul_dx_dw_map;
  ExtractBackwardMatMul(backward_origin_nodes_topological, &backward_matmul_dx_dw_map);
  ExtendDxDwMap(backward_origin_nodes_topological, &backward_matmul_dx_dw_map);
  OverLapGradMatMul(manager, backward_origin_nodes_topological, backward_matmul_dx_dw_map,
                    forward_matmul_unique_id_list);
}
}  // namespace

void OverlapGradMatmulAndGradAllreduce(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = !ms_context->get_param<std::string>(MS_CTX_GRAD_COMM_OVERLAP).empty();
  if (!is_enable) {
    return;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  circle_handler::SetAttrToDepend(graph);
  const auto cell_reuse = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    for (const auto &each_graph : manager->func_graphs()) {
      if (IsCellReuseForwardGraph(each_graph)) {
        auto forward_graph = each_graph;
        auto backward_graph = GetCellReuseBackwardGraph(forward_graph);
        if (backward_graph == nullptr) {
          MS_LOG(INFO) << "Failed to find backward cell reuse graph, skip pass 'overlap_gradmatmul_and_gradallreduce'.";
          continue;
        }
        DoOverLapWay(manager, forward_graph, backward_graph);
      }
    }
  } else {
    DoOverLapWay(manager, graph, graph);
  }
  circle_handler::DetectAndRevertGraphCircle(graph, manager, "OverlapGradMatmulAndGradAllreduce",
                                             "matmul_grad_comm_overlap");
}
}  // namespace parallel
}  // namespace mindspore
