/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "include/frontend/parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
CostGraphPtr entire_costgraph = nullptr;

void CostGraph::Init() {
  inputs_tensor_name_list_.clear();
  tuple_getitem_list_.clear();
  ops_.clear();
  edges_.clear();
  connected_compoents_.clear();
  out_edges_.clear();
  in_edges_.clear();
}

void CostGraph::RemoveOperator(const OperatorInfoPtr &op) {
  for (auto it = ops_.begin(); it != ops_.end();) {
    if ((*it) == op) {
      it = ops_.erase(it);
    } else {
      ++it;
    }
  }
}

bool CostGraph::IsOperatorInCostGraph(const OperatorInfoPtr &op_test) {
  struct IsInGraph {
    const OperatorInfoPtr test_;
    explicit IsInGraph(const OperatorInfoPtr &n) : test_(n) {}
    bool operator()(const OperatorInfoPtr &in) const { return (test_ == in); }
  };
  return std::any_of(ops_.begin(), ops_.end(), IsInGraph(op_test));
}

void CostGraph::AddEdge(OperatorInfoPtr u_node, OperatorInfoPtr v_node, const EdgePtr &edge) {
  std::vector<EdgePtr> curr_edges(edges_[{u_node, v_node}]);
  curr_edges.push_back(edge);
  edges_[{u_node, v_node}] = curr_edges;

  std::vector<EdgePtr> curr_out_edges(out_edges_[u_node]);
  curr_out_edges.push_back(edge);
  out_edges_[u_node] = curr_out_edges;

  std::vector<EdgePtr> curr_in_edges(in_edges_[v_node]);
  curr_in_edges.push_back(edge);
  in_edges_[v_node] = curr_in_edges;
}

bool CostGraph::IsEdgeInCostGraph(const std::string &test_edge_name, size_t output_index, size_t input_index) {
  for (auto &edge_pair : edges_) {
    auto edges = edge_pair.second;
    for (auto &edge : edges) {
      MS_EXCEPTION_IF_NULL(edge);
      bool bool_result = (edge->edge_name() == test_edge_name) && (edge->prev_op_output_index() == output_index) &&
                         (edge->next_op_input_index() == input_index);
      if (bool_result) {
        return true;
      }
    }
  }
  return false;
}

// '_diff_stra_params' contains all TmpIdentity operators to be processed(i.e. params), the users of
// these operators have different strategies, which need an additional propagation process.
std::set<OperatorInfoPtr> _diff_stra_params;
void CostGraph::StrategyPropagate(const std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare> &ops_stras) {
  if (ops_stras.empty()) {
    MS_LOG(EXCEPTION) << "There is no operator that is configured sharding strategy.";
  }
  configured_ops = ops_stras;
  _diff_stra_params.clear();

  for (auto &op : ops_) {
    visited[op] = false;
  }

  auto it = configured_ops.begin();
  while (it != configured_ops.end()) {
    BFS(it);
    ++it;
  }

  ProcessDiffStraParams();
  _diff_stra_params.clear();
  // GetNext as a isolate op can not be propagated
  for (auto &op : entire_costgraph->GetOperators()) {
    // After BFS, if op still doesn't have strategy, then give it a default one,
    // if op is new shape base, it doesn't has selected strategy
    if (op->selected_strategy() == nullptr && !op->is_new_shape_node()) {
      MS_LOG(INFO) << "Set default strategy for op: " << op->name();
      op->SetSelectedStrategy(op->GetStrategyCost()[0]->strategy_ptr, 0);
    }
  }
}

bool CheckSwcIndexInvalid(int64_t swc_index) { return swc_index == -1; }

bool CostGraph::CheckVisitedEdgeConsistency(const EdgePtr &edge) const {
  MS_EXCEPTION_IF_NULL(edge);
  auto prev_op = edge->prev_operator();
  auto next_op = edge->next_operator();
  bool consistency = true;
  if (prev_op->IsReshape()) {
    const auto &reshape_output_lyt =
      next_op->GetInputLayoutFromSWCByStrategy(next_op->selected_strategy(), edge->next_op_input_index());
    auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(prev_op);
    consistency = reshape_ptr->CheckStrategyConsistencyByOutputLayout(reshape_ptr->swc_index(), reshape_output_lyt);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  } else if (next_op->IsReshape()) {
    const auto &reshape_input_lyt =
      prev_op->GetOutputLayoutFromSWCByStrategy(prev_op->selected_strategy(), edge->prev_op_output_index());
    auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(next_op);
    consistency = reshape_ptr->CheckStrategyConsistencyByInputLayout(reshape_ptr->swc_index(), reshape_input_lyt);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  } else {
    consistency = edge->CheckLayoutConsistency(&_diff_stra_params);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  }
  return consistency;
}

bool CostGraph::CheckConfiguredSuccEdgeConsistency(const EdgePtr &edge) const {
  MS_EXCEPTION_IF_NULL(edge);
  auto curr_op = edge->prev_operator();
  auto next_op = edge->next_operator();
  auto next_op_conf_stra = configured_ops.at(next_op);
  bool consistency = true;
  if (curr_op->IsReshape()) {
    const auto &reshape_output_lyt =
      next_op->GetInputLayoutFromSWCByStrategy(next_op_conf_stra, edge->next_op_input_index());
    auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(curr_op);
    consistency = reshape_ptr->CheckStrategyConsistencyByOutputLayout(reshape_ptr->swc_index(), reshape_output_lyt);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
    return consistency;
  } else {
    consistency = edge->CheckLayoutConsistency(&_diff_stra_params);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  }
  return consistency;
}

void CostGraph::CheckConfiguredPrevEdgeConsistency(const EdgePtr &edge) const {
  auto curr_op = edge->next_operator();
  auto prev_op = edge->prev_operator();
  auto prev_op_conf_stra = configured_ops.at(prev_op);
  if (curr_op->IsReshape()) {
    const auto &reshape_input_lyt =
      prev_op->GetOutputLayoutFromSWCByStrategy(prev_op_conf_stra, edge->prev_op_output_index());
    auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(curr_op);
    auto consistency = reshape_ptr->CheckStrategyConsistencyByInputLayout(reshape_ptr->swc_index(), reshape_input_lyt);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  } else {
    auto consistency = edge->CheckLayoutConsistency(&_diff_stra_params);
    if (!consistency) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge->edge_name();
    }
  }
}

bool IsOpNotPropagate(const OperatorInfoPtr &op1, const OperatorInfoPtr &op2) {
  if (op1->IsStandAlone() || op2->IsStandAlone()) {
    MS_LOG(INFO) << "StandAlone in edge " << op1->name() << " - " << op2->name()
                 << " doesn't propagate strategy, continue.";
    return true;
  }
  if (op1->is_new_shape_node() || op2->is_new_shape_node()) {
    MS_LOG(INFO) << "New shape op in edge " << op1->name() << " - " << op2->name()
                 << " doesn't propagate strategy, continue.";
    return true;
  }
  return false;
}

void CostGraph::AddSwcUnderNextOpDevMatrix(const OperatorInfoPtr &prev_op, const OperatorInfoPtr &curr_op,
                                           const std::shared_ptr<Edge> &edge) {
  if (curr_op->IsReshape() || prev_op->IsReshape() || prev_op->IsMultiInput()) {
    return;
  }

  if (prev_op->AddSwcUnderNextOpDevMatrixSingle(curr_op, edge) != SUCCESS) {
    MS_LOG(INFO) << "AddSwcUnderNextOpDevMatrix failed. curr_op:" << curr_op->name();
  }
}

void CostGraph::BFSPrevNodeIfPrevIsReshape(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->next_operator();
  const auto &prev_op = edge->prev_operator();
  if (curr_op->IsVirtualOutput() || curr_op->IsConcat()) {
    MS_LOG(INFO) << "virtualoutput or concat after reshape, no need set strategy.";
    visited.at(prev_op) = true;
    return;
  }
  auto swc_index = edge->GetReshapeSWCIndexByNextOpStrategy(curr_op->selected_strategy());
  if (CheckSwcIndexInvalid(swc_index)) {
    MS_LOG(WARNING) << "No available strategy found at edge: " << edge->edge_name() << " for: " << prev_op->name();
    return;
  }
  (void)next_level.emplace(std::make_pair(prev_op, std::make_pair(nullptr, swc_index)), curr_depth + 1);
  prev_op->set_swc_index(swc_index, curr_depth + 1);
  visited.at(prev_op) = true;
  return;
}

void CostGraph::BFSPrevNodeIfCurrIsReshape(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->next_operator();
  const auto &prev_op = edge->prev_operator();
  auto prev_stra = edge->GetPrevOpStrategyByReshapeSWCIndex(curr_op->swc_index());
  if (prev_stra == nullptr) {
    MS_LOG(WARNING) << "No available strategy found at edge: " << edge->edge_name() << " for: " << prev_op->name();
    return;
  }
  (void)next_level.emplace(std::make_pair(prev_op, std::make_pair(prev_stra, -1)), curr_depth + 1);
  prev_op->SetSelectedStrategy(prev_stra, LongToSize(curr_depth + 1));
  prev_op->ClearStrategyCost();
  if (prev_op->SetCostUnderStrategy(prev_stra) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure: operator " << prev_op->name() << " SetCostUnderStrategy failed";
  }
  visited.at(prev_op) = true;
  return;
}

void CostGraph::BFSPrevNode(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->next_operator();
  const auto &prev_op = edge->prev_operator();
  bool is_has_stra_op = visited.at(prev_op) || configured_ops.find(prev_op) != configured_ops.end();

  prev_op->LayoutPropagationBegin();

  if (IsOpNotPropagate(prev_op, curr_op)) {
    return;
  }

  if (!is_has_stra_op) {
    AddSwcUnderNextOpDevMatrix(prev_op, curr_op, edge);
  }

  if (edge->InitEdgeCost() != SUCCESS && !is_has_stra_op) {
    MS_LOG(EXCEPTION) << "Edge cost initialization failed.";
  }
  if (visited.at(prev_op)) {
    (void)CheckVisitedEdgeConsistency(edge);
    return;
  }
  if ((curr_depth > 0) && (configured_ops.find(prev_op) != configured_ops.end())) {
    CheckConfiguredPrevEdgeConsistency(edge);
  }
  if (configured_ops.find(prev_op) != configured_ops.end()) {
    return;
  }
  if (prev_op->IsReshape()) {
    return BFSPrevNodeIfPrevIsReshape(edge, curr_depth);
  }
  if (curr_op->IsReshape()) {
    return BFSPrevNodeIfCurrIsReshape(edge, curr_depth);
  }

  const auto &candidate_swc = edge->GetPrevOpSwcByNextOpStrategyWithMiniComm(curr_op->selected_strategy());
  if (candidate_swc == nullptr) {
    MS_LOG(EXCEPTION) << "Current op " << curr_op->name() << "'s strategy is "
                      << curr_op->selected_strategy()->ToString() << ". " << prev_op->name()
                      << "'s candidate swc is null in the edge: " << edge->edge_name();
  }
  const auto &prev_op_stra = candidate_swc->strategy_ptr;
  if (prev_op_stra == nullptr) {
    MS_LOG(EXCEPTION) << "Current op " << curr_op->name() << "'s strategy is "
                      << curr_op->selected_strategy()->ToString() << ". " << prev_op->name()
                      << "'s strategy is null in the edge: " << edge->edge_name();
  }
  (void)next_level.emplace(std::make_pair(prev_op, std::make_pair(prev_op_stra, -1)), curr_depth + 1);

  SetOpStrategy(prev_op, candidate_swc, prev_op_stra, curr_depth);

  prev_op->LayoutPropagationEnd();
  visited.at(prev_op) = true;
  return;
}

void CostGraph::BFSNextNodeIfNextIsReshape(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->prev_operator();
  const auto &next_op = edge->next_operator();
  auto swc_index = edge->GetReshapeSWCIndexByPrevOpStrategy(curr_op->selected_strategy());
  if (CheckSwcIndexInvalid(swc_index)) {
    MS_LOG(WARNING) << "No available strategy found at edge: " << edge->edge_name() << " for: " << next_op->name();
    return;
  }
  (void)next_level.emplace(std::make_pair(next_op, std::make_pair(nullptr, swc_index)), curr_depth + 1);
  next_op->set_swc_index(swc_index, curr_depth + 1);
  visited.at(next_op) = true;
  return;
}

void CostGraph::BFSNextNodeIfCurrIsReshape(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->prev_operator();
  const auto &next_op = edge->next_operator();
  if (next_op->IsVirtualOutput() || next_op->IsConcat()) {
    MS_LOG(INFO) << "virtualoutput or concat after reshape, no need set strategy.";
    visited.at(next_op) = true;
    return;
  }
  auto stra = edge->GetNextOpStrategyByReshapeSWCIndex(curr_op->swc_index());
  if (stra == nullptr) {
    MS_LOG(WARNING) << "No available strategy found at edge: " << edge->edge_name() << " for: " << curr_op->name();
    return;
  }
  (void)next_level.emplace(std::make_pair(next_op, std::make_pair(stra, -1)), curr_depth + 1);
  next_op->SetSelectedStrategy(stra, LongToSize(curr_depth + 1));
  next_op->ClearStrategyCost();
  if (next_op->SetCostUnderStrategy(stra) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure: operator " << next_op->name() << " SetCostUnderStrategy failed";
  }
  visited.at(next_op) = true;
  return;
}

void CostGraph::SetOpStrategy(const OperatorInfoPtr &curr_op, const std::shared_ptr<StrategyWithCost> &candidate_swc,
                              const StrategyPtr &curr_op_stra, int64_t curr_depth) {
  curr_op->SetSelectedStrategy(curr_op_stra, LongToSize(curr_depth + 1));
  curr_op->ClearStrategyCost();
  if (curr_op->SetCostUnderStrategyWithCost(candidate_swc) == SUCCESS) {
    curr_op->set_config_by_layout(true);
  } else {
    MS_LOG(WARNING) << "Operator " << curr_op->name()
                    << " failed to set cost by layout. Try to set cost by strategy tuple.";
    curr_op->ClearStrategyCost();
    if (curr_op->SetCostUnderStrategy(curr_op_stra) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Operator " << curr_op->name() << " set cost by strategy tuple failed";
    }
  }
}

void CostGraph::AddSwcUnderPrevOpDevMatrix(const OperatorInfoPtr &curr_op, const OperatorInfoPtr &next_op,
                                           const std::shared_ptr<Edge> &edge) {
  if (curr_op->IsReshape() || next_op->IsReshape()) {
    return;
  }
  if (next_op->IsMultiInput()) {
    next_op->AddVisitedEdge(edge);
    if (next_op->AllInputsVisited()) {
      if (next_op->AddSwcUnderPrevOpDevMatrixMulti() != SUCCESS) {
        MS_LOG(WARNING) << "AddSwcUnderPrevOpDevMatrixMulti failed. curr_op: " << curr_op->name();
      }
    }
  } else {
    if (!curr_op->outputs_tensor_map_before().empty()) {
      if (next_op->AddSwcUnderPrevOpDevMatrixSingle(curr_op->out_dev_matrix_shape(),
                                                    curr_op->outputs_tensor_map_before()[edge->prev_op_output_index()],
                                                    edge->next_op_input_index()) != SUCCESS) {
        MS_LOG(WARNING) << "AddSwcUnderPrevOpDevMatrix failed. curr_op:" << curr_op->name();
      }
    }
  }
}

bool CostGraph::BFSNextNodeInitEdgeAndCheck(const std::shared_ptr<Edge> &edge, const OperatorInfoPtr &curr_op,
                                            const OperatorInfoPtr &next_op, int64_t curr_depth) {
  bool is_has_stra_op = visited.at(next_op) || configured_ops.find(next_op) != configured_ops.end();

  if (!is_has_stra_op && next_op->IsMultiInput()) {
    if (next_op->AllInputsVisited()) {
      next_op->InitVisitedEdges();
    }
  } else {
    if (edge->InitEdgeCost() != SUCCESS && !is_has_stra_op) {
      MS_LOG(EXCEPTION) << "Edge cost initialization failed.";
    }
  }

  if (visited.at(next_op)) {
    (void)CheckVisitedEdgeConsistency(edge);
    return false;
  }
  if ((curr_depth > 0) && (configured_ops.find(next_op) != configured_ops.end())) {
    (void)CheckConfiguredSuccEdgeConsistency(edge);
  }
  if (configured_ops.find(next_op) != configured_ops.end()) {
    return false;
  }
  return true;
}

void CostGraph::BFSNextNode(const std::shared_ptr<Edge> &edge, int64_t curr_depth) {
  MS_EXCEPTION_IF_NULL(edge);
  const auto &curr_op = edge->prev_operator();
  const auto &next_op = edge->next_operator();

  if (IsOpNotPropagate(curr_op, next_op)) {
    return;
  }

  bool is_has_stra_op = visited.at(next_op) || configured_ops.find(next_op) != configured_ops.end();
  next_op->LayoutPropagationBegin();
  if (!is_has_stra_op) {
    (void)AddSwcUnderPrevOpDevMatrix(curr_op, next_op, edge);
  }

  if (!BFSNextNodeInitEdgeAndCheck(edge, curr_op, next_op, curr_depth)) {
    return;
  }

  if (next_op->IsReshape()) {
    return BFSNextNodeIfNextIsReshape(edge, curr_depth);
  }
  if (curr_op->IsReshape()) {
    return BFSNextNodeIfCurrIsReshape(edge, curr_depth);
  }

  if (curr_op->out_strategy() != nullptr) {
    const auto &next_op_stra = edge->GetNextOpStrategyByOutStrategy(curr_op->out_strategy());
    if (next_op_stra == nullptr) {
      MS_LOG(EXCEPTION) << "Current op " << curr_op->name() << "'s strategy is "
                        << curr_op->selected_strategy()->ToString() << ". " << next_op->name()
                        << "'s strategy is null in the edge: " << edge->edge_name();
    }
    (void)next_level.emplace(std::make_pair(next_op, std::make_pair(next_op_stra, -1)), curr_depth + 1);
    next_op->SetSelectedStrategy(next_op_stra, LongToSize(curr_depth + 1));
    next_op->ClearStrategyCost();
    if (next_op->SetCostUnderStrategy(next_op_stra) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Failure: operator " << next_op->name() << " SetCostUnderStrategy failed";
    }
    visited.at(next_op) = true;
    return;
  }

  MS_LOG(INFO) << "BFSNextNode: cur op: " << curr_op->name() << " next_op: " << next_op->name();

  std::shared_ptr<StrategyWithCost> candidate_swc = nullptr;

  if (next_op->IsMultiInput()) {
    MS_LOG(INFO) << "BFSNextNode next_op is multi-input op! cur:" << curr_op->name() << " next: " << next_op->name()
                 << " cnode name: " << next_op->cnode()->fullname_with_scope();
    if (next_op->AllInputsVisited()) {
      MS_LOG(INFO) << "next_op AllInputsVisited";
      candidate_swc = edge->GetNextOpStrategyByCurMultiInput(curr_depth, &waitting_list_);
    } else {
      // There is input of next_op not visited.
      MS_LOG(INFO) << "next_op_visited_edges size: " << next_op->get_visited_edges().size();
      waitting_list_[next_op] = curr_depth;
      // Do nothing when next_op has candidate strategies
      MS_LOG(INFO) << "BFSNextNode next_op is multi-input op! exist candidate strategy";
      return;
    }

    MS_LOG(INFO) << "Get selected next op strategy: " << candidate_swc->strategy_ptr->ToString();
    // Clear VisitedEdges when next_op has selected strategy.
    next_op->ClearVisitedEdges();
    MS_LOG(INFO) << "next_op->ClearVisitedEdges() finish";
  } else {
    candidate_swc = edge->GetNextOpSwcByPrevOpStrategyWithMiniComm(curr_op->selected_strategy());
  }

  if (candidate_swc == nullptr) {
    MS_LOG(EXCEPTION) << "Current op " << curr_op->name() << "'s strategy is "
                      << curr_op->selected_strategy()->ToString() << ". " << next_op->name()
                      << "'s candidate swc is null in the edge: " << edge->edge_name();
    return;
  }
  const auto &next_op_stra = candidate_swc->strategy_ptr;
  if (next_op_stra == nullptr) {
    MS_LOG(EXCEPTION) << "Current op " << curr_op->name() << "'s strategy is "
                      << curr_op->selected_strategy()->ToString() << ". " << next_op->name()
                      << "'s strategy is null in the edge: " << edge->edge_name();
    return;
  }
  (void)next_level.emplace(std::make_pair(next_op, std::make_pair(next_op_stra, -1)), curr_depth + 1);

  SetOpStrategy(next_op, candidate_swc, next_op_stra, curr_depth);

  next_op->LayoutPropagationEnd();
  visited.at(next_op) = true;
}

bool CostGraph::ExistPrevConfiguredOps(
  const OperatorInfoPtr &cur_op,
  const std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare>::const_iterator &configured_ops_it) {
  auto it = configured_ops_it;
  ++it;
  if (it == configured_ops.end()) {
    return false;
  }
  return cur_op->get_topo_index() > it->first->get_topo_index();
}

void CostGraph::BFS(const std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare>::const_iterator &configured_ops_it) {
  const OperatorInfoPtr &op = configured_ops_it->first;
  const StrategyPtr &op_stra = configured_ops_it->second;
  next_level = std::queue<std::pair<std::pair<OperatorInfoPtr, std::pair<StrategyPtr, int64_t>>, int64_t>>();
  (void)next_level.emplace(std::make_pair(op, std::make_pair(op_stra, -1)), 0);
  op->SetSelectedStrategy(op_stra, 0);

  while (!next_level.empty()) {
    auto curr_op = next_level.front().first.first;
    auto configured_stra = next_level.front().first.second.first;
    auto curr_depth = next_level.front().second;
    visited.at(curr_op) = true;
    MS_LOG(INFO) << "curr_depth: " << curr_depth << " curr_op: " << curr_op->name();
    for (auto &edge : curr_op->succ_edges()) {
      BFSNextNode(edge, curr_depth);
    }
    for (auto &edge : curr_op->prev_edges()) {
      BFSPrevNode(edge, curr_depth);
    }
    next_level.pop();

    if (next_level.empty() && !waitting_list_.empty()) {
      MS_LOG(INFO) << "waitting_list_ not empty!";

      auto it = waitting_list_.begin();

      const OperatorInfoPtr &op_with_cands = it->first;
      if (visited.at(op_with_cands) || ExistPrevConfiguredOps(op_with_cands, configured_ops_it)) {
        MS_LOG(INFO) << "ExistPrevConfiguredOps Skip";
        continue;
      }
      int64_t saved_depth = 0;

      op_with_cands->InitVisitedEdges();
      std::shared_ptr<StrategyWithCost> selected_swc = op_with_cands->GetStrategyByVisitedEdges();

      StrategyPtr &selected_strategy = selected_swc->strategy_ptr;

      SetOpStrategy(op_with_cands, selected_swc, selected_strategy, curr_depth);

      MS_LOG(INFO) << "waitting_list_ to next_level, name: " << op_with_cands->name() << " depth: " << saved_depth
                   << " selected strategy: " << selected_strategy->ToString();

      next_level.emplace(std::make_pair(op_with_cands, std::make_pair(selected_strategy, -1)), saved_depth);
      waitting_list_.erase(it);
    }
  }
}

// An additional propagation to deal with incontinuous strategy between the ops that share params.
void CostGraph::ProcessDiffStraParams() const {
  for (auto &curr_identity_op : _diff_stra_params) {
    auto succ_edges = curr_identity_op->succ_edges();
    auto is_target = [&](const std::shared_ptr<Edge> &edge) {
      return configured_ops.find(edge->next_operator()) != configured_ops.end();
    };
    auto it = std::find_if(succ_edges.begin(), succ_edges.end(), is_target);
    // if there are configured ops in the users, update the strategy of identity_op
    if (it != succ_edges.end()) {
      auto consistency = CheckVisitedEdgeConsistency(*it);
      if (!consistency) {
        curr_identity_op->ClearStrategyCost();
        (void)curr_identity_op->GenerateStrategies(0);
        if ((*it)->InitEdgeCost() != SUCCESS) {
          MS_LOG(EXCEPTION) << "Edge cost initialization failed.";
        }
        const auto curr_op = (*it)->next_operator();
        const auto &candidate_swc = (*it)->GetPrevOpSwcByNextOpStrategyWithMiniComm(curr_op->selected_strategy());
        if (candidate_swc == nullptr) {
          MS_LOG(EXCEPTION) << curr_identity_op->name()
                            << "'s candidate swc is null in the edge: " << (*it)->edge_name();
        }
        const auto &prev_op_stra = candidate_swc->strategy_ptr;
        if (prev_op_stra == nullptr) {
          MS_LOG(EXCEPTION) << curr_identity_op->name() << "'s strategy is null in the edge: " << (*it)->edge_name();
        }
        curr_identity_op->SetSelectedStrategy(prev_op_stra, 1);
        curr_identity_op->ClearStrategyCost();
        if (curr_identity_op->SetCostUnderStrategy(prev_op_stra) != SUCCESS) {
          MS_LOG(EXCEPTION) << "Failure: operator " << curr_identity_op->name() << " SetCostUnderStrategy failed";
        }
      }
    }
    for (auto &edge : succ_edges) {
      const auto &next_op = edge->next_operator();
      ParamPropagation(curr_identity_op, edge);
      if (next_op->name().find(CAST) != std::string::npos) {
        for (auto &cast_edge : next_op->succ_edges()) {
          ParamPropagation(next_op, cast_edge);
        }
      }
    }
  }
}

void CostGraph::ParamPropagation(const OperatorInfoPtr &curr_op, const std::shared_ptr<Edge> edge) const {
  const auto &next_op = edge->next_operator();
  MS_LOG(INFO) << "params propagation at " << curr_op->name() << "->" << next_op->name();
  if ((configured_ops.find(next_op) != configured_ops.end())) {
    auto consistency = CheckConfiguredSuccEdgeConsistency(edge);
    if (!consistency) {
      MS_LOG(EXCEPTION)
        << "Incorrect strategy configuration, it may be caused by configuring inconsistent strategies for operators.";
    }
    return;
  }
  auto consistency = CheckVisitedEdgeConsistency(edge);
  if (consistency) {
    MS_LOG(INFO) << "Jump the consistency edge.";
    return;
  }
  next_op->ClearStrategyCost();
  (void)next_op->GenerateStrategies(0);
  if (edge->InitEdgeCost() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Edge cost initialization failed.";
  }
  const auto &candidate_swc = edge->GetNextOpSwcByPrevOpStrategyWithMiniComm(curr_op->selected_strategy());
  if (candidate_swc == nullptr) {
    MS_LOG(EXCEPTION) << next_op->name() << "'s candidate swc is null in the edge: " << edge->edge_name();
  }
  const auto &next_op_stra = candidate_swc->strategy_ptr;
  if (next_op_stra == nullptr) {
    MS_LOG(EXCEPTION) << next_op->name() << "'s strategy is null in the edge: " << edge->edge_name();
  }
  next_op->SetSelectedStrategy(next_op_stra, 1);
  next_op->ClearStrategyCost();
  if (next_op->SetCostUnderStrategy(next_op_stra) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure: operator " << next_op->name() << " SetCostUnderStrategy failed";
  }
}

std::vector<std::shared_ptr<CostGraph>> CostGraph::ConstructConnectedComponents(
  std::vector<OperatorInfoPtr> alive_ops) {
  for (auto &op : alive_ops) {
    visited[op] = false;
  }

  MS_LOG(INFO) << "visited: " << visited.size() << ".";
  for (auto &op : alive_ops) {
    if ((!visited[op]) && op->is_alive()) {
      std::shared_ptr<CostGraph> new_component = std::make_shared<CostGraph>();
      MS_EXCEPTION_IF_NULL(new_component);
      DFS(op, new_component);
      connected_compoents_.push_back(new_component);
    }
  }
  return connected_compoents_;
}

void CostGraph::DFS(const OperatorInfoPtr &current_op, const std::shared_ptr<CostGraph> &component) {
  MS_EXCEPTION_IF_NULL(component);
  visited.at(current_op) = true;
  component->AddOperator(current_op);

  for (auto &edge : current_op->succ_edges()) {
    bool bool_test = (visited.find(edge->next_operator()) != visited.end()) && (!visited.at(edge->next_operator())) &&
                     edge->next_operator()->is_alive();
    if (bool_test) {
      component->AddEdge(current_op, edge->next_operator(), edge);
      DFS(edge->next_operator(), component);
    }
  }

  for (auto &edge : current_op->prev_edges()) {
    bool bool_test = (visited.find(edge->prev_operator()) != visited.end()) && (!visited.at(edge->prev_operator())) &&
                     edge->prev_operator()->is_alive();
    if (bool_test) {
      component->AddEdge(edge->prev_operator(), current_op, edge);
      DFS(edge->prev_operator(), component);
    }
  }
}

// Create final cost list for the graph: u --> v
CostPtrList CostGraph::CreateFinalCostList(const OperatorInfoPtr &u, const std::shared_ptr<Edge> &e,
                                           const OperatorInfoPtr &v) const {
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(e);
  CostPtrList ret;
  for (const auto &u_strategy : u->GetStrategyCost()) {
    for (const auto &v_strategy : v->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(u_strategy);
      MS_EXCEPTION_IF_NULL(v_strategy);
      auto u_strategy_ptr = u_strategy->strategy_ptr;
      auto v_strategy_ptr = v_strategy->strategy_ptr;
      CostPtrList clist1 = u_strategy->cost_list;
      CostPtrList clist2 = e->GetCostList(u_strategy_ptr, v_strategy_ptr);
      CostPtrList clist3 = v_strategy->cost_list;
      for (const auto &cost1 : clist1) {
        for (const auto &cost2 : clist2) {
          for (const auto &cost3 : clist3) {
            MS_EXCEPTION_IF_NULL(cost1);
            MS_EXCEPTION_IF_NULL(cost2);
            MS_EXCEPTION_IF_NULL(cost3);
            double computation = cost1->computation_cost_ + cost2->computation_cost_ + cost3->computation_cost_;
            double memory = cost1->memory_with_reuse_ + cost2->memory_with_reuse_ + cost3->memory_with_reuse_;
            double communication = cost1->communication_cost_ + cost2->communication_cost_ + cost3->communication_cost_;
            double communication_forward =
              cost1->communication_forward_ + cost2->communication_forward_ + cost3->communication_forward_;
            double communication_without_para = cost1->communication_without_parameter_ +
                                                cost2->communication_without_parameter_ +
                                                cost3->communication_without_parameter_;
            auto decision =
              std::make_shared<FinalDecision>(u_strategy->strategy_ptr, v_strategy->strategy_ptr, cost1, cost2, cost3);
            auto cost = std::make_shared<Cost>(computation, communication, decision);
            const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
            MS_EXCEPTION_IF_NULL(cost);
            cost->communication_without_parameter_ = communication_without_para;
            cost->communication_with_partial_para_ =
              communication_without_para + gamma * (communication - communication_without_para);
            cost->memory_with_reuse_ = memory;
            cost->communication_forward_ = communication_forward;
            ret.push_back(cost);
          }
        }
      }
    }
  }

  Simplify(&ret);
  return ret;
}

// Create final cost list for the graph containing a single node: u
CostPtrList CostGraph::CreateFinalSingleCostList(const OperatorInfoPtr &u) const {
  MS_EXCEPTION_IF_NULL(u);
  CostPtrList ret;
  for (const auto &u_strategy : u->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(u_strategy);
    auto u_strategy_ptr = u_strategy->strategy_ptr;
    CostPtrList clist1 = u_strategy->cost_list;
    for (const auto &cost1 : clist1) {
      MS_EXCEPTION_IF_NULL(cost1);
      auto decision = std::make_shared<FinalSingleDecision>(u_strategy_ptr, cost1);
      auto new_cost = std::make_shared<Cost>(cost1->computation_cost_, cost1->communication_cost_, decision);
      const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
      MS_EXCEPTION_IF_NULL(new_cost);
      new_cost->communication_without_parameter_ = cost1->communication_without_parameter_;
      new_cost->communication_with_partial_para_ =
        cost1->communication_without_parameter_ +
        gamma * (cost1->communication_cost_ - cost1->communication_without_parameter_);
      new_cost->memory_with_reuse_ = cost1->memory_with_reuse_;
      new_cost->communication_forward_ = cost1->communication_forward_;
      ret.push_back(new_cost);
    }
  }

  Simplify(&ret);
  return ret;
}

CostPtr CostGraph::SelectCostWithMinInferenceTime(const CostPtrList &cost_list, double memory) const {
  // Select the cost with minimum inference time. Currently, the inference time is modeled as =
  // costmodel_alpha_ * computation_cost + costmodel_beta_ * communication_forward_
  if (cost_list.empty()) {
    MS_LOG(ERROR) << "Final cost list is null.";
    return nullptr;
  }
  CostPtrList after_mem_filter;
  double minimum_memory = DBL_MAX;
  // Filter out the valid costs.
  for (auto &a_cost : cost_list) {
    if (a_cost->memory_with_reuse_ <= memory) {
      after_mem_filter.push_back(a_cost);
    } else if (a_cost->memory_with_reuse_ < minimum_memory) {
      minimum_memory = a_cost->memory_with_reuse_;
    }
  }
  if (after_mem_filter.empty()) {
    MS_LOG(ERROR) << "No available cost. The minimum memory cost is: " << minimum_memory
                  << ", the memory capacity is: " << memory << ".";
    return nullptr;
  }
  // Init the returned value with first cost.
  CostPtr ret = after_mem_filter[0];
  const auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  const auto beta = CostModelContext::GetInstance()->costmodel_beta();

  double minimum = alpha * ret->computation_cost_ + beta * ret->communication_forward_;
  MS_LOG(INFO) << "Cost 0: "
               << "memory_cost: " << ret->memory_with_reuse_ << ", computation_cost_: " << ret->computation_cost_
               << ", communication_forward_: " << ret->communication_forward_
               << ", communication_with_partial_para_: " << ret->communication_with_partial_para_
               << ", communication_cost_: " << ret->communication_cost_
               << ", communication_without_parameter_: " << ret->communication_without_parameter_ << ".";
  MS_LOG(INFO) << "Cost 0: total_cost: " << minimum;
  for (size_t i = 1; i < after_mem_filter.size(); ++i) {
    MS_EXCEPTION_IF_NULL(after_mem_filter[i]);
    MS_LOG(INFO) << "Cost " << i << ": memory_cost: " << after_mem_filter[i]->memory_with_reuse_
                 << ", computation_cost_: " << after_mem_filter[i]->computation_cost_
                 << ", communication_forward_: " << after_mem_filter[i]->communication_forward_
                 << ", communication_with_partial_para_: " << after_mem_filter[i]->communication_with_partial_para_
                 << ", communication_cost_: " << after_mem_filter[i]->communication_cost_
                 << ", communication_without_parameter_: " << after_mem_filter[i]->communication_without_parameter_
                 << ".";
    auto tmp = alpha * after_mem_filter[i]->computation_cost_ + beta * after_mem_filter[i]->communication_forward_;
    MS_LOG(INFO) << "Cost " << i << ": total_cost: " << tmp;
    if (minimum > tmp) {
      minimum = tmp;
      ret = after_mem_filter[i];
      MS_LOG(INFO) << "Selected: " << i;
    }
  }
  return ret;
}

CostPtr CostGraph::SelectCostWithMinTrainingTime(const CostPtrList &cost_list, double memory) const {
  // Select the cost with minimum training time. Currently, the training time is modeled as =
  // costmodel_alpha_ * computation_cost + costmodel_beta_ * communication_with_partial_para_
  if (cost_list.empty()) {
    MS_LOG(ERROR) << "Final cost list is null.";
    return nullptr;
  }
  CostPtrList after_mem_filter;
  double minimum_memory = DBL_MAX;
  // Filter out the valid costs.
  for (auto &a_cost : cost_list) {
    if (a_cost->memory_with_reuse_ <= memory) {
      after_mem_filter.push_back(a_cost);
    } else if (a_cost->memory_with_reuse_ < minimum_memory) {
      minimum_memory = a_cost->memory_with_reuse_;
    }
  }
  if (after_mem_filter.empty()) {
    MS_LOG(ERROR) << "No available cost. The minimum memory cost is: " << minimum_memory
                  << ", the memory capacity is: " << memory << ".";
    return nullptr;
  }
  // Init the returned value with first cost.
  CostPtr ret = after_mem_filter[0];
  const auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  const auto beta = CostModelContext::GetInstance()->costmodel_beta();

  double minimum = alpha * ret->computation_cost_ + beta * ret->communication_with_partial_para_;
  MS_LOG(INFO) << "Cost 0: "
               << "memory_cost: " << ret->memory_with_reuse_ << ", computation_cost_: " << ret->computation_cost_
               << ", communication_with_partial_para_: " << ret->communication_with_partial_para_
               << ", communication_cost_: " << ret->communication_cost_
               << ", communication_without_parameter_: " << ret->communication_without_parameter_ << ".";
  MS_LOG(INFO) << "Cost 0: total_cost: " << minimum;
  for (size_t i = 1; i < after_mem_filter.size(); ++i) {
    MS_EXCEPTION_IF_NULL(after_mem_filter[i]);
    MS_LOG(INFO) << "Cost " << i << ": memory_cost: " << after_mem_filter[i]->memory_with_reuse_
                 << ", computation_cost_: " << after_mem_filter[i]->computation_cost_
                 << ", communication_with_partial_para_: " << after_mem_filter[i]->communication_with_partial_para_
                 << ", communication_cost_: " << after_mem_filter[i]->communication_cost_
                 << ", communication_without_parameter_: " << after_mem_filter[i]->communication_without_parameter_
                 << ".";
    auto tmp =
      alpha * after_mem_filter[i]->computation_cost_ + beta * after_mem_filter[i]->communication_with_partial_para_;
    MS_LOG(INFO) << "Cost " << i << ": total_cost: " << tmp;
    if (minimum > tmp) {
      minimum = tmp;
      ret = after_mem_filter[i];
      MS_LOG(INFO) << "Selected: " << i;
    }
  }
  return ret;
}

CostPtrList CostGraph::SelectCostListWithMinTrainingTimeMultiple(const std::vector<CostPtrList> &all_cost_list,
                                                                 double available_memory) const {
  CostPtrList selected_cost_list(all_cost_list.size(), nullptr);
  double minimum = DBL_MAX;
  double total_memory = 0.0;
  CostPtrList ret(all_cost_list.size(), nullptr);
  // Check whether valid costs exist.
  for (size_t i = 0; i < all_cost_list.size(); ++i) {
    if (all_cost_list[i][0] == nullptr) {
      MS_LOG(ERROR) << "The cost list " << i << " is empty.";
      return ret;
    } else {
      double memory_i_cost = DBL_MAX;
      for (size_t j = 0; j < all_cost_list[i].size(); ++j) {
        if (all_cost_list[i][j]->memory_with_reuse_ < memory_i_cost) {
          memory_i_cost = all_cost_list[i][j]->memory_with_reuse_;
        }
      }
      total_memory += memory_i_cost;
    }
  }
  if (total_memory >= available_memory) {
    MS_LOG(ERROR) << "No strategy can be found under current memory: " << available_memory
                  << ", minimum strategy cost: " << total_memory << ".";
    return selected_cost_list;
  }

  std::function<void(size_t)> recursive = [&all_cost_list, &selected_cost_list, &minimum, &ret, &recursive,
                                           &available_memory](size_t k) {
    const auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
    const auto beta = CostModelContext::GetInstance()->costmodel_beta();
    if (k == all_cost_list.size()) {
      double tmp_memory = 0.0;
      double tmp_minimum = 0.0;
      for (size_t i = 0; i < selected_cost_list.size(); ++i) {
        MS_EXCEPTION_IF_NULL(selected_cost_list[i]);
        tmp_memory += selected_cost_list[i]->memory_with_reuse_;
        tmp_minimum += alpha * selected_cost_list[i]->computation_cost_ +
                       beta * selected_cost_list[i]->communication_with_partial_para_;
      }
      MS_LOG(INFO) << "tmp_memory: " << tmp_memory << ", tmp_minimum: " << tmp_minimum << ", minimum: " << minimum
                   << ".";
      if (tmp_memory < available_memory && tmp_minimum < minimum) {
        ret = selected_cost_list;
        minimum = tmp_minimum;
        MS_LOG(INFO) << "selected tmp_memory: " << tmp_memory << ", tmp_minimum: " << tmp_minimum << ".";
      }
      return;
    }

    MS_LOG(DEBUG) << "The value minimum: " << minimum << ", available_memory: " << available_memory << ".";
    for (auto &c : all_cost_list[k]) {
      selected_cost_list[k] = c;
      recursive(k + 1);
    }
  };
  recursive(0);
  return ret;
}

Status CostGraph::SearchStrategyForMultiNodeFinalGraph(const std::vector<OperatorInfoPtr> &alive_ops) {
  MS_LOG(INFO) << "There are " << alive_ops.size() << " nodes in the final graph.";
  auto connected_components = ConstructConnectedComponents(alive_ops);
  MS_LOG(INFO) << "There are " << connected_components.size() << " components in the final graph.";
  std::vector<CostPtrList> all_list;
  for (size_t j = 0; j < connected_components.size(); ++j) {
    auto one_component = connected_components[j];
    MS_EXCEPTION_IF_NULL(one_component);
    if (one_component->GetOperators().size() == 1) {
      MS_LOG(INFO) << "There are 1 operator in a component in the final graph.";
      auto cost_list_1 = one_component->CreateFinalSingleCostList(one_component->GetOperators()[0]);
      all_list.push_back(cost_list_1);
    } else if (one_component->GetOperators().size() == 2) {
      MS_LOG(INFO) << "There are 2 operators in a component in the final graph.";
      OperatorInfoPtr u;
      OperatorInfoPtr v;
      auto first_op = one_component->GetOperators()[0];
      auto second_op = one_component->GetOperators()[1];
      MS_EXCEPTION_IF_NULL(first_op);
      MS_EXCEPTION_IF_NULL(second_op);
      if (!first_op->GetAliveSuccEdges().empty() &&
          first_op->GetAliveSuccEdges()[0]->next_operator().get() == second_op.get()) {
        u = first_op;
        v = second_op;
      } else if (!second_op->GetAliveSuccEdges().empty() &&
                 second_op->GetAliveSuccEdges()[0]->next_operator().get() == first_op.get()) {
        u = second_op;
        v = first_op;
      } else {
        MS_LOG(EXCEPTION) << "The final graph is not the case of u --> v, " << first_op->GetAliveSuccEdges().size()
                          << ", " << second_op->GetAliveSuccEdges().size() << ".";
      }
      MS_EXCEPTION_IF_NULL(u);
      auto e = u->GetAliveSuccEdges()[0];
      auto cost_list = one_component->CreateFinalCostList(u, e, v);
      all_list.push_back(cost_list);
    } else {
      MS_LOG(EXCEPTION) << "There are " << one_component->GetOperators().size()
                        << " operators in a component in the final graph.";
    }
  }
  const auto device_mem_capacity = CostModelContext::GetInstance()->device_memory_capacity();
  auto selected_cost_list = SelectCostListWithMinTrainingTimeMultiple(all_list, device_mem_capacity);
  for (size_t k = 0; k < selected_cost_list.size(); ++k) {
    auto selected_cost = selected_cost_list[k];
    if (selected_cost == nullptr) {
      MS_LOG(ERROR) << "No valid strategy can be found under the current device memory: " << device_mem_capacity << ".";
      return FAILED;
    }
    MS_EXCEPTION_IF_NULL(connected_components[k]);
    MS_EXCEPTION_IF_NULL(selected_cost->decision_ptr_);
    if (connected_components[k]->GetOperators().size() == 1) {
      auto u = connected_components[k]->GetOperators()[0];
      auto decision_f = selected_cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
      MS_EXCEPTION_IF_NULL(decision_f);
      u->SetSelectedStrategyAndCost(decision_f->u_strategy_, decision_f->u_cost_);
      MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
    } else if (connected_components[k]->GetOperators().size() == 2) {
      OperatorInfoPtr u = nullptr;
      OperatorInfoPtr v = nullptr;
      auto first_op = connected_components[k]->GetOperators()[0];
      auto second_op = connected_components[k]->GetOperators()[1];
      MS_EXCEPTION_IF_NULL(first_op);
      MS_EXCEPTION_IF_NULL(second_op);
      if (!first_op->GetAliveSuccEdges().empty() &&
          first_op->GetAliveSuccEdges()[0]->next_operator().get() == second_op.get()) {
        u = first_op;
        v = second_op;
      } else if (!second_op->GetAliveSuccEdges().empty() &&
                 second_op->GetAliveSuccEdges()[0]->next_operator().get() == first_op.get()) {
        u = second_op;
        v = first_op;
      }
      MS_EXCEPTION_IF_NULL(u);
      auto e = u->GetAliveSuccEdges()[0];
      MS_EXCEPTION_IF_NULL(v);
      MS_EXCEPTION_IF_NULL(e);
      auto decision = selected_cost->decision_ptr_->cast<FinalDecisionPtr>();
      MS_EXCEPTION_IF_NULL(decision);
      u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->left_cost_);
      v->SetSelectedStrategyAndCost(decision->v_strategy_, decision->right_cost_);
      e->set_selected_cost(decision->middle_cost_);
      MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
    }
  }
  return SUCCESS;
}

Status CostGraph::SearchStrategyForTwoNodeFinalGraph(const std::vector<OperatorInfoPtr> &alive_ops) {
  // In this case, the final graph should contains exactly 2 nodes.
  if (alive_ops.empty()) {
    MS_LOG(INFO) << "0 Operator in the final graph.";
    return SUCCESS;
  }
  OperatorInfoPtr u;
  OperatorInfoPtr v;
  MS_EXCEPTION_IF_NULL(alive_ops[0]);
  MS_EXCEPTION_IF_NULL(alive_ops[1]);
  const auto phase = CostModelContext::GetInstance()->run_phase();
  const auto device_mem_capacity = CostModelContext::GetInstance()->device_memory_capacity();
  if (!alive_ops[0]->GetAliveSuccEdges().empty() &&
      alive_ops[0]->GetAliveSuccEdges()[0]->next_operator().get() == alive_ops[1].get()) {
    u = alive_ops[0];
    v = alive_ops[1];
  } else if (!alive_ops[1]->GetAliveSuccEdges().empty() &&
             alive_ops[1]->GetAliveSuccEdges()[0]->next_operator().get() == alive_ops[0].get()) {
    u = alive_ops[1];
    v = alive_ops[0];
  } else {
    if (!alive_ops[0]->GetAliveSuccEdges().empty() || !alive_ops[1]->GetAliveSuccEdges().empty()) {
      MS_LOG(EXCEPTION) << "The final graph is not the case of u --> v, " << alive_ops[0]->GetAliveSuccEdges().size()
                        << ", " << alive_ops[1]->GetAliveSuccEdges().size() << ".";
    } else {
      // In this case, the final graph consists of two single nodes
      MS_LOG(INFO) << "There are 2 single nodes in the final graph.";
      std::vector<CostPtrList> all_list;
      auto connected_components = ConstructConnectedComponents(alive_ops);
      MS_LOG(INFO) << "There are " << connected_components.size() << " components in the final graph.";
      for (size_t i = 0; i < connected_components.size(); ++i) {
        MS_LOG(INFO) << "There are 1 operator in a component in the final graph.";
        auto one_component = connected_components[i];
        MS_EXCEPTION_IF_NULL(one_component);
        auto cost_list = one_component->CreateFinalSingleCostList(one_component->GetOperators()[0]);
        all_list.push_back(cost_list);
      }
      CostPtrList selected_cost_list;
      if (phase == TRAINING_PHASE) {
        // training phase
        selected_cost_list = SelectCostListWithMinTrainingTimeMultiple(all_list, device_mem_capacity);
      } else {
        // inference phase
        MS_LOG(EXCEPTION) << "Currently, searching strategy for the two-separated-node final graph in the inference "
                             "phase is not supported.";
      }
      for (size_t k = 0; k < selected_cost_list.size(); ++k) {
        auto selected_cost = selected_cost_list[k];
        if (selected_cost == nullptr) {
          MS_LOG(ERROR) << "No valid strategy can be found under the current device memory: " << device_mem_capacity
                        << ".";
          return FAILED;
        }
        MS_EXCEPTION_IF_NULL(connected_components[k]);
        auto one_operator = connected_components[k]->GetOperators()[0];
        MS_EXCEPTION_IF_NULL(selected_cost->decision_ptr_);
        auto decision = selected_cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
        MS_EXCEPTION_IF_NULL(decision);
        one_operator->SetSelectedStrategyAndCost(decision->u_strategy_, decision->u_cost_);
        MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
      }

      return SUCCESS;
    }
  }
  MS_LOG(INFO) << "There are 2 nodes in the final graph.";
  // In this case, the finale graph is exactly of the form: u --> v
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  auto e = u->GetAliveSuccEdges()[0];
  MS_EXCEPTION_IF_NULL(e);
  auto f_cost_list = CreateFinalCostList(u, e, v);
  CostPtr cost = nullptr;
  if (phase == TRAINING_PHASE) {
    // training phase
    cost = SelectCostWithMinTrainingTime(f_cost_list, device_mem_capacity);
  } else {
    MS_LOG(EXCEPTION) << "Currently, searching strategy for the two-connected-node final graph in the inference "
                         "phase is not supported.";
  }
  if (cost == nullptr) {
    MS_LOG(ERROR) << "No valid strategy can be found under the current device memory: " << device_mem_capacity << ".";
    return FAILED;
  }
  MS_EXCEPTION_IF_NULL(cost->decision_ptr_);
  auto f_decision = cost->decision_ptr_->cast<FinalDecisionPtr>();
  MS_EXCEPTION_IF_NULL(f_decision);
  u->SetSelectedStrategyAndCost(f_decision->u_strategy_, f_decision->left_cost_);
  v->SetSelectedStrategyAndCost(f_decision->v_strategy_, f_decision->right_cost_);
  e->set_selected_cost(f_decision->middle_cost_);
  MS_LOG(INFO) << "Searching the strategy for the eliminated final graph ended.";
  return SUCCESS;
}

// searching the strategy for the final eliminated graph
Status CostGraph::SearchStrategy() {
  MS_LOG(INFO) << "Searching the strategy for the eliminated final graph began.";
  std::vector<OperatorInfoPtr> alive_ops;
  (void)std::for_each(ops_.begin(), ops_.end(), [&alive_ops](const OperatorInfoPtr &op) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->is_alive()) {
      alive_ops.push_back(op);
    }
  });
  const auto phase = CostModelContext::GetInstance()->run_phase();
  const auto device_mem_capacity = CostModelContext::GetInstance()->device_memory_capacity();

  if (alive_ops.size() > 2) {
    if (phase == TRAINING_PHASE) {
      // training phase
      return SearchStrategyForMultiNodeFinalGraph(alive_ops);
    } else {
      // inference phase
      MS_LOG(EXCEPTION)
        << "Currently, searching strategy for the multi-node final graph in inference phase is not supported.";
    }
  } else if (alive_ops.size() == 1) {
    MS_LOG(INFO) << "There are 1 single node in the final graph.";
    OperatorInfoPtr u = alive_ops[0];
    auto cost_list = CreateFinalSingleCostList(u);
    CostPtr cost = nullptr;
    if (phase == TRAINING_PHASE) {
      // training phase
      cost = SelectCostWithMinTrainingTime(cost_list, device_mem_capacity);
    } else {
      // inference phase
      cost = SelectCostWithMinInferenceTime(cost_list, device_mem_capacity);
    }
    if (cost == nullptr) {
      MS_LOG(ERROR) << "No valid strategy can be found under the current device memory: " << device_mem_capacity << ".";
      return FAILED;
    }
    MS_EXCEPTION_IF_NULL(u);
    MS_EXCEPTION_IF_NULL(cost->decision_ptr_);
    auto decision = cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
    MS_EXCEPTION_IF_NULL(decision);
    u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->u_cost_);
    MS_LOG(INFO) << "Searching the strategy for the eliminated final graph ended.";
    return SUCCESS;
  } else {
    return SearchStrategyForTwoNodeFinalGraph(alive_ops);
  }
}

// Given a graph which contains the following subgraph: u --> v --> w, the node v can be eliminated
// return the v and the edge u --> v
OperatorInfoPtr CostGraph::CheckOpElimination() const {
  for (auto &op : ops_) {
    bool bool_test = op->is_alive() && op->GetAliveSuccEdges().size() == 1 && op->GetAlivePrevEdges().size() == 1;
    if (bool_test) {
      if ((op->GetAliveSuccEdges()[0]->next_operator() != op) && (op->GetAlivePrevEdges()[0]->prev_operator() != op)) {
        return op;
      }
    }
  }
  return nullptr;
}

// Check the graph whether an EdgeElimination can be performed
std::vector<std::shared_ptr<Edge>> CostGraph::CheckEdgeElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (!op->is_alive()) {
      continue;
    }
    std::map<void *, int64_t> count;
    for (auto &edge_su : op->GetAliveSuccEdges()) {
      MS_EXCEPTION_IF_NULL(edge_su);
      auto v = edge_su->next_operator();
      count[v.get()]++;
    }
    for (auto &pair : count) {
      auto *op_ptr = pair.first;
      int64_t op_count = pair.second;
      if (op_count > 1) {
        std::vector<std::shared_ptr<Edge>> ret;
        for (auto &edge : op->GetAliveSuccEdges()) {
          MS_EXCEPTION_IF_NULL(edge);
          if (edge->next_operator().get() == op_ptr) {
            ret.push_back(edge);
          }
        }
        return ret;
      }
    }
  }
  return {};
}

// Check the graph whether a MergeElimination can be performed
OperatorInfoPtr CostGraph::CheckMergeElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = op->is_alive() && op->GetAlivePrevEdges().empty() && op->GetAliveSuccEdges().size() == 1;
    if (bool_test) {
      auto next_op = op->GetAliveSuccEdges()[0]->next_operator();
      MS_EXCEPTION_IF_NULL(next_op);
      if (!next_op->GetAlivePrevEdges().empty()) {
        return op;
      }
    }
  }
  return nullptr;
}

// Check the graph whether a ContractElimination can be performed
OperatorInfoPtr CostGraph::CheckContractElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = op->is_alive() && op->GetAlivePrevEdges().size() == 1 && op->GetAliveSuccEdges().empty();
    if (bool_test) {
      auto edge = op->GetAlivePrevEdges()[0];
      MS_EXCEPTION_IF_NULL(edge);
      auto prev_op = edge->prev_operator();
      MS_EXCEPTION_IF_NULL(prev_op);
      if (!prev_op->GetAliveSuccEdges().empty()) {
        return op;
      }
    }
  }
  return nullptr;
}

std::pair<OperatorInfoPtr, OperatorInfoPtr> CostGraph::CheckSourceElimination() const {
  size_t source_count = 0;
  std::vector<OperatorInfoPtr> op_vector(2, nullptr);
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = op->is_alive() && op->GetAlivePrevEdges().empty() && op->GetAliveSuccEdges().size() > 0;
    if (bool_test) {
      op_vector[source_count++] = op;
      if (source_count == 2) {
        return std::make_pair(op_vector[0], op_vector[1]);
      }
    }
  }
  return std::make_pair(nullptr, nullptr);
}

void CostGraph::CreateSourceEliminationSubCostList(StrategyPtr op1_old_stra, const CostPtrList &op1_old_clist,
                                                   StrategyPtr op2_old_stra, const CostPtrList &op2_old_clist,
                                                   CostPtrList *op1_new_clist) const {
  for (auto &op1_cost : op1_old_clist) {
    for (auto &op2_cost : op2_old_clist) {
      double computation = op1_cost->computation_cost_ + op2_cost->computation_cost_;
      double memory = op1_cost->memory_with_reuse_ + op2_cost->memory_with_reuse_;
      double communication = op1_cost->communication_cost_ + op2_cost->communication_cost_;
      double communication_forward = op1_cost->communication_forward_ + op2_cost->communication_forward_;
      double communication_without_para =
        op1_cost->communication_without_parameter_ + op2_cost->communication_without_parameter_;
      auto decision = std::make_shared<SourceEliminationDecision>(op1_old_stra, op1_cost, op2_old_stra, op2_cost);
      auto new_cost = std::make_shared<Cost>(computation, communication, decision);
      const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
      MS_EXCEPTION_IF_NULL(new_cost);
      new_cost->communication_without_parameter_ = communication_without_para;
      new_cost->communication_with_partial_para_ =
        communication_without_para + gamma * (communication - communication_without_para);
      new_cost->memory_with_reuse_ = memory;
      new_cost->communication_forward_ = communication_forward;
      MS_EXCEPTION_IF_NULL(op1_new_clist);
      op1_new_clist->push_back(std::move(new_cost));
    }
  }
}

// Check the graph whether a TriangleElimination can be performed
std::pair<OperatorInfoPtr, std::shared_ptr<Edge>> CostGraph::CheckTriangleElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = (op->is_alive()) && (op->GetAlivePrevEdges().empty()) && (op->GetAliveSuccEdges().size() == 2);
    if (bool_test) {
      auto edge1 = op->GetAliveSuccEdges()[0];
      auto edge2 = op->GetAliveSuccEdges()[1];
      MS_EXCEPTION_IF_NULL(edge1);
      MS_EXCEPTION_IF_NULL(edge2);
      auto first_op = edge1->next_operator();
      auto second_op = edge2->next_operator();
      MS_EXCEPTION_IF_NULL(first_op);
      for (auto &first_op_succ_edge : first_op->GetAliveSuccEdges()) {
        if (first_op_succ_edge->next_operator() == second_op) {
          return {op, first_op_succ_edge};
        }
      }
      MS_EXCEPTION_IF_NULL(second_op);
      for (auto &second_op_succ_edge : second_op->GetAliveSuccEdges()) {
        if (second_op_succ_edge->next_operator() == first_op) {
          return {op, second_op_succ_edge};
        }
      }
    }
  }
  return {nullptr, nullptr};
}

// Check the graph whether a StarElimination can be performed.
// NOTE: this elimination MUST be performed only when the above 5 operation cannot be applied.
OperatorInfoPtr CostGraph::CheckStarElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = (op->is_alive()) && (op->GetAlivePrevEdges().empty()) && (op->GetAliveSuccEdges().size() > 1);
    if (bool_test) {
      return op;
    }
  }
  return nullptr;
}

// This method is for 'eliminating operator' operation in the DP algorithm. It creates a new edge to replace
// 'lefe_edge', 'op' and 'right_edge'. As a consequence, it creates new costlist for the new edge.
std::shared_ptr<Edge> CostGraph::EliminationOp(const OperatorInfoPtr &op) const {
  // in this case, the operators are organised in the form of u-->op-->v, and the goal
  // is to eliminate 'op'.
  MS_EXCEPTION_IF_NULL(op);
  MS_LOG(INFO) << "Now eliminating node: " << op->name() << ".";
  auto edge_u_op = op->GetAlivePrevEdges()[0];
  auto edge_op_v = op->GetAliveSuccEdges()[0];
  MS_EXCEPTION_IF_NULL(edge_u_op);
  MS_EXCEPTION_IF_NULL(edge_op_v);
  auto u = edge_u_op->prev_operator();
  auto v = edge_op_v->next_operator();
  std::vector<size_t> output_indexs;
  std::vector<size_t> input_indexs;
  size_t output_index;
  size_t input_index;
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  std::string new_edge_name = u->name() + OPERATOR_TO_OPERATOR_CONNECTOR + v->name();
  std::shared_ptr<Edge> new_edge;
  if (edge_u_op->is_combined()) {
    output_indexs = edge_u_op->prev_op_output_indexs();
  } else {
    output_index = edge_u_op->prev_op_output_index();
    output_indexs.push_back(output_index);
  }
  if (edge_op_v->is_combined()) {
    input_indexs = edge_op_v->next_op_input_indexs();
  } else {
    input_index = edge_op_v->next_op_input_index();
    input_indexs.push_back(input_index);
  }

  if (!edge_u_op->is_combined() && !edge_op_v->is_combined()) {
    new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_index, input_index, false);
  } else {
    new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_indexs, input_indexs, true);
  }
  MS_EXCEPTION_IF_NULL(new_edge);
  new_edge->set_pre_op_output(edge_u_op->prev_op_output());
  new_edge->set_next_op_input(edge_op_v->next_op_input());
  new_edge->OpEliminationSetNewCost(edge_u_op, op, edge_op_v);
  u->ReplaceSuccEdge(op, new_edge);
  v->ReplacePreEdge(op, new_edge);
  op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating node: " << op->name() << " succeeded.";
  return new_edge;
}

// This method is for 'eliminating edges' operation in the DP algorithm. It creates a new edge to replace the 'edges',
// and sets new costlist for the new edge.
std::shared_ptr<Edge> CostGraph::EliminationEdges(const std::vector<std::shared_ptr<Edge>> &edges) const {
  MS_LOG(INFO) << "Now eliminating " << edges.size() << " edges.";
  MS_EXCEPTION_IF_NULL(edges[0]);
  auto u = edges[0]->prev_operator();
  auto v = edges[0]->next_operator();
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  std::string new_edge_name = u->name() + OPERATOR_TO_OPERATOR_CONNECTOR + v->name();
  std::vector<size_t> output_indexs;
  std::vector<size_t> input_indexs;

  for (auto &edge : edges) {
    MS_EXCEPTION_IF_NULL(edge);
    if (edge->is_combined()) {
      auto from_output_indexs = edge->prev_op_output_indexs();
      auto from_input_indexs = edge->next_op_input_indexs();
      (void)std::copy(from_output_indexs.begin(), from_output_indexs.end(), std::back_inserter(output_indexs));
      (void)std::copy(from_input_indexs.begin(), from_input_indexs.end(), std::back_inserter(input_indexs));
    } else {
      output_indexs.push_back(edge->prev_op_output_index());
      input_indexs.push_back(edge->next_op_input_index());
    }
  }

  std::shared_ptr<Edge> new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_indexs, input_indexs, true);
  MS_EXCEPTION_IF_NULL(new_edge);
  new_edge->set_pre_op_output(edges[0]->prev_op_output());
  new_edge->set_next_op_input(edges[0]->next_op_input());

  new_edge->EdgeEliminationSetNewCost(u, edges, v);

  u->ReplaceSuccEdges(v, new_edge);
  v->ReplacePreEdges(u, new_edge);
  MS_LOG(INFO) << "Eliminating " << edges.size() << " edges succeeded.";
  return new_edge;
}

// Given 'op_cost_list', 'edge_cost_list', and 'tar_cost_list', this method is to create 'tar_cost_list_new'
// for this contract under the strategy 'op_strategy'
void CostGraph::CreateMergeEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList &op_cost_list,
                                                  const CostPtrList &edge_cost_list, StrategyPtr tar_op_strategy,
                                                  const CostPtrList &tar_cost_list,
                                                  CostPtrList *const tar_cost_list_new) const {
  for (size_t i = 0; i < op_cost_list.size(); ++i) {
    auto &op_cost = op_cost_list[i];
    MS_EXCEPTION_IF_NULL(op_cost);
    for (size_t j = 0; j < edge_cost_list.size(); ++j) {
      auto &edge_cost = edge_cost_list[j];
      MS_EXCEPTION_IF_NULL(edge_cost);
      for (size_t k = 0; k < tar_cost_list.size(); ++k) {
        auto &tar_cost = tar_cost_list[k];
        MS_EXCEPTION_IF_NULL(tar_cost);
        double computation = op_cost->computation_cost_ + edge_cost->computation_cost_ + tar_cost->computation_cost_;
        double memory = op_cost->memory_with_reuse_ + edge_cost->memory_with_reuse_ + tar_cost->memory_with_reuse_;
        double communication =
          op_cost->communication_cost_ + edge_cost->communication_cost_ + tar_cost->communication_cost_;
        double communication_forward =
          op_cost->communication_forward_ + edge_cost->communication_forward_ + tar_cost->communication_forward_;
        double communication_without_para = op_cost->communication_without_parameter_ +
                                            edge_cost->communication_without_parameter_ +
                                            tar_cost->communication_without_parameter_;

        auto decision =
          std::make_shared<MergeEliminationDecision>(op_strategy, op_cost, edge_cost, tar_op_strategy, tar_cost);
        auto new_cost = std::make_shared<Cost>(computation, communication, decision);
        const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
        MS_EXCEPTION_IF_NULL(new_cost);
        new_cost->communication_without_parameter_ = communication_without_para;
        new_cost->communication_with_partial_para_ =
          communication_without_para + gamma * (communication - communication_without_para);
        new_cost->memory_with_reuse_ = memory;
        new_cost->communication_forward_ = communication_forward;
        MS_EXCEPTION_IF_NULL(tar_cost_list_new);
        tar_cost_list_new->push_back(std::move(new_cost));
      }
    }
  }
}

// This method is for the 'Merge' operation in DP algorithm. It creates new costlist for each strategy in the
// target_op
OperatorInfoPtr CostGraph::EliminationMerge(const OperatorInfoPtr &op) const {
  MS_EXCEPTION_IF_NULL(op);
  auto target_op = op->GetAliveSuccEdges()[0]->next_operator();
  auto edge_ptr = op->GetAliveSuccEdges()[0];
  MS_EXCEPTION_IF_NULL(target_op);
  MS_EXCEPTION_IF_NULL(edge_ptr);
  MS_LOG(INFO) << "Now merging " << op->name() << " into " << target_op->name() << ".";
  bool valid = false;

  for (auto &tar_stra_cost : target_op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(tar_stra_cost);
    auto tar_stra = tar_stra_cost->strategy_ptr;
    auto tar_clist_origin = tar_stra_cost->cost_list;
    CostPtrList tar_clist_new;

    for (auto &op_stra_cost : op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(op_stra_cost);
      auto op_stra = op_stra_cost->strategy_ptr;
      auto op_clist = op_stra_cost->cost_list;
      auto edge_clist = edge_ptr->GetCostList(op_stra, tar_stra);

      CreateMergeEliminationSubCostList(op_stra, op_clist, edge_clist, tar_stra, tar_clist_origin, &tar_clist_new);
    }
    Simplify(&tar_clist_new);
    // Set the new costlist w.r.t the strategy
    tar_stra_cost->cost_list = tar_clist_new;
    if ((!valid) && (!tar_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Merging " << op->name() << " into " << target_op->name() << " failed.";
  }
  op->SetNotAlive();
  MS_LOG(INFO) << "Merging " << op->name() << " into " << target_op->name() << " succeeded.";
  return target_op;
}

// Given 'contract_op_cost_list', 'edge_cost_list', and 'tar_cost_list', this method is to create 'tar_cost_list_new'
// for this contract under the strategy 'contract_op_stra'
void CostGraph::CreateContractEliminationSubCostList(StrategyPtr contract_op_stra,
                                                     const CostPtrList &contract_op_cost_list,
                                                     const CostPtrList &edge_cost_list, StrategyPtr target_op_stra,
                                                     const CostPtrList &tar_cost_list,
                                                     CostPtrList *tar_cost_list_new) const {
  for (size_t i = 0; i < contract_op_cost_list.size(); ++i) {
    auto &contract_op_cost = contract_op_cost_list[i];
    MS_EXCEPTION_IF_NULL(contract_op_cost);
    for (size_t j = 0; j < edge_cost_list.size(); ++j) {
      auto &edge_cost = edge_cost_list[j];
      MS_EXCEPTION_IF_NULL(edge_cost);
      for (size_t k = 0; k < tar_cost_list.size(); ++k) {
        auto &tar_cost = tar_cost_list[k];
        MS_EXCEPTION_IF_NULL(tar_cost);
        double computation =
          contract_op_cost->computation_cost_ + edge_cost->computation_cost_ + tar_cost->computation_cost_;
        double memory =
          contract_op_cost->memory_with_reuse_ + edge_cost->memory_with_reuse_ + tar_cost->memory_with_reuse_;
        double communication =
          contract_op_cost->communication_cost_ + edge_cost->communication_cost_ + tar_cost->communication_cost_;
        double communication_forward = contract_op_cost->communication_forward_ + edge_cost->communication_forward_ +
                                       tar_cost->communication_forward_;
        double communication_without_para = contract_op_cost->communication_without_parameter_ +
                                            edge_cost->communication_without_parameter_ +
                                            tar_cost->communication_without_parameter_;

        auto decision = std::make_shared<ContractEliminationDecision>(contract_op_stra, contract_op_cost, edge_cost,
                                                                      target_op_stra, tar_cost);
        auto new_cost = std::make_shared<Cost>(computation, communication, decision);
        auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
        new_cost->communication_without_parameter_ = communication_without_para;
        new_cost->communication_with_partial_para_ =
          communication_without_para + gamma * (communication - communication_without_para);
        new_cost->memory_with_reuse_ = memory;
        new_cost->communication_forward_ = communication_forward;
        tar_cost_list_new->push_back(std::move(new_cost));
      }
    }
  }
}

// This method is for the 'Contract' operation in DP algorithm. It creates new costlist for each strategy in the
// target_op
OperatorInfoPtr CostGraph::EliminationContract(const OperatorInfoPtr &op) const {
  MS_EXCEPTION_IF_NULL(op);
  auto target_op = op->GetAlivePrevEdges()[0]->prev_operator();
  auto edge_ptr = op->GetAlivePrevEdges()[0];
  MS_LOG(INFO) << "Now contracting " << op->name() << " into " << target_op->name() << ".";
  bool valid = false;

  for (auto &tar_stra_cost : target_op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(tar_stra_cost);
    auto tar_stra = tar_stra_cost->strategy_ptr;
    auto tar_clist_origin = tar_stra_cost->cost_list;
    CostPtrList tar_clist_new;

    for (auto &op_stra_cost : op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(op_stra_cost);
      auto op_stra = op_stra_cost->strategy_ptr;
      auto op_clist = op_stra_cost->cost_list;
      auto edge_clist = edge_ptr->GetCostList(tar_stra, op_stra);

      CreateContractEliminationSubCostList(op_stra, op_clist, edge_clist, tar_stra, tar_clist_origin, &tar_clist_new);
    }
    Simplify(&tar_clist_new);
    // Set the new costlist w.r.t the strategy
    tar_stra_cost->cost_list = tar_clist_new;
    if ((!valid) && (!tar_clist_new.empty())) {
      valid = true;
    }
  }
  if (!valid) {
    MS_LOG(EXCEPTION) << "Contracting " << op->name() << " into " << target_op->name() << " failed.";
  }
  op->SetNotAlive();
  MS_LOG(INFO) << "Contracting " << op->name() << " into " << target_op->name() << " succeeded.";
  return target_op;
}

void CostGraph::CreateTriangleEliminationSubCostList(StrategyPtr elimi_op_stra, StrategyPtr left_op_stra,
                                                     StrategyPtr right_op_stra, const CostPtr &right_op_cost,
                                                     const CostPtrList &elimi_op_clist,
                                                     const CostPtrList &left_edge_clist, const CostPtr &right_edge_cost,
                                                     const CostPtrList &left_node_clist_origin,
                                                     CostPtrList *left_node_clist_new) const {
  MS_EXCEPTION_IF_NULL(right_edge_cost);
  MS_EXCEPTION_IF_NULL(right_op_cost);
  MS_EXCEPTION_IF_NULL(left_node_clist_new);
  for (auto &elimi_op_cost : elimi_op_clist) {
    MS_EXCEPTION_IF_NULL(elimi_op_cost);
    for (auto &left_edge_cost : left_edge_clist) {
      MS_EXCEPTION_IF_NULL(left_edge_cost);
      for (auto &left_node_cost : left_node_clist_origin) {
        MS_EXCEPTION_IF_NULL(left_node_cost);
        double new_computation = elimi_op_cost->computation_cost_ + left_edge_cost->computation_cost_ +
                                 left_node_cost->computation_cost_ + right_edge_cost->computation_cost_;
        double new_memory = elimi_op_cost->memory_with_reuse_ + left_edge_cost->memory_with_reuse_ +
                            left_node_cost->memory_with_reuse_ + right_edge_cost->memory_with_reuse_;
        double new_commu_cost = elimi_op_cost->communication_cost_ + left_edge_cost->communication_cost_ +
                                left_node_cost->communication_cost_ + right_edge_cost->communication_cost_;
        double new_commu_forward = elimi_op_cost->communication_forward_ + left_edge_cost->communication_forward_ +
                                   left_node_cost->communication_forward_ + right_edge_cost->communication_forward_;
        double new_commu_without =
          elimi_op_cost->communication_without_parameter_ + left_edge_cost->communication_without_parameter_ +
          left_node_cost->communication_without_parameter_ + right_edge_cost->communication_without_parameter_;
        const auto triangle_star_stra_overwrite = CostModelContext::GetInstance()->triangle_star_strategy_overwrite();
        const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();

        if (triangle_star_stra_overwrite) {
          new_computation += right_op_cost->computation_cost_;
          new_memory += right_op_cost->memory_with_reuse_;
          new_commu_cost += right_op_cost->communication_cost_;
          new_commu_forward += right_op_cost->communication_forward_;
          new_commu_without += right_op_cost->communication_without_parameter_;
        }

        auto decision =
          std::make_shared<TriangleEliminationDecision>(elimi_op_stra, elimi_op_cost, left_edge_cost, right_edge_cost,
                                                        left_op_stra, left_node_cost, right_op_stra, right_op_cost);
        auto new_cost = std::make_shared<Cost>(new_computation, new_commu_cost, decision);
        new_cost->communication_without_parameter_ = new_commu_without;
        new_cost->communication_with_partial_para_ = new_commu_without + gamma * (new_commu_cost - new_commu_without);
        new_cost->memory_with_reuse_ = new_memory;
        new_cost->communication_forward_ = new_commu_forward;
        left_node_clist_new->push_back(std::move(new_cost));
      }
    }
  }
}

void CostGraph::CreateTriangleEliminationCostList(const OperatorInfoPtr &elimi_op, const CostPtrList &right_node_clist,
                                                  const CostPtrList &right_edge_clist, const StrategyPtr &elimi_op_stra,
                                                  const StrategyPtr &left_node_stra, const StrategyPtr &right_node_stra,
                                                  const CostPtrList &elimi_op_clist, const CostPtrList &left_edge_clist,
                                                  const CostPtrList &left_node_clist_origin,
                                                  CostPtrList *left_node_clist_new) const {
  MS_EXCEPTION_IF_NULL(elimi_op);
  for (auto &right_node_cost : right_node_clist) {
    MS_EXCEPTION_IF_NULL(right_node_cost);
    for (auto &right_edge_cost : right_edge_clist) {
      MS_EXCEPTION_IF_NULL(right_edge_cost);
      CreateTriangleEliminationSubCostList(elimi_op_stra, left_node_stra, right_node_stra, right_node_cost,
                                           elimi_op_clist, left_edge_clist, right_edge_cost, left_node_clist_origin,
                                           left_node_clist_new);
    }
  }
}

OperatorInfoPtr CostGraph::EliminationTriangle(const OperatorInfoPtr &elimi_op,
                                               const std::shared_ptr<Edge> &edge_left_right) const {
  MS_EXCEPTION_IF_NULL(edge_left_right);
  MS_EXCEPTION_IF_NULL(elimi_op);
  MS_LOG(INFO) << "Now eliminating triangle: " << elimi_op->name() << ".";
  auto left_node = edge_left_right->prev_operator();
  auto right_node = edge_left_right->next_operator();
  auto left_edge = elimi_op->GetAliveSuccEdges()[0];
  auto right_edge = elimi_op->GetAliveSuccEdges()[1];
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  MS_EXCEPTION_IF_NULL(left_edge);
  MS_EXCEPTION_IF_NULL(right_edge);
  MS_LOG(INFO) << "The left operator is: " << left_node->name() << ".";
  MS_LOG(INFO) << "The right operator is: " << right_node->name() << ".";

  if (left_edge->next_operator() != left_node) {
    auto tmp = left_edge;
    left_edge = right_edge;
    right_edge = tmp;
  }
  bool valid = false;

  for (auto &left_node_stra_cost : left_node->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(left_node_stra_cost);
    auto left_node_stra = left_node_stra_cost->strategy_ptr;
    auto left_node_clist_origin = left_node_stra_cost->cost_list;
    CostPtrList left_node_clist_new;

    for (auto &elimi_op_stra_cost : elimi_op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(elimi_op_stra_cost);
      auto elimi_op_stra = elimi_op_stra_cost->strategy_ptr;
      auto elimi_op_clist = elimi_op_stra_cost->cost_list;
      auto left_edge_clist = left_edge->GetCostList(elimi_op_stra, left_node_stra);

      for (auto &right_node_stra_cost : right_node->GetStrategyCost()) {
        MS_EXCEPTION_IF_NULL(right_node_stra_cost);
        auto right_node_stra = right_node_stra_cost->strategy_ptr;
        auto right_node_clist = right_node_stra_cost->cost_list;
        auto right_edge_clist = right_edge->GetCostList(elimi_op_stra, right_node_stra);

        CreateTriangleEliminationCostList(elimi_op, right_node_clist, right_edge_clist, elimi_op_stra, left_node_stra,
                                          right_node_stra, elimi_op_clist, left_edge_clist, left_node_clist_origin,
                                          &left_node_clist_new);
      }
    }
    Simplify(&left_node_clist_new);
    // Set the new costlist w.r.t the strategy
    left_node_stra_cost->cost_list = left_node_clist_new;
    if ((!valid) && (!left_node_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Eliminating triangle: " << elimi_op->name()
                      << " failed. It may be caused by "
                         "configuring inconsistent strategies for operators.";
  }
  elimi_op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating triangle: " << elimi_op->name() << " succeeded.";
  return left_node;
}

void CostGraph::CreateStarEliminationSubCostList(const StrategyPtr &first_succ_node_stra,
                                                 const CostPtrList &first_succ_node_clist,
                                                 const CostPtrList &first_succ_edge_clist,
                                                 const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                                 std::vector<StrategyPtr> succ_nodes_stras,
                                                 CostPtrList &succ_edges_costs, CostPtrList &succ_nodes_costs,
                                                 CostPtrList *first_succ_node_clist_new) const {
  for (auto &first_succ_node_cost : first_succ_node_clist) {
    for (auto &first_succ_edge_cost : first_succ_edge_clist) {
      for (auto &merged_node_cost : merged_op_clist) {
        MS_EXCEPTION_IF_NULL(merged_node_cost);
        succ_nodes_stras[0] = first_succ_node_stra;
        succ_edges_costs[0] = first_succ_edge_cost;
        succ_nodes_costs[0] = first_succ_node_cost;

        double computation_cost = merged_node_cost->computation_cost_,
               memory_cost = merged_node_cost->memory_with_reuse_, commu_cost = merged_node_cost->communication_cost_,
               commu_without = merged_node_cost->communication_without_parameter_,
               commu_forward = merged_node_cost->communication_forward_;
        const auto triangle_star_stra_overwrite = CostModelContext::GetInstance()->triangle_star_strategy_overwrite();
        const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
        for (size_t i = 0; i < succ_nodes_stras.size(); ++i) {
          MS_EXCEPTION_IF_NULL(succ_edges_costs[i]);
          if (i == 0) {
            computation_cost += succ_edges_costs[i]->computation_cost_ + succ_nodes_costs[i]->computation_cost_;
            memory_cost += succ_edges_costs[i]->memory_with_reuse_ + succ_nodes_costs[i]->memory_with_reuse_;
            commu_cost += succ_edges_costs[i]->communication_cost_ + succ_nodes_costs[i]->communication_cost_;
            commu_forward += succ_edges_costs[i]->communication_forward_ + succ_nodes_costs[i]->communication_forward_;
            commu_without += succ_edges_costs[i]->communication_without_parameter_ +
                             succ_nodes_costs[i]->communication_without_parameter_;
          } else {
            computation_cost += succ_edges_costs[i]->computation_cost_;
            memory_cost += succ_edges_costs[i]->memory_with_reuse_;
            commu_cost += succ_edges_costs[i]->communication_cost_;
            commu_forward += succ_edges_costs[i]->communication_forward_;
            commu_without += succ_edges_costs[i]->communication_without_parameter_;
            if (triangle_star_stra_overwrite) {
              computation_cost += succ_nodes_costs[i]->computation_cost_;
              memory_cost += succ_nodes_costs[i]->memory_with_reuse_;
              commu_cost += succ_nodes_costs[i]->communication_cost_;
              commu_forward += succ_nodes_costs[i]->communication_forward_;
              commu_without += succ_nodes_costs[i]->communication_without_parameter_;
            }
          }
        }

        auto decision = std::make_shared<StarEliminationDecision>(merged_op_stra, merged_node_cost, succ_edges_costs,
                                                                  succ_nodes_stras, succ_nodes_costs);
        auto new_cost = std::make_shared<Cost>(computation_cost, commu_cost, decision);
        new_cost->communication_without_parameter_ = commu_without;
        new_cost->communication_with_partial_para_ = commu_without + gamma * (commu_cost - commu_without);
        new_cost->memory_with_reuse_ = memory_cost;
        new_cost->communication_forward_ = commu_forward;
        first_succ_node_clist_new->push_back(std::move(new_cost));
      }
    }
  }
}

void CostGraph::CreateStarEliminationCostList(std::vector<std::shared_ptr<Edge>> &succ_edges,
                                              const StrategyPtr &first_succ_node_stra,
                                              const CostPtrList &first_succ_node_clist,
                                              const CostPtrList &first_succ_edge_clist,
                                              const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                              CostPtrList *first_succ_node_clist_new) const {
  std::vector<StrategyPtr> succ_nodes_stras(succ_edges.size(), nullptr);
  CostPtrList succ_edges_costs(succ_edges.size(), nullptr);
  CostPtrList succ_nodes_costs(succ_edges.size(), nullptr);
  std::function<void(size_t)> recursive = [&first_succ_node_stra, &first_succ_node_clist, &first_succ_edge_clist,
                                           &merged_op_stra, &merged_op_clist, &succ_nodes_stras, &succ_edges_costs,
                                           &succ_nodes_costs, &first_succ_node_clist_new, &succ_edges, &recursive,
                                           this](size_t k) {
    if (k == succ_edges.size()) {
      CreateStarEliminationSubCostList(first_succ_node_stra, first_succ_node_clist, first_succ_edge_clist,
                                       merged_op_stra, merged_op_clist, succ_nodes_stras, succ_edges_costs,
                                       succ_nodes_costs, first_succ_node_clist_new);
      return;
    }
    MS_LOG(DEBUG) << "The size of first_succ_node_clist: " << first_succ_node_clist.size()
                  << ", first_succ_edge_clist: " << first_succ_edge_clist.size()
                  << ", merged_op_clist: " << merged_op_clist.size()
                  << ", first_succ_node_clist_new: " << first_succ_node_clist_new->size() << ".";
    auto succ_edge = succ_edges[k];
    MS_EXCEPTION_IF_NULL(succ_edge);
    auto succ_node = succ_edge->next_operator();
    MS_EXCEPTION_IF_NULL(succ_node);
    for (auto &succ_node_stra_cost : succ_node->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(succ_node_stra_cost);
      auto succ_node_stra = succ_node_stra_cost->strategy_ptr;
      auto succ_node_clist = succ_node_stra_cost->cost_list;
      auto succ_edge_clist = succ_edge->GetCostList(merged_op_stra, succ_node_stra);

      for (auto &succ_node_cost : succ_node_clist) {
        MS_EXCEPTION_IF_NULL(succ_node_cost);
        for (auto &succ_edge_cost : succ_edge_clist) {
          MS_EXCEPTION_IF_NULL(succ_edge_cost);
          succ_nodes_stras[k] = succ_node_stra;
          succ_edges_costs[k] = succ_edge_cost;
          succ_nodes_costs[k] = succ_node_cost;
          recursive(k + 1);
        }
      }
    }
  };

  recursive(1);
}

std::vector<std::shared_ptr<Edge>> CostGraph::EliminationStar(const OperatorInfoPtr &merged_op) const {
  MS_EXCEPTION_IF_NULL(merged_op);
  auto succ_edges = merged_op->GetAliveSuccEdges();
  MS_LOG(INFO) << "Now eliminating star centered at: " << merged_op->name() << ".";
  for (auto &succ_edge : succ_edges) {
    MS_EXCEPTION_IF_NULL(succ_edge->next_operator());
    MS_LOG(INFO) << "The successive operator is: " << succ_edge->next_operator()->name() << ".";
  }

  MS_EXCEPTION_IF_NULL(succ_edges[0]);
  auto first_succ_node = succ_edges[0]->next_operator();
  auto first_succ_edge = succ_edges[0];
  bool valid = false;

  // 'merged_op' is merged into first_node
  MS_EXCEPTION_IF_NULL(first_succ_node);
  for (auto &first_succ_node_stra_cost : first_succ_node->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(first_succ_node_stra_cost);
    auto first_succ_node_stra = first_succ_node_stra_cost->strategy_ptr;
    auto first_succ_node_clist = first_succ_node_stra_cost->cost_list;
    CostPtrList first_succ_node_clist_new;

    for (auto &merged_op_stra_cost : merged_op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(merged_op_stra_cost);
      auto merged_op_stra = merged_op_stra_cost->strategy_ptr;
      auto merged_op_clist = merged_op_stra_cost->cost_list;
      auto first_succ_edge_clist = first_succ_edge->GetCostList(merged_op_stra, first_succ_node_stra);

      CreateStarEliminationCostList(succ_edges, first_succ_node_stra, first_succ_node_clist, first_succ_edge_clist,
                                    merged_op_stra, merged_op_clist, &first_succ_node_clist_new);
    }
    Simplify(&first_succ_node_clist_new);
    // Set the new costlist w.r.t the strategy
    first_succ_node_stra_cost->cost_list = first_succ_node_clist_new;
    if ((!valid) && (!first_succ_node_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Eliminating star centered at: " << merged_op->name()
                      << " failed. It may be caused by "
                         "configuring inconsistent strategies for operators.";
  }

  merged_op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating star centered at: " << merged_op->name() << " succeeded.";
  return succ_edges;
}

size_t CostGraph::GetNumEdges() const {
  size_t sum = 0;
  for (const auto &kv : edges_) {
    auto &edges = kv.second;
    sum += edges.size();
  }
  return sum;
}

Status CostGraph::InitReshapeStrategy() {
  // reshape init should be apply after the init of it's previous node and next node.
  for (size_t i = 0; i < ops_.size(); ++i) {
    if (ops_[i]->IsReshape()) {
      auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(ops_[i]);
      auto in_edges = GetOriginalPrevEdges(ops_[i]);
      auto out_edges = GetOriginalNextEdges(ops_[i]);
      // deal with consecutive reshape op
      auto pre_reshape_iter = std::find_if(in_edges.begin(), in_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->prev_operator()->IsReshape();
      });
      auto next_reshape_iter = std::find_if(out_edges.begin(), out_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->next_operator()->IsReshape();
      });
      if (pre_reshape_iter != in_edges.end() || next_reshape_iter != out_edges.end()) {
        reshape_info->set_strategy(nullptr);
        continue;
      }
      auto pre_iter = std::find_if(in_edges.begin(), in_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->prev_operator()->name() == reshape_info->pre_operator_name();
      });
      auto next_iter = std::find_if(out_edges.begin(), out_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->next_operator()->name() == reshape_info->next_operator_name();
      });
      bool reshape_is_first_op = reshape_info->pre_operator_name() == reshape_info->name();
      if (reshape_is_first_op) {
        (void)reshape_info->InitSelectedStrategy(reshape_info->selected_strategy(), nullptr);
      }
      if (pre_iter != in_edges.end() || reshape_is_first_op) {
        MS_LOG(DEBUG) << "Set reshape input layout by " << reshape_info->pre_operator_name();
        int64_t pre_index = reshape_info->pre_operator_index();
        TensorInfo pre_info;
        std::shared_ptr<OperatorInfo> pre_op_info;
        if (reshape_is_first_op) {
          pre_op_info = reshape_info;
          pre_info = pre_op_info->inputs_tensor_info()[LongToSize(pre_index)];
        } else {
          pre_op_info = (*pre_iter)->prev_operator();
          pre_info = pre_op_info->outputs_tensor_info()[LongToSize(pre_index)];
        }
        reshape_info->SetInputLayout(pre_info.tensor_layout());
        if (pre_iter != in_edges.end()) {
          Dimensions stra = pre_info.InferStrategy();
          if (stra.empty()) {
            MS_LOG(EXCEPTION) << "Infer strategy by tensor_info failed";
          }
          Strategies stra_inputs = {stra};
          StrategyPtr reshape_stra =
            std::make_shared<Strategy>((*pre_iter)->prev_operator()->strategy()->GetInputStage(), stra_inputs);
          reshape_info->set_strategy(reshape_stra);
        }
      }
      if (next_iter != out_edges.end()) {
        MS_LOG(DEBUG) << "Set reshape output layout by " << reshape_info->next_operator_name();
        int64_t next_index = reshape_info->next_operator_index();
        reshape_info->SetOutputLayout(
          (*next_iter)->next_operator()->inputs_tensor_info()[LongToSize(next_index)].tensor_layout());
      }
      if (reshape_info->Init(nullptr, nullptr) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::InitSelectedStrategy() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->IsReshape()) {
      continue;
    }
    if (op->is_config_by_layout()) {
      op->set_auto_parallel(false);
      continue;
    }
    StrategyPtr out_strategy = nullptr;
    if (op->out_strategy() != nullptr) {
      out_strategy = op->out_strategy();
    }
    auto result_op = op->InitSelectedStrategy(op->selected_strategy(), out_strategy);
    if (result_op != SUCCESS) {
      return result_op;
    }
  }

  return SUCCESS;
}

Status CostGraph::ComputeOpsAndEdgesParameterInvolved() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    const auto &output_parameter = op->ComputeOpAndPrevEdgeParameterInvolved();
    if ((output_parameter != 0) && (output_parameter != 1)) {
      MS_LOG(ERROR) << "Computing parameter_involved for " << op->name() << " failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

void CostGraph::DFSForTopoOrder(const OperatorInfoPtr &current_op, std::vector<OperatorInfoPtr> *topo_order) {
  MS_EXCEPTION_IF_NULL(current_op);
  MS_EXCEPTION_IF_NULL(topo_order);

  visited.at(current_op) = true;
  for (const auto &s_edge : current_op->succ_edges()) {
    if (!visited.at(s_edge->next_operator())) {
      DFSForTopoOrder(s_edge->next_operator(), topo_order);
    }
  }
  topo_order->push_back(current_op);
}

// Compute a topological order of the costgraph
void CostGraph::TopologyOrder(std::vector<OperatorInfoPtr> *topo_order) {
  for (auto &op : ops_) {
    visited[op] = false;
  }

  for (auto &op : ops_) {
    if (!visited[op]) {
      DFSForTopoOrder(op, topo_order);
    }
  }
}
void CostGraph::MarkCriticalOpsAndEdges(const std::map<OperatorInfoPtr, int64_t> &candidate_ops) {
  for (auto &op : ops_) {
    auto search = candidate_ops.find(op);
    if (search != candidate_ops.end()) {
      // Mark the critical operators
      op->mark_output_critical();
      // Mark the successive edges
      for (auto &s_edge : op->succ_edges()) {
        s_edge->mark_output_critical();
      }
    } else {
      op->mark_output_not_critical();
    }
  }
}

Status CostGraph::DetermineCriticalOps(const std::vector<OperatorInfoPtr> &topo_order) {
  if (topo_order.size() == 0) {
    MS_LOG(ERROR) << "0 operator in costgraph.";
    return FAILED;
  }
  auto &first_op = topo_order[0];
  if (first_op->prev_edges().size() > 0) {
    MS_LOG(ERROR) << "The first operator in the first of topological order of "
                     "costgraph should have 0 incoming edge, but has "
                  << first_op->prev_edges() << "edges.";
    return FAILED;
  }
  // The 'curr_memory_state' records <OperatorInfo, remaining_output_cnt>, where remaining_output_cnt is the number
  // of the output of OperatorInfo that currently has not been used
  std::map<OperatorInfoPtr, int64_t> curr_memory_state;
  (void)curr_memory_state.emplace(std::make_pair(first_op, SizeToLong(first_op->succ_edges().size())));
  std::map<OperatorInfoPtr, int64_t> max_memory_state = curr_memory_state;
  // The 'curr_memory_size' records the current total memory size, which is the sum of outputs of operators that has
  // not been used
  double curr_memory_size = first_op->GetOutputsTotalSize();
  double max_memory_size = curr_memory_size;

  for (size_t finished = 1; finished < topo_order.size(); ++finished) {
    // Produce
    (void)curr_memory_state.emplace(
      std::make_pair(topo_order[finished], SizeToLong(topo_order[finished]->succ_edges().size())));
    curr_memory_size += topo_order[finished]->GetOutputsTotalSize();
    // Consume
    for (const auto &prev_edge : topo_order[finished]->prev_edges()) {
      const auto &prev_op = prev_edge->prev_operator();
      curr_memory_state[prev_op]--;
    }
    for (const auto &prev_edge : topo_order[finished]->prev_edges()) {
      const auto &prev_op = prev_edge->prev_operator();
      if (curr_memory_state[prev_op] < 0) {
        MS_LOG(ERROR) << "Failure: " << prev_op->name() << "'s current output count: " << curr_memory_state[prev_op];
        return FAILED;
      } else if (curr_memory_state[prev_op] == 0) {
        (void)curr_memory_state.erase(prev_op);
        curr_memory_size -= prev_op->GetOutputsTotalSize();
      }
    }

    if (curr_memory_size < 0) {
      MS_LOG(ERROR) << "Memory size calculation failed: " << curr_memory_size;
    }
    // Modify the max
    if (curr_memory_size > max_memory_size) {
      max_memory_size = curr_memory_size;
      max_memory_state = curr_memory_state;
    }
  }
  // Mark those critical operators
  MarkCriticalOpsAndEdges(max_memory_state);
  return SUCCESS;
}

Status CostGraph::ComputeOpsAndEdgesOutputCritical() {
  // Two steps to do:
  // 1. Compute a topological order of the costgraph
  // 2. Determine and mark the operators (and necessary edges) that are critical
  std::vector<OperatorInfoPtr> topo_order;
  TopologyOrder(&topo_order);
  std::reverse(std::begin(topo_order), std::end(topo_order));

  if (DetermineCriticalOps(topo_order) != SUCCESS) {
    MS_LOG(ERROR) << "Determining critical operators failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status CostGraph::CalculateOpsMemoryCost() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->CalculateMemoryCost() != SUCCESS) {
      MS_LOG(ERROR) << "Calculate Operator: " << op->name() << " cost for memory usage failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateOpsMemoryCostForInference() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->CalculateMemoryCostForInference() != SUCCESS) {
      MS_LOG(ERROR) << "Calculate Operator: " << op->name() << " cost for memory usage failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateEdgesMemoryCost() {
  for (const auto &edge_pair : edges_) {
    const auto &edges = edge_pair.second;
    for (auto &one_edge : edges) {
      if (one_edge->CalculateMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculate Edge: " << one_edge->edge_name() << " cost for memory usage failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateEdgesMemoryCostForInference() {
  for (const auto &edge_pair : edges_) {
    const auto &edges = edge_pair.second;
    for (auto &one_edge : edges) {
      if (one_edge->CalculateMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculate Edge: " << one_edge->edge_name() << " cost for memory usage failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

OperatorInfoPtr CostGraph::FindTmpIdentityByParameterName(const std::string &p_name) const {
  for (auto one_op : ops_) {
    if (one_op->name().find(IDENTITY_INFO) != std::string::npos) {
      if (one_op->refkey_parameter_name() == p_name) {
        return one_op;
      }
    }
  }
  return nullptr;
}
Status CostGraph::CorrectOpsMemoryCost() {
  for (auto &one_op : ops_) {
    if ((one_op->name().find(IDENTITY_INFO) != std::string::npos) && (one_op->is_output_parameter_involve() == 1)) {
      if (one_op->GetAliveSuccEdges().size() > 1) {
        // Filter out the case when the TmpIdentity being used by multiple operators
        std::map<size_t, int64_t> output_count;
        for (size_t i = 0; i < one_op->GetAliveSuccEdges().size(); ++i) {
          auto output_index = one_op->GetAliveSuccEdges()[i]->prev_op_output_index();
          output_count[output_index]++;
        }
        for (size_t i = 0; i < one_op->GetAliveSuccEdges().size(); ++i) {
          auto output_index = one_op->GetAliveSuccEdges()[i]->prev_op_output_index();
          if (output_count[output_index] <= 1) {
            continue;
          }
          auto next_op = one_op->GetAliveSuccEdges()[i]->next_operator();
          MS_EXCEPTION_IF_NULL(next_op);
          auto input_index = one_op->GetAliveSuccEdges()[i]->next_op_input_index();
          if (next_op->CorrectMemoryCost(input_index) != SUCCESS) {
            MS_LOG(ERROR) << "The operator name: " << one_op->name() << ", the next operator name: " << next_op->name()
                          << ", the output_index: " << output_index << ", the input_index: " << input_index << ".";
            return FAILED;
          }
          output_count[output_index]--;
        }
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateMemoryCost() {
  const auto phase = CostModelContext::GetInstance()->run_phase();
  if (phase == TRAINING_PHASE) {
    // training phase
    if (ComputeOpsAndEdgesParameterInvolved() == SUCCESS) {
      // Calculate operators' memory usage
      if (CalculateOpsMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' cost for memory cost failed.";
        return FAILED;
      }
      // Calculate edges' memory usage
      if (CalculateEdgesMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating edges' cost for memory cost failed.";
        return FAILED;
      }
      // Correct memory usage caused by TmpIdentity
      if (CorrectOpsMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Correcting operators' cost for memory cost failed.";
        return FAILED;
      }
    } else {
      MS_LOG(ERROR) << "Computing operators' parameter_involved failed.";
      return FAILED;
    }
  } else {
    // inference phase
    if (ComputeOpsAndEdgesOutputCritical() == SUCCESS) {
      // Calculate operators' memory usage
      if (CalculateOpsMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' memory cost for inference failed.";
        return FAILED;
      }
      // Calculate edges's memory usage
      if (CalculateEdgesMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' memory cost for inference failed.";
        return FAILED;
      }
    } else {
      MS_LOG(ERROR) << "Computing operators' critical flag failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

void CostGraph::CheckApproximateCostGraphEdges() {
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (!approximation) {
    return;
  }
  for (const auto &s_edge : edges_) {
    auto &edges_vector = s_edge.second;
    for (auto &edge_ptr : edges_vector) {
      MS_EXCEPTION_IF_NULL(edge_ptr);
      if (edge_ptr->CheckStrategyCostPossibility()) {
        continue;
      }
      MS_LOG(INFO) << "Checking StrategyCost for edge: " << edge_ptr->edge_name()
                   << " impossible, re-initing the operators and edges";
      auto prev_op = edge_ptr->prev_operator();
      MS_EXCEPTION_IF_NULL(prev_op);
      auto next_op = edge_ptr->next_operator();
      MS_EXCEPTION_IF_NULL(next_op);
      // Check the 'prev_op'
      prev_op->ExactStrategiesAndRelatedEdges();
      // Check the 'next_op'
      next_op->ExactStrategiesAndRelatedEdges();
    }
  }
}

bool CostGraph::FindReshapeCostInCache(const TensorLayout &from, const TensorLayout &to, CostPtr *result) {
  auto key_from = std::make_pair(from.device_arrangement(), from.tensor_map());
  auto key_to = std::make_pair(to.device_arrangement(), to.tensor_map());
  auto key = std::make_pair(key_from, key_to);
  auto iter = reshape_cost_cache_.find(key);
  if (iter != reshape_cost_cache_.end()) {
    *result = iter->second;
    return true;
  }
  return false;
}

void CostGraph::SaveReshapeCostToCache(const TensorLayout &from, const TensorLayout &to, const CostPtr &result) {
  auto key_from = std::make_pair(from.device_arrangement(), from.tensor_map());
  auto key_to = std::make_pair(to.device_arrangement(), to.tensor_map());
  auto key = std::make_pair(key_from, key_to);
  reshape_cost_cache_.emplace(key, result);
}

bool CostGraph::FindEdgeCostPtrInCache(const TensorLayout &prev_layout, const TensorLayout &next_layout,
                                       CostPtrList *cost) {
  auto key_prev = std::make_pair(prev_layout.device_arrangement(), prev_layout.tensor_map());
  auto key_next = std::make_pair(next_layout.device_arrangement(), next_layout.tensor_map());
  auto key = std::make_pair(key_prev, key_next);
  auto iter = cost_map_cache_.find(key);
  if (iter != cost_map_cache_.end()) {
    *cost = iter->second;
    return true;
  }
  return false;
}

void CostGraph::SaveEdgeCostPtrToCache(const TensorLayout &prev_layout, const TensorLayout &next_layout,
                                       const CostPtrList &cost) {
  auto key_prev = std::make_pair(prev_layout.device_arrangement(), prev_layout.tensor_map());
  auto key_next = std::make_pair(next_layout.device_arrangement(), next_layout.tensor_map());
  auto key = std::make_pair(key_prev, key_next);
  cost_map_cache_.emplace(key, cost);
}
}  // namespace parallel
}  // namespace mindspore
