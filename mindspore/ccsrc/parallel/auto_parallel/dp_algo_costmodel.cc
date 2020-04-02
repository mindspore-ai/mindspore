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

#include "parallel/auto_parallel/dp_algo_costmodel.h"

#include <memory>
#include <utility>
#include <vector>

namespace mindspore {
namespace parallel {
Status GetStrategy(const CostGraphPtr& graph) {
  MS_LOG(INFO) << "Searching strategies begins.";
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<EliminationPtr> eliminations;
  bool flag = true;

  // Phase 1: Shrink the CostGraph using 6 operations, and record them in the order.
  // Note: the checking and applying of the 6 operations MUST in current order.
  while (flag) {
    flag = false;
    auto node = graph->CheckOpElimination();
    if (node != nullptr) {
      // Applying the Operator Elimination
      flag = true;
      auto l_edge = node->GetAlivePrevEdges()[0];
      auto r_edge = node->GetAliveSuccEdges()[0];
      auto n_edge = graph->EliminationOp(node);
      auto elimi = std::make_shared<OpElimination>(n_edge, l_edge, node, r_edge);
      eliminations.emplace_back(std::move(elimi));
    }
    auto edges = graph->CheckEdgeElimination();
    if ((!flag) && (!edges.empty())) {
      // Applying the Edge Elimination
      flag = true;
      auto n_edge = graph->EliminationEdges(edges);
      auto elimi = std::make_shared<EdgeElimination>(n_edge, edges);
      eliminations.emplace_back(std::move(elimi));
    }
    auto merge_node = graph->CheckMergeElimination();
    if ((!flag) && (merge_node != nullptr)) {
      // Applying the Merge Elimination
      flag = true;
      auto succ_edge = merge_node->GetAliveSuccEdges()[0];
      auto target_node = graph->EliminationMerge(merge_node);
      auto elimi = std::make_shared<MergeElimination>(merge_node, succ_edge, target_node);
      eliminations.emplace_back(std::move(elimi));
    }
    auto contracted_node = graph->CheckContractElimination();
    if ((!flag) && (contracted_node != nullptr)) {
      // Applying the Contract Elimination
      flag = true;
      auto prev_edge = contracted_node->GetAlivePrevEdges()[0];
      auto target_node = graph->EliminationContract(contracted_node);
      auto elimi = std::make_shared<ContractElimination>(target_node, prev_edge, contracted_node);
      eliminations.emplace_back(std::move(elimi));
    }
    auto triangle_pair = graph->CheckTriangleElimination();
    if ((!flag) && (triangle_pair.first != nullptr)) {
      // Applying the Triangle Elimination
      flag = true;
      auto eliminated_node = triangle_pair.first;
      auto l_r_edge = triangle_pair.second;

      auto left_node = l_r_edge->prev_operator();
      auto right_node = l_r_edge->next_operator();
      auto left_edge = eliminated_node->GetAliveSuccEdges()[0];
      auto right_edge = eliminated_node->GetAliveSuccEdges()[1];
      MS_EXCEPTION_IF_NULL(left_edge);
      if (left_edge->next_operator() != left_node) {
        auto tmp = left_edge;
        left_edge = right_edge;
        right_edge = tmp;
      }
      auto left_node_cpy = graph->EliminationTriangle(eliminated_node, l_r_edge);
      auto elimi =
        std::make_shared<TriangleElimination>(eliminated_node, left_edge, left_node_cpy, right_edge, right_node);
      eliminations.emplace_back(std::move(elimi));
    }
    auto star_center = graph->CheckStarElimination();
    if ((!flag) && (star_center != nullptr)) {
      // Applying the Star Elimination
      flag = true;
      auto succ_edges = graph->EliminationStar(star_center);
      std::vector<OperatorInfoPtr> succ_nodes;
      for (size_t i = 0; i < succ_edges.size(); ++i) {
        MS_EXCEPTION_IF_NULL(succ_edges[i]);
        succ_nodes.push_back(succ_edges[i]->next_operator());
      }
      auto elimi = std::make_shared<StarElimination>(star_center, succ_edges, succ_nodes);
      eliminations.emplace_back(std::move(elimi));
    }
  }

  // Phase 2: Search the cost_list in the final graph, and determine the optimal one
  if (graph->SearchStrategy() != SUCCESS) {
    MS_LOG(ERROR) << "Searching strategy for the final failed.";
    return FAILED;
  }

  // Phase 3: Recover the original CostGraph, the determine strategy for each operator
  if (RecoverStrategy(eliminations) == SUCCESS) {
    MS_LOG(INFO) << "Searching strategies ends.";
    return SUCCESS;
  } else {
    MS_LOG(EXCEPTION) << "Searching strategies failed.";
  }
}

Status RecoverStrategy(std::vector<EliminationPtr> eliminations) {
  std::vector<EliminationPtr>::reverse_iterator rit;

  for (rit = eliminations.rbegin(); rit != eliminations.rend(); ++rit) {
    if ((*rit)->isa<OpElimination>()) {
      auto elimination = (*rit)->cast<OpEliminationPtr>();
      auto e = elimination->new_edge_;
      auto w = elimination->op_;
      MS_EXCEPTION_IF_NULL(e);
      MS_EXCEPTION_IF_NULL(w);
      auto left_edge = elimination->left_edge_;
      auto right_edge = elimination->right_edge_;
      MS_EXCEPTION_IF_NULL(left_edge);
      MS_EXCEPTION_IF_NULL(right_edge);
      auto decision = e->selected_cost()->decision_ptr_->cast<OpEliminationDecisionPtr>();
      w->SetSelectedStrategyAndCost(decision->op_strategy_, decision->middle_cost_);
      left_edge->set_selected_cost(decision->left_cost_);
      right_edge->set_selected_cost(decision->right_cost_);
      MS_LOG(INFO) << "Recover opElimination succeeded.";
    } else if ((*rit)->isa<EdgeElimination>()) {
      auto elimination = (*rit)->cast<EdgeEliminationPtr>();
      auto new_edge = elimination->new_edge_;
      MS_EXCEPTION_IF_NULL(new_edge);
      auto& edges = elimination->edges_;
      auto decision = new_edge->selected_cost()->decision_ptr_->cast<EdgeEliminationDecisionPtr>();
      for (size_t j = 0; j < edges.size(); ++j) {
        MS_EXCEPTION_IF_NULL(edges[j]);
        edges[j]->set_selected_cost(decision->edges_cost_list_[j]);
      }
      MS_LOG(INFO) << "Recover edgeElimination succeeded.";
    } else if ((*rit)->isa<MergeElimination>()) {
      auto elimination = (*rit)->cast<MergeEliminationPtr>();
      auto target_node = elimination->target_node_;
      MS_EXCEPTION_IF_NULL(target_node);
      auto merged_node = elimination->merged_node_;
      MS_EXCEPTION_IF_NULL(merged_node);
      auto merged_edge = elimination->dir_edge_;
      MS_EXCEPTION_IF_NULL(merged_edge);
      MS_EXCEPTION_IF_NULL(target_node->selected_cost());
      MS_EXCEPTION_IF_NULL(target_node->selected_cost()->decision_ptr_);
      auto decision = target_node->selected_cost()->decision_ptr_->cast<MergeEliminationDecisionPtr>();
      merged_node->SetSelectedStrategyAndCost(decision->merged_op_strategy_, decision->merged_op_cost_);
      merged_edge->set_selected_cost(decision->edge_cost_);
      target_node->SetSelectedStrategyAndCost(decision->target_op_strategy_, decision->target_op_cost_);

      MS_LOG(INFO) << "Recover mergeElimination succeeded.";
    } else if ((*rit)->isa<ContractElimination>()) {
      auto elimination = (*rit)->cast<ContractEliminationPtr>();
      auto target_node = elimination->target_node_;
      auto contracted_node = elimination->contracted_node_;
      auto contracted_edge = elimination->dir_edge_;
      auto decision = target_node->selected_cost()->decision_ptr_->cast<ContractEliminationDecisionPtr>();

      contracted_node->SetSelectedStrategyAndCost(decision->contracted_op_strategy_, decision->contracted_op_cost_);
      contracted_edge->set_selected_cost(decision->edge_cost_);
      target_node->SetSelectedStrategyAndCost(decision->target_op_strategy_, decision->target_cost_);
      MS_LOG(INFO) << "Recover contractElimination succeeded.";
    } else if ((*rit)->isa<TriangleElimination>()) {
      auto elimination = (*rit)->cast<TriangleEliminationPtr>();
      auto left_node = elimination->left_node_;
      auto left_edge = elimination->left_edge_;
      auto eliminated_node = elimination->eliminated_node_;
      auto right_edge = elimination->right_edge_;
      auto right_node = elimination->right_node_;
      auto decision = left_node->selected_cost()->decision_ptr_->cast<TriangleEliminationDecisionPtr>();

      eliminated_node->SetSelectedStrategyAndCost(decision->eliminated_op_strategy_, decision->eliminated_op_cost_);
      left_edge->set_selected_cost(decision->left_edge_cost_);
      right_edge->set_selected_cost(decision->right_edge_cost_);
      left_node->SetSelectedStrategyAndCost(decision->left_node_strategy_, decision->left_node_cost_);
      right_node->SetSelectedStrategyAndCost(decision->right_node_strategy_, decision->right_node_cost_);
      MS_LOG(INFO) << "Recover triangleElimination succeeded.";
    } else if ((*rit)->isa<StarElimination>()) {
      auto elimination = (*rit)->cast<StarEliminationPtr>();
      auto merged_node = elimination->eliminated_node_;
      auto succ_edges = elimination->succ_edges_;
      auto succ_nodes = elimination->succ_ops_;
      // decision is hided in succ_nodes[0]
      auto decision = succ_nodes[0]->selected_cost()->decision_ptr_->cast<StarEliminationDecisionPtr>();

      merged_node->SetSelectedStrategyAndCost(decision->eliminated_op_strategy_, decision->eliminated_op_cost_);
      for (size_t i = 0; i < succ_edges.size(); ++i) {
        succ_edges[i]->set_selected_cost(decision->succ_edges_cost_list_[i]);
      }
      for (size_t j = 0; j < succ_nodes.size(); ++j) {
        succ_nodes[j]->SetSelectedStrategyAndCost(decision->succ_ops_stra_list_[j], decision->succ_ops_cost_list_[j]);
      }
      MS_LOG(INFO) << "Recover starElimination succeeded.";
    } else {
      MS_LOG(ERROR) << "Unknown Elimination type.";
      return FAILED;
    }
  }

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
