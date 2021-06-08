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

#include "frontend/parallel/auto_parallel/dp_algo_costmodel.h"

#include <memory>
#include <utility>
#include <vector>

namespace mindspore {
namespace parallel {
Status GetStrategy(const CostGraphPtr &graph) {
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
      auto elimi_op = std::make_shared<OpElimination>(n_edge, l_edge, node, r_edge);
      (void)eliminations.emplace_back(std::move(elimi_op));
    }
    if (!flag) {
      auto edges = graph->CheckEdgeElimination();
      if (!edges.empty()) {
        // Applying the Edge Elimination
        flag = true;
        auto new_edge = graph->EliminationEdges(edges);
        auto elimi_edge = std::make_shared<EdgeElimination>(new_edge, edges);
        (void)eliminations.emplace_back(std::move(elimi_edge));
      }
    }
    if (!flag) {
      auto merge_node = graph->CheckMergeElimination();
      if (merge_node != nullptr) {
        // Applying the Merge Elimination
        flag = true;
        auto succ_edge = merge_node->GetAliveSuccEdges()[0];
        auto target_node = graph->EliminationMerge(merge_node);
        auto elimi_merge = std::make_shared<MergeElimination>(merge_node, succ_edge, target_node);
        (void)eliminations.emplace_back(std::move(elimi_merge));
      }
    }
    if (!flag) {
      auto contracted_node = graph->CheckContractElimination();
      if ((contracted_node != nullptr)) {
        // Applying the Contract Elimination
        flag = true;
        auto prev_edge = contracted_node->GetAlivePrevEdges()[0];
        auto target_node = graph->EliminationContract(contracted_node);
        auto elimi_contract = std::make_shared<ContractElimination>(target_node, prev_edge, contracted_node);
        (void)eliminations.emplace_back(std::move(elimi_contract));
      }
    }
    if (!flag) {
      auto triangle_pair = graph->CheckTriangleElimination();
      if (triangle_pair.first != nullptr) {
        // Applying the Triangle Elimination
        flag = true;
        auto eliminated_node = triangle_pair.first;
        auto l_r_edge = triangle_pair.second;

        auto left_node = l_r_edge->prev_operator();
        auto left_edge = eliminated_node->GetAliveSuccEdges()[0];
        auto right_edge = eliminated_node->GetAliveSuccEdges()[1];
        MS_EXCEPTION_IF_NULL(left_edge);
        if (left_edge->next_operator() != left_node) {
          auto tmp = left_edge;
          left_edge = right_edge;
          right_edge = tmp;
        }
        auto left_node_cpy = graph->EliminationTriangle(eliminated_node, l_r_edge);
        auto right_node = l_r_edge->next_operator();
        auto elimi_tri =
          std::make_shared<TriangleElimination>(eliminated_node, left_edge, left_node_cpy, right_edge, right_node);
        (void)eliminations.emplace_back(std::move(elimi_tri));
      }
    }
    if (!flag) {
      auto star_center = graph->CheckStarElimination();
      if (star_center != nullptr) {
        // Applying the Star Elimination
        flag = true;
        auto succ_edges = graph->EliminationStar(star_center);
        std::vector<OperatorInfoPtr> succ_nodes;
        for (size_t i = 0; i < succ_edges.size(); ++i) {
          MS_EXCEPTION_IF_NULL(succ_edges[i]);
          succ_nodes.push_back(succ_edges[i]->next_operator());
        }
        auto elimi_star = std::make_shared<StarElimination>(star_center, succ_edges, succ_nodes);
        (void)eliminations.emplace_back(std::move(elimi_star));
      }
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
  const auto triangle_star_overwrite = CostModelContext::GetInstance()->triangle_star_strategy_overwrite();
  for (rit = eliminations.rbegin(); rit != eliminations.rend(); ++rit) {
    if ((*rit)->isa<OpElimination>()) {
      auto elimination_op = (*rit)->cast<OpEliminationPtr>();
      auto e = elimination_op->new_edge_;
      auto w = elimination_op->op_;
      auto left_edge_op = elimination_op->left_edge_;
      auto right_edge_op = elimination_op->right_edge_;
      auto decision_op = e->selected_cost()->decision_ptr_->cast<OpEliminationDecisionPtr>();
      w->SetSelectedStrategyAndCost(decision_op->op_strategy_, decision_op->middle_cost_);
      left_edge_op->set_selected_cost(decision_op->left_cost_);
      right_edge_op->set_selected_cost(decision_op->right_cost_);
      MS_LOG(INFO) << "Recover opElimination succeeded.";
    } else if ((*rit)->isa<EdgeElimination>()) {
      auto elimination_edge = (*rit)->cast<EdgeEliminationPtr>();
      auto new_edge = elimination_edge->new_edge_;
      auto &edges = elimination_edge->edges_;
      auto decision_edge = new_edge->selected_cost()->decision_ptr_->cast<EdgeEliminationDecisionPtr>();
      for (size_t j = 0; j < edges.size(); ++j) {
        MS_EXCEPTION_IF_NULL(edges[j]);
        edges[j]->set_selected_cost(decision_edge->edges_cost_list_[j]);
      }
      MS_LOG(INFO) << "Recover edgeElimination succeeded.";
    } else if ((*rit)->isa<MergeElimination>()) {
      auto elimination_merge = (*rit)->cast<MergeEliminationPtr>();
      auto target_node_m = elimination_merge->target_node_;
      auto merged_node = elimination_merge->merged_node_;
      auto merged_edge = elimination_merge->dir_edge_;
      MS_EXCEPTION_IF_NULL(target_node_m->selected_cost());
      MS_EXCEPTION_IF_NULL(target_node_m->selected_cost()->decision_ptr_);
      auto decision = target_node_m->selected_cost()->decision_ptr_->cast<MergeEliminationDecisionPtr>();
      merged_node->SetSelectedStrategyAndCost(decision->merged_op_strategy_, decision->merged_op_cost_);
      merged_edge->set_selected_cost(decision->edge_cost_);
      target_node_m->SetSelectedStrategyAndCost(decision->target_op_strategy_, decision->target_op_cost_);
      MS_LOG(INFO) << "Recover mergeElimination succeeded.";
    } else if ((*rit)->isa<ContractElimination>()) {
      auto elimination_cont = (*rit)->cast<ContractEliminationPtr>();
      auto target_node = elimination_cont->target_node_;
      auto contracted_node = elimination_cont->contracted_node_;
      auto contracted_edge = elimination_cont->dir_edge_;
      auto decision_cont = target_node->selected_cost()->decision_ptr_->cast<ContractEliminationDecisionPtr>();
      contracted_node->SetSelectedStrategyAndCost(decision_cont->contracted_op_strategy_,
                                                  decision_cont->contracted_op_cost_);
      contracted_edge->set_selected_cost(decision_cont->edge_cost_);
      target_node->SetSelectedStrategyAndCost(decision_cont->target_op_strategy_, decision_cont->target_cost_);
      MS_LOG(INFO) << "Recover contractElimination succeeded.";
    } else if ((*rit)->isa<TriangleElimination>()) {
      auto elimination_tri = (*rit)->cast<TriangleEliminationPtr>();
      auto left_node = elimination_tri->left_node_;
      auto left_edge = elimination_tri->left_edge_;
      auto eliminated_node = elimination_tri->eliminated_node_;
      auto right_edge_tri = elimination_tri->right_edge_;
      auto right_node = elimination_tri->right_node_;
      auto decision_tri = left_node->selected_cost()->decision_ptr_->cast<TriangleEliminationDecisionPtr>();

      eliminated_node->SetSelectedStrategyAndCost(decision_tri->eliminated_op_strategy_,
                                                  decision_tri->eliminated_op_cost_);
      left_edge->set_selected_cost(decision_tri->left_edge_cost_);
      right_edge_tri->set_selected_cost(decision_tri->right_edge_cost_);
      // 'left_node' recovers the strategy.
      left_node->SetSelectedStrategyAndCost(decision_tri->left_node_strategy_, decision_tri->left_node_cost_);
      if (triangle_star_overwrite) {
        // 'right_node' recovers the strategy.
        MS_LOG(INFO) << "Overwrite the right-node: " << right_node->name() << " in recovering triangle elimination.";
        right_node->SetSelectedStrategyAndCost(decision_tri->right_node_strategy_, decision_tri->right_node_cost_);
      } else {
        // In this case, 'right_node' is not overwritten strategy, and it checks strategy consistency.
        right_node->CheckSelectedStrategy(decision_tri->right_node_strategy_);
      }
      MS_LOG(INFO) << "Recover triangleElimination succeeded.";
    } else if ((*rit)->isa<StarElimination>()) {
      auto elimination_star = (*rit)->cast<StarEliminationPtr>();
      auto merged_node_star = elimination_star->eliminated_node_;
      auto succ_edges = elimination_star->succ_edges_;
      auto succ_nodes = elimination_star->succ_ops_;
      // decision is hidden in succ_nodes[0]
      auto decision_star = succ_nodes[0]->selected_cost()->decision_ptr_->cast<StarEliminationDecisionPtr>();
      merged_node_star->SetSelectedStrategyAndCost(decision_star->eliminated_op_strategy_,
                                                   decision_star->eliminated_op_cost_);
      for (size_t i = 0; i < succ_edges.size(); ++i) {
        succ_edges[i]->set_selected_cost(decision_star->succ_edges_cost_list_[i]);
      }
      MS_EXCEPTION_IF_NULL(succ_nodes[0]);
      MS_EXCEPTION_IF_NULL(decision_star->succ_ops_stra_list_[0]);
      MS_EXCEPTION_IF_NULL(decision_star->succ_ops_cost_list_[0]);
      // Star is eliminated into 'succ_nodes[0]'
      succ_nodes[0]->SetSelectedStrategyAndCost(decision_star->succ_ops_stra_list_[0],
                                                decision_star->succ_ops_cost_list_[0]);
      for (size_t k = 1; k < succ_nodes.size(); ++k) {
        if (triangle_star_overwrite) {
          // 'succ_nodes[k]' is overwritten strategy and cost.
          succ_nodes[k]->SetSelectedStrategyAndCost(decision_star->succ_ops_stra_list_[k],
                                                    decision_star->succ_ops_cost_list_[k]);
        } else {
          // In this case, 'succ_nodes[k]' is NOT overwritten strategy and cost, however, it checks the strategy.
          succ_nodes[k]->CheckSelectedStrategy(decision_star->succ_ops_stra_list_[k]);
        }
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
