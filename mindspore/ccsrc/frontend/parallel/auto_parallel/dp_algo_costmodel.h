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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_DP_ALGO_COSTMODEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_DP_ALGO_COSTMODEL_H_

#include <memory>
#include <utility>
#include <vector>
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "ir/value.h"

namespace mindspore {
namespace parallel {
// There are 3 meta phases of the Dynamic Programming (DP) algorithm. The input is a CostGraph, and the goal
// is to compute the strategy for each operator in the CostGraph.
//
// Phase 1: Shrink the CostGraph using 6 operations, and record them in the order
//       Using for operations: Operator Elimination, Edge Elimination, Merge Elimination, and Contract Elimination,
//       each connected component in the CostGraph can be shrunk in to the final graph: u --> v. See the
//       interpretation of 6 operations in costmodel.h.
// Phase 2: Search the cost_list in the final graph, and determine the optimal one
//       Create the cost_list for the final graph, and choose the optimal one: one the minimum quantity
//       COST_MODEL_ALPHA * computation_cost + COST_MODEL_BETA * communication_cost
// Phase 3: Recover the original CostGraph, the determine strategy for each operator
//       After determining the optimal cost for the final graph, the algorithm recovers the original graph by applying
//       the 4 operations in the reverse order in the Phase 1. Because each operation decision contains the strategy,
//       the operators' strategies can be all determined.

struct Elimination : public Base {
  enum EliminationType { OPERA, EDGE, MERGE, CONTRACT, SOURCE, TRIANGLE, STAR };
  Elimination(EdgePtr n_edge, EliminationType ty) : new_edge_(std::move(n_edge)), type_(ty) {}

  EdgePtr new_edge_;
  EliminationType type_;
};

// Operator Elimination
struct OpElimination : public Elimination {
  OpElimination(EdgePtr n_edge, EdgePtr l_edge, OperatorInfoPtr op_info, EdgePtr r_edge)
      : Elimination(std::move(n_edge), Elimination::EliminationType::OPERA),
        left_edge_(std::move(l_edge)),
        op_(std::move(op_info)),
        right_edge_(std::move(r_edge)) {}

  EdgePtr left_edge_;
  OperatorInfoPtr op_;
  EdgePtr right_edge_;
  MS_DECLARE_PARENT(OpElimination, Elimination);
};

// Edge Elimination
struct EdgeElimination : public Elimination {
  EdgeElimination(const EdgePtr &n_edge, std::vector<EdgePtr> eds)
      : Elimination(n_edge, Elimination::EliminationType::EDGE), edges_(std::move(eds)) {}

  std::vector<EdgePtr> edges_;
  MS_DECLARE_PARENT(EdgeElimination, Elimination);
};

// Merge Elimination
struct MergeElimination : public Elimination {
  MergeElimination(OperatorInfoPtr u_info, EdgePtr merged_target_edge, OperatorInfoPtr v_info)
      : Elimination(nullptr, Elimination::EliminationType::MERGE),
        merged_node_(std::move(u_info)),
        dir_edge_(std::move(merged_target_edge)),
        target_node_(std::move(v_info)) {}

  OperatorInfoPtr merged_node_;
  EdgePtr dir_edge_;
  OperatorInfoPtr target_node_;
  MS_DECLARE_PARENT(MergeElimination, Elimination);
};

// Contract Elimination
struct ContractElimination : public Elimination {
  ContractElimination(OperatorInfoPtr tar_info, EdgePtr tar_con_edge, OperatorInfoPtr con_info)
      : Elimination(nullptr, Elimination::EliminationType::CONTRACT),
        contracted_node_(std::move(con_info)),
        dir_edge_(std::move(tar_con_edge)),
        target_node_(std::move(tar_info)) {}

  OperatorInfoPtr contracted_node_;
  EdgePtr dir_edge_;
  OperatorInfoPtr target_node_;
  MS_DECLARE_PARENT(ContractElimination, Elimination);
};

// Source Elimination
struct SourceElimination : public Elimination {
  SourceElimination(OperatorInfoPtr p_source, std::vector<EdgePtr> p_succ_edges, std::vector<EdgePtr> p_new_succ_edges,
                    OperatorInfoPtr s_source, std::vector<EdgePtr> s_succ_edges, std::vector<EdgePtr> s_new_succ_edges)
      : Elimination(nullptr, Elimination::EliminationType::SOURCE),
        primary_source_(std::move(p_source)),
        primary_succ_edges_(std::move(p_succ_edges)),
        primary_new_succ_edges_(std::move(p_new_succ_edges)),
        secondary_source_(std::move(s_source)),
        secondary_succ_edges_(std::move(s_succ_edges)),
        secondary_new_succ_edges_(std::move(s_new_succ_edges)) {}
  OperatorInfoPtr primary_source_;
  std::vector<EdgePtr> primary_succ_edges_;
  std::vector<EdgePtr> primary_new_succ_edges_;
  OperatorInfoPtr secondary_source_;
  std::vector<EdgePtr> secondary_succ_edges_;
  std::vector<EdgePtr> secondary_new_succ_edges_;
  MS_DECLARE_PARENT(SourceElimination, Elimination);
};

// Triangle Elimination
struct TriangleElimination : public Elimination {
  TriangleElimination(OperatorInfoPtr elim_node, EdgePtr l_edge, OperatorInfoPtr l_node, EdgePtr r_edge,
                      OperatorInfoPtr r_node)
      : Elimination(nullptr, Elimination::EliminationType::TRIANGLE),
        eliminated_node_(std::move(elim_node)),
        left_edge_(std::move(l_edge)),
        left_node_(std::move(l_node)),
        right_edge_(std::move(r_edge)),
        right_node_(std::move(r_node)) {}

  OperatorInfoPtr eliminated_node_;
  EdgePtr left_edge_;
  OperatorInfoPtr left_node_;
  EdgePtr right_edge_;
  OperatorInfoPtr right_node_;
  MS_DECLARE_PARENT(TriangleElimination, Elimination);
};

// Star Elimination
struct StarElimination : public Elimination {
  StarElimination(OperatorInfoPtr elimi_node, std::vector<EdgePtr> s_edges, std::vector<OperatorInfoPtr> s_ops)
      : Elimination(nullptr, Elimination::EliminationType::STAR),
        eliminated_node_(std::move(elimi_node)),
        succ_edges_(std::move(s_edges)),
        succ_ops_(std::move(s_ops)) {}

  OperatorInfoPtr eliminated_node_;
  std::vector<EdgePtr> succ_edges_;
  std::vector<OperatorInfoPtr> succ_ops_;
  MS_DECLARE_PARENT(StarElimination, Elimination);
};

using EliminationPtr = std::shared_ptr<Elimination>;
using OpEliminationPtr = std::shared_ptr<OpElimination>;
using EdgeEliminationPtr = std::shared_ptr<EdgeElimination>;
using MergeEliminationPtr = std::shared_ptr<MergeElimination>;
using ContractEliminationPtr = std::shared_ptr<ContractElimination>;
using SourceEliminationPtr = std::shared_ptr<SourceElimination>;
using TriangleEliminationPtr = std::shared_ptr<TriangleElimination>;
using StarEliminationPtr = std::shared_ptr<StarElimination>;

// Phase 1 and Phase 2
Status GetStrategy(const CostGraphPtr &graph);

// Phase 3
Status RecoverStrategy(std::vector<EliminationPtr> eliminations);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_DP_ALGO_COSTMODEL_H_
