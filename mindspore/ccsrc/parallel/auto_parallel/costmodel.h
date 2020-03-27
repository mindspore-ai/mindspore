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

#ifndef MINDSPORE_CCSRC_PARALLEL_AUTO_PARALLEL_COSTMODEL_H_
#define MINDSPORE_CCSRC_PARALLEL_AUTO_PARALLEL_COSTMODEL_H_

#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_info.h"

namespace mindspore {
namespace parallel {
struct Decision;
using OperatorName = std::string;
using Attr = std::pair<std::string, ValuePtr>;
using Param = std::pair<std::pair<std::string, ValuePtr>, int32_t>;
using OperatorParams = std::vector<Param>;
using OperatorAttrs = std::vector<Attr>;
// OutPutInfo.fist: true if the operator's output is a tuple
// OutPutInfo.second: elements number of the tuple output. Only meaningful if OutPutInfo.fist is true.
using OutPutInfo = std::pair<bool, uint32_t>;
using OutPutInfoVector = std::vector<OutPutInfo>;
using OperatorArgs = std::pair<OperatorAttrs, OperatorParams>;
using Operator = std::pair<OperatorName, OperatorArgs>;
using OperatorVector = std::vector<Operator>;
using RedistributionOpListPtr = std::shared_ptr<std::pair<OperatorVector, OutPutInfoVector>>;

struct Cost {
  Cost();
  Cost(double memory, double commuication, const std::shared_ptr<Decision>& decision_ = nullptr)
      : memory_cost_(memory), communication_cost_(commuication), decision_ptr_(std::move(decision_)) {
    communication_without_parameter_ = 0.0;
    communication_with_partial_para_ = 0.0;
    communication_redis_forward_ = 0.0;
    communication_redis_backward_ = 0.0;
  }
  double memory_cost_;
  // 'communication_cost_' includes communications from operators (forward and backward) and edges
  double communication_cost_;
  // communication_without_parameter_ = communication_cost_ - (backward communication from operators)
  double communication_without_parameter_;
  // communication_with_partial_para_ =
  // communication_without_parameter_ + COST_MODEL_GAMMA * (communication_cost_ - communication_without_parameter_ )
  double communication_with_partial_para_;
  double communication_redis_forward_;
  double communication_redis_backward_;
  std::shared_ptr<Decision> decision_ptr_;
};

using CostPtr = std::shared_ptr<Cost>;
using CostPtrList = std::vector<std::shared_ptr<Cost>>;

class StrategyWithCost {
 public:
  StrategyWithCost(StrategyPtr strategy, std::vector<TensorInfo> inputs_, std::vector<TensorInfo> outputs_)
      : strategy_ptr(std::move(strategy)), inputs_ptr(std::move(inputs_)), outputs_ptr(std::move(outputs_)) {}

  StrategyWithCost(const StrategyWithCost& swc) = delete;
  StrategyWithCost(StrategyWithCost&& swc)
      : strategy_ptr(swc.strategy_ptr),
        inputs_ptr(swc.inputs_ptr),
        outputs_ptr(swc.outputs_ptr),
        cost_list(swc.cost_list) {}
  ~StrategyWithCost() = default;

  StrategyPtr strategy_ptr;
  std::vector<TensorInfo> inputs_ptr;
  std::vector<TensorInfo> outputs_ptr;
  CostPtrList cost_list;
};

enum DecisionType {
  OP_ELIMINATION,
  EDGE_ELIMINATION,
  MERGE_ELIMINATION,
  CONTRACT_ELIMINATION,
  TRIANGLE_ELIMINATION,
  STAR_ELIMINATION,
  FINAL_TYPE,
  FINAL_SINGLE
};

struct Decision : public Base {
  ~Decision() override = default;
  DecisionType type_;
};

// 'OpEliminationDecision' is for the Operator Elimination in DP algorithm: u --> v --> w ==> u --> w.
// This data structure records the strategy 'op_strategy_' for v, the edge cost 'left_cost_' for 'u --> v', the
// operator cost 'middle_cost_' for v, and the edge cost 'right_cost_' for 'v --> w'
struct OpEliminationDecision : public Decision {
  OpEliminationDecision(StrategyPtr op_stra, CostPtr l_cost, CostPtr m_cost, CostPtr r_cost)
      : op_strategy_(std::move(op_stra)),
        left_cost_(std::move(l_cost)),
        middle_cost_(std::move(m_cost)),
        right_cost_(std::move(r_cost)) {
    type_ = DecisionType::OP_ELIMINATION;
  }

  StrategyPtr op_strategy_;
  CostPtr left_cost_;
  CostPtr middle_cost_;
  CostPtr right_cost_;
  MS_DECLARE_PARENT(OpEliminationDecision, Decision);
};

/* 'EdgeEliminationDecision' is for the Edge Elimination in DP algorithm:
   ____
  /    \
 u      v   ==>  u --> v, which replace the multi-edges by a single edge.
  \____/
 This data structure records the cost list for all edges 'edges_cost_list_'
 */
struct EdgeEliminationDecision : public Decision {
  explicit EdgeEliminationDecision(CostPtrList cost_list) : edges_cost_list_(std::move(cost_list)) {
    type_ = DecisionType::EDGE_ELIMINATION;
  }

  CostPtrList edges_cost_list_;
  MS_DECLARE_PARENT(EdgeEliminationDecision, Decision);
};

// 'MergeEliminationDecision' is for the Merge Elimination in DP algorithm:
//       w
//       |
//       |   ==>  u --> v
// u --> v                    In the original graph, v has two alive incoming edges, w has one alive outgoing edge,
// and w has zero alive incoming edges. After the Merge Elimination, the result graph contains only 'u -- >v'.
// This data structure records the strategy 'merged_op_strategy_' for operator 'w',
// the cost 'merged_op_cost_' for operator 'w', and the edge cost 'edge_cost_' for 'w --> v'.
struct MergeEliminationDecision : public Decision {
  MergeEliminationDecision(StrategyPtr op_stra, CostPtr op_cost, CostPtr edge_c, StrategyPtr tar_op_stra,
                           CostPtr target_op_c)
      : merged_op_strategy_(std::move(op_stra)),
        merged_op_cost_(std::move(op_cost)),
        edge_cost_(std::move(edge_c)),
        target_op_strategy_(std::move(tar_op_stra)),
        target_op_cost_(std::move(target_op_c)) {
    type_ = DecisionType::MERGE_ELIMINATION;
  }

  StrategyPtr merged_op_strategy_;
  CostPtr merged_op_cost_;
  CostPtr edge_cost_;
  StrategyPtr target_op_strategy_;
  CostPtr target_op_cost_;
  MS_DECLARE_PARENT(MergeEliminationDecision, Decision);
};

// 'ContractEliminationDecision' is for the Contract Elimination in DP algorithm:
//  u --> v
//  |
//  |      ==>   u --> w
//  w                         In the original graph, u has two alive outgoing edges, v has one alive incoming edge,
// and v has zero outgoing edge. After the Contract Elimination, the result graph contains only 'u --> w'.
// This data structure records the strategy 'contracted_op_strategy_' for operator 'v', the cost for
// operator 'contracted_op_cost_', and the edge cost for 'edge_cost_'.
struct ContractEliminationDecision : public Decision {
  ContractEliminationDecision(StrategyPtr contra_stra, CostPtr contra_op_cost, CostPtr edge_cost,
                              StrategyPtr target_stra, CostPtr tar_cost)
      : contracted_op_strategy_(std::move(contra_stra)),
        contracted_op_cost_(std::move(contra_op_cost)),
        edge_cost_(std::move(edge_cost)),
        target_op_strategy_(std::move(target_stra)),
        target_cost_(std::move(tar_cost)) {
    type_ = DecisionType::CONTRACT_ELIMINATION;
  }

  StrategyPtr contracted_op_strategy_;
  CostPtr contracted_op_cost_;
  CostPtr edge_cost_;
  StrategyPtr target_op_strategy_;
  CostPtr target_cost_;
  MS_DECLARE_PARENT(ContractEliminationDecision, Decision);
};

/* 'TriangleEliminationDecision' is for the Triangle Elimination in DP algorithm:
 *
 *       u
 *      / \
 *     /   \
 *    v --- w   ==> v --- w  In the original graph, u has 2 outgoing edges, v has 1 outgoing edge,
 * and w has 2 incoming edges, u can be eliminated into v.
 * 'eliminated_op_strategy_' is for u, 'eliminated_op_cost_' is for u, 'eliminated_left_edge_' is for edge u --> v,
 * 'eliminated_right_edge_' is for edge u --> w.
 */
struct TriangleEliminationDecision : public Decision {
  TriangleEliminationDecision(StrategyPtr elimi_stra, CostPtr elimi_op_cost, CostPtr l_edge_cost, CostPtr r_edge_cost,
                              StrategyPtr left_stra, CostPtr l_node_cost, StrategyPtr right_stra, CostPtr r_node_cost)
      : eliminated_op_strategy_(std::move(elimi_stra)),
        eliminated_op_cost_(std::move(elimi_op_cost)),
        left_edge_cost_(std::move(l_edge_cost)),
        right_edge_cost_(std::move(r_edge_cost)),
        left_node_strategy_(std::move(left_stra)),
        left_node_cost_(std::move(l_node_cost)),
        right_node_strategy_(std::move(right_stra)),
        right_node_cost_(std::move(r_node_cost)) {
    type_ = DecisionType::TRIANGLE_ELIMINATION;
  }

  StrategyPtr eliminated_op_strategy_;
  CostPtr eliminated_op_cost_;
  CostPtr left_edge_cost_;
  CostPtr right_edge_cost_;
  StrategyPtr left_node_strategy_;
  CostPtr left_node_cost_;
  StrategyPtr right_node_strategy_;
  CostPtr right_node_cost_;
  MS_DECLARE_PARENT(TriangleEliminationDecision, Decision);
};

/* 'StarEliminationDecision' is for the Star Elimination in DP algorithm:
 *
 *  v <--- u ---> w  ==> v    w  In the original graph, u has 0 incoming edges, and multiple outgoing edges.
 *  In addition, v and w have other complicated connections, resulting in v and w can not be performed other
 *  eliminations. After the StarElimination, u is merged into v, and the resulting graph is splitted into multiple
 *  connected components.
 *  NOTE: this elimination MUST be performed only when the above 5 operation cannot be applied.
 */
struct StarEliminationDecision : public Decision {
  StarEliminationDecision(StrategyPtr elimi_op_stra, CostPtr elimi_op_cost, CostPtrList succ_edges_clist,
                          std::vector<StrategyPtr> succ_ops_stra_list, CostPtrList succ_ops_clist)
      : eliminated_op_strategy_(std::move(elimi_op_stra)),
        eliminated_op_cost_(std::move(elimi_op_cost)),
        succ_edges_cost_list_(std::move(succ_edges_clist)),
        succ_ops_stra_list_(std::move(succ_ops_stra_list)),
        succ_ops_cost_list_(std::move(succ_ops_clist)) {
    type_ = DecisionType::STAR_ELIMINATION;
  }

  StrategyPtr eliminated_op_strategy_;
  CostPtr eliminated_op_cost_;
  CostPtrList succ_edges_cost_list_;
  std::vector<StrategyPtr> succ_ops_stra_list_;
  CostPtrList succ_ops_cost_list_;
  MS_DECLARE_PARENT(StarEliminationDecision, Decision);
};

// This data structure records the decision for the graph which contains two nodes: u --> v. This includes
// the strategy 'u_strategy_' for 'u', the strategy 'v_strategy_' for 'v', the cost 'left_cost_' for 'u'.
struct FinalDecision : public Decision {
  FinalDecision(StrategyPtr u_stra, StrategyPtr v_stra, CostPtr l_cost, CostPtr m_cost, CostPtr r_cost)
      : u_strategy_(std::move(u_stra)),
        v_strategy_(std::move(v_stra)),
        left_cost_(std::move(l_cost)),
        middle_cost_(std::move(m_cost)),
        right_cost_(std::move(r_cost)) {
    type_ = DecisionType::FINAL_TYPE;
  }

  StrategyPtr u_strategy_;
  StrategyPtr v_strategy_;
  CostPtr left_cost_;
  CostPtr middle_cost_;
  CostPtr right_cost_;
  MS_DECLARE_PARENT(FinalDecision, Decision);
};

// This data structure records the final decision for the graph containing a single node: u. This includes
// the strategy 'u_strategy_' for 'u', the cost 'u_cost_' for 'u'.
struct FinalSingleDecision : public Decision {
  FinalSingleDecision(StrategyPtr u_stra, CostPtr u_cost) : u_strategy_(std::move(u_stra)), u_cost_(std::move(u_cost)) {
    type_ = DecisionType::FINAL_SINGLE;
  }

  StrategyPtr u_strategy_;
  CostPtr u_cost_;
  MS_DECLARE_PARENT(FinalSingleDecision, Decision);
};

using DecisionPtr = std::shared_ptr<Decision>;
using OpEliminationDecisionPtr = std::shared_ptr<OpEliminationDecision>;
using EdgeEliminationDecisionPtr = std::shared_ptr<EdgeEliminationDecision>;
using MergeEliminationDecisionPtr = std::shared_ptr<MergeEliminationDecision>;
using ContractEliminationDecisionPtr = std::shared_ptr<ContractEliminationDecision>;
using TriangleEliminationDecisionPtr = std::shared_ptr<TriangleEliminationDecision>;
using StarEliminationDecisionPtr = std::shared_ptr<StarEliminationDecision>;
using FinalDecisionPtr = std::shared_ptr<FinalDecision>;
using FinalSingleDecisionPtr = std::shared_ptr<FinalSingleDecision>;

void Simplify(CostPtrList* clist);
void SimplifyForDreasingCommunicationWithPartialPara(CostPtrList* clist);
void RefineForPracticalCost(const CostPtr&, bool is_redistribution);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_AUTO_PARALLEL_COSTMODEL_H_
