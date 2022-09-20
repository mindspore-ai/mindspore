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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_GRAPH_COSTMODEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_GRAPH_COSTMODEL_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/costmodel_context.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {
class CostGraph;
using CostGraphPtr = std::shared_ptr<CostGraph>;
extern CostGraphPtr entire_costgraph;

class CostGraph {
  // 'CostGraph' consists of Operators and edges between them. An edge is created between two Operators if they have
  // output-input dependency relationship.
 public:
  CostGraph() {}
  ~CostGraph() = default;
  void Init();
  void AddOperator(const OperatorInfoPtr &op) { ops_.push_back(op); }
  OperatorInfoPtr FindOperatorByIndex(size_t index) {
    if (index >= ops_.size()) {
      MS_LOG(ERROR) << "The index: " << index << " is out of the range of ops_: " << ops_.size() << ".";
      return nullptr;
    }
    return ops_[index];
  }
  void RemoveOperator(const OperatorInfoPtr &op);
  bool IsOperatorInCostGraph(const OperatorInfoPtr &op);
  void StrategyPropagate(const std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare> &);
  void BFS(const OperatorInfoPtr &op, const StrategyPtr &op_stra,
           const std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare> &configured_ops,
           std::map<OperatorInfoPtr, bool> *visited) const;
  // the edge is in the form: u --> v
  void AddEdge(OperatorInfoPtr u_node, OperatorInfoPtr v_node, const EdgePtr &edge);
  std::vector<std::shared_ptr<Edge>> GetOriginalPrevEdges(const OperatorInfoPtr &v_node) { return in_edges_[v_node]; }
  std::vector<std::shared_ptr<Edge>> GetOriginalNextEdges(const OperatorInfoPtr &u_node) { return out_edges_[u_node]; }
  // An edge is uniquely identified by its name, and its output index and input index.
  bool IsEdgeInCostGraph(const std::string &, size_t, size_t);

  std::vector<std::shared_ptr<CostGraph>> ConstructConnectedComponents(std::vector<OperatorInfoPtr>);
  void DFS(const OperatorInfoPtr &current_op, std::map<OperatorInfoPtr, bool> *visited,
           const std::shared_ptr<CostGraph> &component);

  CostPtrList CreateFinalCostList(const OperatorInfoPtr &u, const EdgePtr &e, const OperatorInfoPtr &v) const;
  CostPtrList CreateFinalSingleCostList(const OperatorInfoPtr &u) const;
  CostPtr SelectCostWithMinInferenceTime(const CostPtrList &cost_list, double memory) const;
  CostPtr SelectCostWithMinTrainingTime(const CostPtrList &cost_list, double memory) const;
  CostPtrList SelectCostListWithMinTrainingTimeMultiple(const std::vector<CostPtrList> &all_costlist,
                                                        double memory) const;
  Status SearchStrategyForMultiNodeFinalGraph(const std::vector<OperatorInfoPtr> &);
  Status SearchStrategyForTwoNodeFinalGraph(const std::vector<OperatorInfoPtr> &);
  std::vector<std::shared_ptr<Edge>> GetOriginalEdgeBetweenOperators(OperatorInfoPtr u_node, OperatorInfoPtr v_node) {
    return edges_[{u_node, v_node}];
  }

  // Search the cost_list in the final graph, and determine the optimal one
  Status SearchStrategy();

  // Given a graph which contains the following subgraph: u --> v --> w, the node v can be eliminated
  OperatorInfoPtr CheckOpElimination() const;
  // Given a graph which contains the following subgraph where there are multiple edges between u and v, these edges
  // can be eliminated into one
  std::vector<EdgePtr> CheckEdgeElimination() const;
  // Given a graph which contains the following subgraph:
  //        u
  //        |
  //  w --- v --- x
  // where u has 0 incoming edge, u has 1 outgoing edge, and v has > 1 incoming edges, u can be merged into v.
  // u is returned.
  OperatorInfoPtr CheckMergeElimination() const;
  // Given a graph which contains the following subgraph:
  //        u
  //        |
  //        v --- x
  // where v has 2 outgoing edges, and u has 1 incoming edges and no outgoing edges. In this case, u can be contracted
  // into v. u is returned.
  OperatorInfoPtr CheckContractElimination() const;
  /* Given a graph which contains the following subgraph:
   *       u
   *      / \
   *     /   \
   *    v --- w
   * where u has 2 outgoing edges, v has 1 outgoing edge, and w has 2 incoming edges, u can be eliminated into v.
   * The returned value includes u and the edge <u, <v, w>>.
   */
  std::pair<OperatorInfoPtr, EdgePtr> CheckTriangleElimination() const;
  /* Given a graph which contains the following subgraph:
   *  v <--- u ---> w
   * where u has 0 incoming edges, and multiple outgoing edges. In addition, v and w have other complicated connections,
   * resulting in v and w can not be performed ContractElimination. u is returned.
   * NOTE: this elimination MUST be performed only when the above 5 operation cannot be applied.
   */
  OperatorInfoPtr CheckStarElimination() const;
  // Applying Operator Elimination in DP algorithm
  EdgePtr EliminationOp(const OperatorInfoPtr &op) const;
  // Applying Edge Elimination in DP algorithm
  EdgePtr EliminationEdges(const std::vector<EdgePtr> &edges) const;
  // Applying Merge Elimination in DP algorithm
  OperatorInfoPtr EliminationMerge(const OperatorInfoPtr &op) const;
  void CreateMergeEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList &op_cost_list,
                                         const CostPtrList &edge_cost_list, StrategyPtr tar_op_strategy,
                                         const CostPtrList &tar_cost_list, CostPtrList *tar_cost_list_new) const;
  // Applying Contract Elimination in DP algorithm
  OperatorInfoPtr EliminationContract(const OperatorInfoPtr &op) const;
  void CreateContractEliminationSubCostList(StrategyPtr, const CostPtrList &, const CostPtrList &, StrategyPtr,
                                            const CostPtrList &, CostPtrList *) const;

  // Applying Triangle Elimination in DP algorithm. return the left_node
  OperatorInfoPtr EliminationTriangle(const OperatorInfoPtr &elimi_op, const EdgePtr &edge_left_right) const;
  void CreateTriangleEliminationCostList(const OperatorInfoPtr &, const CostPtrList &, const CostPtrList &,
                                         const StrategyPtr &, const StrategyPtr &, const StrategyPtr &,
                                         const CostPtrList &, const CostPtrList &, const CostPtrList &,
                                         CostPtrList *) const;
  // Given the relevant costlist, create the TriangleElimination cost
  void CreateTriangleEliminationSubCostList(StrategyPtr elimi_op_stra, StrategyPtr left_op_stra,
                                            StrategyPtr right_op_stra, const CostPtr &right_op_cost,
                                            const CostPtrList &elimi_op_clist, const CostPtrList &left_edge_clist,
                                            const CostPtr &right_edge_cost, const CostPtrList &left_node_clist_origin,
                                            CostPtrList *left_node_clist_new) const;

  // Applying the Star Elimination in DP algorithm. Return the successive edges of this merged_op
  // NOTE: this elimination MUST be performed only when the above 5 operation cannot be applied.
  std::vector<EdgePtr> EliminationStar(const OperatorInfoPtr &op) const;
  void CreateStarEliminationCostList(std::vector<EdgePtr> &, const StrategyPtr &, const CostPtrList &,
                                     const CostPtrList &, const StrategyPtr &, const CostPtrList &,
                                     CostPtrList *) const;
  void CreateStarEliminationSubCostList(const StrategyPtr &, const CostPtrList &, const CostPtrList &,
                                        const StrategyPtr &, const CostPtrList &, std::vector<StrategyPtr>,
                                        CostPtrList &, CostPtrList &, CostPtrList *) const;
  // Return <op1, op2>. we merge 'op2' into 'op1'
  std::pair<OperatorInfoPtr, OperatorInfoPtr> CheckSourceElimination() const;
  void CreateSourceEliminationSubCostList(StrategyPtr op1_old_stra, const CostPtrList &op1_old_clist,
                                          StrategyPtr op2_old_stra, const CostPtrList &op2_old_clist,
                                          CostPtrList *op1_new_clist) const;
  // We merge 'op2' into op1. The returned value are '<Edges1, Edges2>'. 'Edges1' are newly updated edges for 'op1',
  // 'Edges2' are newly updated edges for 'op2'.
  std::pair<std::vector<std::shared_ptr<Edge>>, std::vector<std::shared_ptr<Edge>>> EliminationSources(
    const OperatorInfoPtr op1, const OperatorInfoPtr op2) const;
  // Calculate memory cost for training phase or inference phase.
  Status CalculateMemoryCost();
  // When the input of a operator is neither a WEIGHT, nor a output of a subsequent operator involving WEIGHT, then
  // the memory cost can be resused. This is used to calculate memory in the training phase.
  Status CalculateOpsMemoryCost();
  // When the input of the edge is neither a WEIGHT, nor a output of a subsequent operator involving WEIGHT, then
  // the memory cost can be reused. This is used to calculate memory in the training phase.
  Status CalculateEdgesMemoryCost();
  // Calculate memory cost of operators in the inference phase.
  Status CalculateOpsMemoryCostForInference();
  // Calculate memory cost of edges in the inference phase.
  Status CalculateEdgesMemoryCostForInference();
  Status ComputeOpsAndEdgesParameterInvolved();
  // Compute for each operator whether the output is critical.
  Status ComputeOpsAndEdgesOutputCritical();

  std::vector<OperatorInfoPtr> GetOperators() const { return ops_; }
  size_t GetNumEdges() const;
  Status InitReshapeStrategy();
  Status InitSelectedStrategy();
  OperatorInfoPtr FindTmpIdentityByParameterName(const std::string &p_name) const;
  // When TmpIdentity is used by multiple operators, the corresponding parameter's memory cost should be calculated only
  // once (instead of multiple times), this method is used to correct this.
  Status CorrectOpsMemoryCost();
  // When APPROXIMATION is enabled in the DP algorithm, some edges may have no valid strategies.
  // This method is to re-init those edge involved operators.
  void CheckApproximateCostGraphEdges();
  // Needed by rec_parser
  void add_inputs_tensor_name(const std::vector<std::string> &inputs_tensor_name) {
    inputs_tensor_name_list_.push_back(inputs_tensor_name);
  }
  const std::vector<std::vector<std::string>> get_inputs_tensor_name_list() const { return inputs_tensor_name_list_; }
  void set_inputs_tensor_name_list(const std::vector<std::vector<std::string>> &inputs_tensor_name_list) {
    inputs_tensor_name_list_ = inputs_tensor_name_list;
  }
  // Needed by rec_parser 2
  void add_param_users_uniqueid(const std::vector<std::string> &param_users_uniqueid) {
    param_users_uniqueid_list_.push_back(param_users_uniqueid);
  }
  const std::vector<std::vector<std::string>> get_param_users_uniqueid_list() const {
    return param_users_uniqueid_list_;
  }
  void add_tuple_getitem(const std::pair<std::string, std::string> &tuple_getitem) {
    auto ret = tuple_getitem_list_.insert(tuple_getitem);
    if (ret.second == false) {
      MS_LOG(EXCEPTION) << "The insert item is already exist.";
    }
  }
  const std::map<std::string, std::string> get_tuple_getitem_list() const { return tuple_getitem_list_; }

 private:
  void TopologyOrder(std::vector<OperatorInfoPtr> *topo_order);
  void DFSForTopoOrder(const OperatorInfoPtr &current_op, std::map<OperatorInfoPtr, bool> *visited,
                       std::vector<OperatorInfoPtr> *topo_order);
  Status DetermineCriticalOps(const std::vector<OperatorInfoPtr> &topo_order);
  void MarkCriticalOpsAndEdges(const std::map<OperatorInfoPtr, int64_t> &candidate_ops);
  // Needed by rec_parser
  std::vector<std::vector<std::string>> inputs_tensor_name_list_;
  // Needed by rec_parser 2
  std::vector<std::vector<std::string>> param_users_uniqueid_list_;
  std::map<std::string, std::string> tuple_getitem_list_;
  std::vector<OperatorInfoPtr> ops_;
  std::map<std::pair<OperatorInfoPtr, OperatorInfoPtr>, std::vector<EdgePtr>> edges_;
  std::vector<std::shared_ptr<CostGraph>> connected_compoents_;
  std::map<OperatorInfoPtr, std::vector<EdgePtr>> out_edges_;
  std::map<OperatorInfoPtr, std::vector<EdgePtr>> in_edges_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_GRAPH_COSTMODEL_H_
