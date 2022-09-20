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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_EDGE_COSTMODEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_EDGE_COSTMODEL_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {
using CostPtrKey = std::pair<StrategyPtr, StrategyPtr>;
using EdgePtr = std::shared_ptr<mindspore::parallel::Edge>;

struct OpsPtrCompare {
  bool operator()(const OperatorInfoPtr &a, const OperatorInfoPtr &b) const { return a->name().compare(b->name()) < 0; }
};

class Edge {
  // An 'Edge' connects two Operators in the CostGraph.
 public:
  Edge(const std::string &edge_name, const std::shared_ptr<OperatorInfo> &prev_op,
       const std::shared_ptr<OperatorInfo> &next_op, const size_t &output_index_, const size_t &input_index_,
       const bool &is_com)
      : edge_name_(edge_name),
        prev_op_(prev_op),
        next_op_(next_op),
        prev_op_output_index_(output_index_),
        next_op_input_index_(input_index_),
        is_combined_(is_com),
        is_identity_edge(false) {}

  Edge(const std::string &edge_name, const std::shared_ptr<OperatorInfo> &prev_op,
       const std::shared_ptr<OperatorInfo> &next_op, const size_t &output_index_, const size_t &input_index_,
       const bool &is_com, const bool &is_iden)
      : edge_name_(edge_name),
        prev_op_(prev_op),
        next_op_(next_op),
        prev_op_output_index_(output_index_),
        next_op_input_index_(input_index_),
        is_combined_(is_com),
        is_identity_edge(is_iden) {}

  Edge(const std::string &edge_name, const std::shared_ptr<OperatorInfo> &prev_op,
       const std::shared_ptr<OperatorInfo> &next_op, const std::vector<size_t> &output_indexs_,
       const std::vector<size_t> &input_indexs_, const bool &is_com)
      : edge_name_(edge_name),
        prev_op_(prev_op),
        next_op_(next_op),
        prev_op_output_index_(0),
        next_op_input_index_(0),
        pre_op_output_indexs_(output_indexs_),
        next_op_input_indexs_(input_indexs_),
        is_combined_(is_com),
        is_identity_edge(false) {}

  ~Edge() = default;
  std::shared_ptr<OperatorInfo> prev_operator() const { return prev_op_; }
  std::shared_ptr<OperatorInfo> next_operator() const { return next_op_; }
  std::string edge_name() const { return edge_name_; }
  // Init cost_map_: for each output layout and input layout, calculate the cost
  Status InitEdgeCost();
  std::map<CostPtrKey, CostPtrList> GetCostMap() { return cost_map_; }
  CostPtr GetCostByStrategyPair(const CostPtrKey &stra_pair);

  StrategyPtr GetNextOpStrategyByPrevOpStrategyWithMiniComm(const StrategyPtr &prev_op_stra);
  StrategyPtr GetPrevOpStrategyByNextOpStrategyWithMiniComm(const StrategyPtr &next_op_stra);
  int64_t GetReshapeSWCIndexByNextOpStrategy(const StrategyPtr &next_op_stra);
  int64_t GetReshapeSWCIndexByPrevOpStrategy(const StrategyPtr &prev_op_stra);
  StrategyPtr GetPrevOpStrategyByReshapeSWCIndex(int64_t swc_index);
  StrategyPtr GetNextOpStrategyByReshapeSWCIndex(int64_t swc_index);
  bool CheckStrategyConsistency(StrategyPtr prev_stra, StrategyPtr next_stra);

  void SetCostMapAndInputOutput(const std::map<CostPtrKey, CostPtrList> &cost_map);
  // For two operators u--->v, given the output tensor layout of u,
  // and the input tensor layout of v, return the redistribution cost,
  // and the op_list to carry out the redistribution.
  Status GetRedistributionCost(const TensorLayout &prev_op_output_layout, const TensorLayout &next_op_input_layout,
                               size_t type_length, const TypePtr &type, CostPtr *cost);

  void set_pre_op_output(const std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> &output_set) {
    pre_op_output_ = output_set;
  }
  void set_next_op_input(const std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> &input_set) {
    next_op_input_ = input_set;
  }

  // Given a pair of output strategy and input strategy, return the corresponding costlist
  CostPtrList GetCostList(StrategyPtr output_str, StrategyPtr input_str);

  std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> prev_op_output() const {
    return pre_op_output_;
  }
  std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> next_op_input() const {
    return next_op_input_;
  }

  bool is_combined() const { return is_combined_; }
  size_t prev_op_output_index() const { return prev_op_output_index_; }
  size_t next_op_input_index() const { return next_op_input_index_; }
  std::vector<size_t> prev_op_output_indexs() const { return pre_op_output_indexs_; }
  std::vector<size_t> next_op_input_indexs() const { return next_op_input_indexs_; }

  CostPtrList CreateEdgeEliminationCostList(const StrategyPtr &output_st_ptr,
                                            const std::vector<std::shared_ptr<Edge>> &edges,
                                            const StrategyPtr &input_st_ptr) const;
  // In the Edge Elimination operation in DP algorithm, 'edges' is replaced by a new edge. This method is used to
  // set cost for this new edge
  void EdgeEliminationSetNewCost(std::shared_ptr<OperatorInfo> u, const std::vector<std::shared_ptr<Edge>> &edges,
                                 std::shared_ptr<OperatorInfo> v);
  void CreateOpEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList &left_cost_list,
                                      const CostPtrList &middle_cost_list, const CostPtrList &right_cost_list,
                                      CostPtrList *ret_cost_list) const;

  CostPtrList CreateOpEliminationCostList(const std::shared_ptr<Edge> &e1, const StrategyPtr &output_st_ptr,
                                          const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &e2,
                                          const StrategyPtr &input_st_ptr) const;
  // In the Operation Elimination operation in DP algorithm, 'op', 'e1' and 'e2' are replaced by a new edge.
  // This method is used to set cost for this new edge
  void OpEliminationSetNewCost(const std::shared_ptr<Edge> &e1, const std::shared_ptr<OperatorInfo> &op,
                               const std::shared_ptr<Edge> &e2);

  void set_selected_cost(const CostPtr &cost) { selected_cost_ = cost; }
  const CostPtr &selected_cost() const { return selected_cost_; }
  void set_parameter_involve(int64_t para_invol) { is_output_parameter_involve_ = para_invol; }
  // In the training phase, when the input of a operator contains WEIGHT or a output from other operators involving
  // WEIGHT, then these input should stay in memory until it is used in the backward phase, which is kept in memory
  // at the end of forward phase.
  Status CalculateMemoryCost();
  // In the inference phase,
  Status CalculateMemoryCostForInference();
  void mark_output_critical() { is_output_critical_ = 1; }
  // Whether there exists any available strategy in 'cost_map_'
  bool CheckStrategyCostPossibility() const;

 private:
  std::string edge_name_;
  std::shared_ptr<OperatorInfo> prev_op_, next_op_;
  std::map<CostPtrKey, CostPtrList> cost_map_;
  // pre_op_output_
  std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> pre_op_output_;
  std::vector<std::pair<std::shared_ptr<Strategy>, std::vector<TensorInfo>>> next_op_input_;
  // the index of outputs of prev_op, and the index of inputs of next_op
  size_t prev_op_output_index_, next_op_input_index_;

  // pre_op_output_indexs_ and next_op_input_indexs_ store the indices of inputs and outputs if is_combined = true
  std::vector<size_t> pre_op_output_indexs_;
  std::vector<size_t> next_op_input_indexs_;
  // is this edge constructed by combining multiple edges? If is is, then is_combined = true, else is_combined = false
  bool is_combined_;
  // When a Parameter in the ANF graph being used by multiple operators, we include the Parameter in the costgraph by
  // replace the Parameter by a TmpIdentity operator, and connecting this TmpIdentity operator with subsequent
  // operators. The resulting edges are different from those normal edges, thus this Bool variable distinguishes them.
  // If it is true, then we should guarantee that the strategy for output tensor consistent with the input tensor.
  bool is_identity_edge;
  CostPtr selected_cost_ = nullptr;
  // In the training phase, 'is_output_parameter_involve_' is used to mark whether the output of the previous operator
  // is parameter-involved
  int64_t is_output_parameter_involve_ = -1;  // -1: unset; 0: not parameter_involved; 1: parameter_involved
  // In the inference phase, this is used to mark whether the output of the previous operator is critical.
  int64_t is_output_critical_ = 0;

  // Returns whether two double variable are equal.
  bool IsDoubleEqual(double x, double y) const { return std::abs(x - y) < EPS; }
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_EDGE_COSTMODEL_H_
