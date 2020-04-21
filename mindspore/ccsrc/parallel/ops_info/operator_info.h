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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPERATOR_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPERATOR_INFO_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/utils.h"
#include "ir/base.h"
#include "parallel/auto_parallel/costmodel.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/device_manager.h"
#include "parallel/device_matrix.h"
#include "parallel/group_manager.h"
#include "parallel/ops_info/ops_utils.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
using ForwardOp = OperatorVector;
using MirrorOps = std::vector<OperatorVector>;
using Ops = std::vector<OperatorVector>;
using VirtualDivOp = OperatorVector;
using TensorMaps = std::vector<std::vector<int32_t>>;
using TensorLayouts = std::vector<TensorLayout>;
using different_type = std::vector<int32_t>::difference_type;
using PrimitiveAttrs = std::unordered_map<std::string, ValuePtr>;
using Strategys = std::vector<Dimensions>;
using ReplaceGraphPtr = std::shared_ptr<std::pair<std::vector<AnfNodePtr>, AnfNodePtr>>;

class Edge;

class OperatorInfo {
 public:
  OperatorInfo(std::string name, Shapes inputs_shape, Shapes outputs_shape, PrimitiveAttrs attrs, OperatorCostPtr cost)
      : name_(std::move(name)),
        inputs_shape_(std::move(inputs_shape)),
        outputs_shape_(std::move(outputs_shape)),
        attrs_(std::move(attrs)),
        is_alive_(true),
        operator_cost_(cost),
        outputs_type_() {
    std::vector<bool> not_parameteter(inputs_shape_.size(), false);
    is_parameter_ = not_parameteter;
    refkey_parameter_name_ = "";
  }

  virtual ~OperatorInfo() = default;

  Status set_is_parameter(const std::vector<bool> &is_parameter);
  Status SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                     const std::vector<size_t> &output_lengths);
  // Set outputs dtype.
  // If only one output, outputs_type.size() is 1.
  // If output is tuple, outputs_type.size() is greater than 1.
  Status set_outputs_type(const std::vector<TypePtr> &outputs_type);
  const std::vector<TypePtr> &outputs_type() const { return outputs_type_; }
  virtual Status Init(const StrategyPtr &strategy) = 0;
  virtual Status InitForCostModel(const StrategyPtr &strategy) = 0;  // only init the necessary parts

  // Given the stage_id (which indicates the number of devices),
  // generate all strategies for this operator
  virtual Status GenerateStrategies(int32_t stage_id) = 0;
  const OperatorCostPtr &operator_cost() const { return operator_cost_; }
  void set_cost(const OperatorCostPtr &cost) { operator_cost_ = cost; }
  virtual Status SetCostUnderStrategy(const StrategyPtr &strategy) = 0;

  virtual std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategies();
  virtual void ReComputeBatchSplitFlagList();
  void ComputeBatchSplitFlagList();

  double GetForwardMemoryCostFromCNode();
  // This is a common method for setting operator cost for a given strategy, in which the validity of this strategy
  // is checked
  Status SetCostUnderStrategyBase(const StrategyPtr &strategy);
  std::vector<std::shared_ptr<StrategyWithCost>> GetStrategyCost() { return strategy_cost_; }
  // When the input of a operator contains WEIGHT or a output from other operators involving WEIGHT, then these input
  // should stay in memory until it is used in the backward phase, which is kept in memory at the end of forward phase.
  Status CalculateMemoryCost();
  int ComputeOpAndPrevEdgeParameterInvolved();

  ForwardOp forward_op() const { return forward_op_; }
  ForwardOp replace_op() const { return replace_op_; }
  OutPutInfoVector replace_op_info() const { return replace_op_info_; }
  virtual ReplaceGraphPtr replace_graph(const CNodePtr &) { return replace_graph_; }
  MirrorOps mirror_ops() const { return mirror_ops_; }
  Ops sub_ops() const { return sub_ops_; }
  VirtualDivOp virtual_div_op() const { return virtual_div_op_; }
  Shape dev_matrix_shape() const { return dev_matrix_shape_; }
  std::vector<TensorInfo> inputs_tensor_info() const { return inputs_tensor_info_; }
  std::vector<TensorInfo> outputs_tensor_info() const { return outputs_tensor_info_; }
  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  RankList global_device_list() const { return global_device_list_; }

  void AddSuccEdge(const std::shared_ptr<Edge> &e) { succ_edges_.push_back(e); }
  void AddPrevEdge(const std::shared_ptr<Edge> &e) { prev_edges_.push_back(e); }
  std::vector<std::shared_ptr<Edge>> succ_edges() const { return succ_edges_; }
  std::vector<std::shared_ptr<Edge>> prev_edges() const { return prev_edges_; }
  std::vector<std::shared_ptr<Edge>> GetAliveSuccEdges();
  std::vector<std::shared_ptr<Edge>> GetAlivePrevEdges();
  void ReplacePreEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge);
  void ReplaceSuccEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge);
  void ReplacePreEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge);
  void ReplaceSuccEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge);
  std::vector<size_t> GetOutputTypeLengths() const { return operator_cost()->outputs_type_lengths(); }
  void SetSelectedStrategyAndCost(const StrategyPtr &s_strategy, const CostPtr &cost) {
    selected_strategy_ = s_strategy;
    selected_cost_ = cost;
  }
  StrategyPtr selected_strategy() const { return selected_strategy_; }
  CostPtr selected_cost() const { return selected_cost_; }
  Status InitSelectedStrategy(const StrategyPtr &s_strategy) { return Init(s_strategy); }
  void set_input_value(const std::vector<ValuePtr> &input_value) { input_value_ = input_value; }
  void set_outputs_dtype(const TypePtr &dtype) { outputs_dtype_ = dtype; }
  void set_cnode(const CNodePtr &cnode) { cnode_ = cnode; }
  bool is_alive() const { return is_alive_; }
  void SetNotAlive() { is_alive_ = false; }
  StrategyPtr strategy() const { return strategy_; }
  void set_strategy(const StrategyPtr &strategy) { strategy_ = strategy; }
  void set_refkey_parameter_name(std::string p_name) { refkey_parameter_name_ = std::move(p_name); }
  const std::string &refkey_parameter_name() const { return refkey_parameter_name_; }
  // When the output of a Parameter (require_grad) being used by multiple operators, the Parameter's cost is calculated
  // multiple times. This method is to correct this, and makes the cost is calulated only once.
  Status CorrectMemoryCost(size_t input_index);
  int is_output_parameter_involve() const { return is_output_parameter_involve_; }
  int used_devices() const { return used_devices_; }
  // needed by rec_parser
  void set_type(const std::string &type) { type_ = type; }
  const std::string &type() const { return type_; }
  void set_cnode_name(const std::string &cnode_name) { cnode_name_ = cnode_name; }
  const std::string &cnode_name() const { return cnode_name_; }
  const std::unordered_map<std::string, ValuePtr> &attrs() const { return attrs_; }

 protected:
  // needed by rec_parser
  std::string type_;
  std::string cnode_name_;
  virtual Status CheckStrategy(const StrategyPtr &strategy) = 0;
  virtual Status InferTensorMap() = 0;
  virtual Status InferForwardCommunication() = 0;
  virtual Status InferMirrorOps() = 0;
  virtual Status GetAttrs() = 0;
  virtual Status InferTensorInfo() = 0;
  virtual Status InferDevMatrixShape() = 0;
  void SetDeviceListByStrategy();
  void SetRepeatedCalcDevMatrix();
  Status CreateGroupByTensorMap(const Shape &tensor_map, std::vector<Group> *group);
  Status CreateGroupByDim(size_t axis, std::vector<Group> *group);
  Status InferAttrs();
  void ResetQueueMember();
  Status InitWithAutoRepeatCalc(const StrategyPtr &strategy);
  Status InitWithManualRepeatCalc(const StrategyPtr &strategy);
  Status InitForCostModelWithAutoRepeatCalc(const StrategyPtr &strategy);
  Status InitForCostModelWithManualRepeatCalc(const StrategyPtr &strategy);
  Status InferRepeatedCalcInfo();
  Status InferVirtualDivOps();

  // Calculate the number of repeated calculations for the output by the number of devices and the output tensor map.
  // The tensor map of Outputs[0] is used by default. If there are multiple outputs, need to identify which output
  // is used for grad and overload the function. If the output is a scalar, need to override the function too.
  virtual Status InferAsLossDivisor();
  Status InferSliceShape(const Strategys &inputs_strategy, const Strategys &outputs_strategy,
                         Shapes *inputs_slice_shape, Shapes *outputs_slice_shape);
  void BreakingTiesForPerferringDataParallel(const StrategyPtr &, const CostPtr &);

  std::string name_;
  Shapes inputs_shape_;
  Shapes outputs_shape_;
  std::unordered_map<std::string, ValuePtr> attrs_;
  std::vector<ValuePtr> input_value_;
  TypePtr outputs_dtype_;

  StrategyPtr strategy_;
  std::vector<TensorInfo> inputs_tensor_info_;
  std::vector<TensorInfo> outputs_tensor_info_;
  Shape dev_matrix_shape_;  // if repeated calculation, it contains the repeated_calc_num as the first dimension
  int32_t repeated_calc_num_ = 1;
  int32_t as_loss_divisor_ = 1;
  TensorMaps inputs_tensor_map_;
  TensorMaps outputs_tensor_map_;
  ForwardOp forward_op_;
  Ops sub_ops_;
  ForwardOp replace_op_;
  OutPutInfoVector replace_op_info_;
  ReplaceGraphPtr replace_graph_;
  MirrorOps mirror_ops_;
  VirtualDivOp virtual_div_op_;
  RankList global_device_list_;  // the size of global_device_list equal to the size of stageID
  RankList local_device_list_;   // the size equal to global_device_list_.size() / repeated_calc_num_
  bool infer_attrs_completed_ = false;

  bool is_auto_parallel_ = false;  // false: semi_auto_parallel; true: auto_parallel
  // 'corrected_input_indices_' used to store the indices of input that have ALREADY been corrected.
  std::vector<size_t> corrected_input_indices_;
  // Given a parallization strategy, there is a cost.
  std::vector<std::shared_ptr<StrategyWithCost>> strategy_cost_;
  // For each input in 'inputs_', there is a bool variable indicating whether that the corresponding input is parameter
  std::vector<bool> is_parameter_;
  // For each input in 'inputs_', a bool variable is true if the corresponding one is a parameter or a output of
  // pre-operator that has parameters as input.
  std::vector<bool> is_parameter_involve_;
  int is_output_parameter_involve_ = -1;  // -1: unset; 0: not parameter_involved; 1: parameter_involved
  // for each input and output, the followings record the number of bytes of each element
  std::vector<size_t> inputs_type_lengths_;
  std::vector<size_t> outputs_type_lengths_;
  std::vector<std::shared_ptr<Edge>> prev_edges_;
  std::vector<std::shared_ptr<Edge>> succ_edges_;
  StrategyPtr selected_strategy_;
  // Used in DP algorithm
  bool is_alive_;
  CostPtr selected_cost_;
  std::vector<bool> split_flag_list_;
  std::string refkey_parameter_name_;
  CNodePtr cnode_;
  int32_t used_devices_ = -1;

 private:
  OperatorCostPtr operator_cost_;
  std::vector<TypePtr> outputs_type_;
};

Shape GetSliceShape(const Shape &tensor_shape, const Dimensions &strategy);
Status CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape, bool);
Operator CreateVirtualDivOp(int32_t div_num);
Operator CreateAllReduceOp(const std::string &reduce_op, const std::string &group);
Operator CreateGetTensorSliceOp(const TensorLayout &tensor_layout);
OperatorVector CreateMirrorOps(const std::string &group_name, size_t dev_num);
int32_t ComputeRepeatDeviceNumByTensorMap(const Shape &dev_matrix_shape, const Shape &tensor_map);
std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategiesBySplitFlag(
  const Shapes &shapes, const std::vector<bool> &split_flag_list);

void PrintStrategy(const StrategyPtr &strategy);
// generate strategies for that all inputs' dimensions are independent, such as: ([a, b, c, d])
Status GenerateStrategiesForIndependentInputs(int32_t stage_id, const Shapes &inputs_shape,
                                              const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp_vector);
// generate strategies for that have two inputs, and input0 or input1 maybe broadcast,
// and the corresponding dimensions that are not broadcast are all relevant dimensions
// such as: ([a, b, c, d], [a, b, c, d]) or ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
// or ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
// or ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesWithBroadcast(int32_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                       std::vector<StrategyPtr> *sp_vector);

Shapes GetRefKeyNodeShape(const AnfNodePtr &node, const FuncGraphPtr &func_graph);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPERATOR_INFO_H_
