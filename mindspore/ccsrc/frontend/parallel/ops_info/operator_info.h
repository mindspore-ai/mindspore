/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_OPERATOR_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_OPERATOR_INFO_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "base/base.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "utils/log_adapter.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace parallel {
using ForwardOp = OperatorVector;
using MirrorOps = std::vector<OperatorVector>;
using Ops = std::vector<OperatorVector>;
using VirtualDivOp = OperatorVector;
using TensorMaps = std::vector<Shape>;
using TensorLayouts = std::vector<TensorLayout>;
using different_type = std::vector<int64_t>::difference_type;
using PrimitiveAttrs = mindspore::HashMap<std::string, ValuePtr>;
using ReplaceGraphPtr = std::shared_ptr<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>;

#define FILTER_LOG(x) (x) ? void(0) : MS_LOG(ERROR)

enum InferStrategyMode {
  SAME_MODE = 0,
  BROADCAST_MODE = 1,
  INDEPENDENT_MODE = 2,
  INDIVIDUAL_MODE = 3,
  INVALID_MODE = 4,
};

class Edge;

class OperatorInfo {
 public:
  OperatorInfo(std::string name, Shapes inputs_shape, Shapes outputs_shape, PrimitiveAttrs attrs,
               const OperatorCostPtr &cost)
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
    stage_device_list_ = g_device_manager->GetDeviceListInThisStage();
    stage_device_size_ = SizeToLong(stage_device_list_.size());
    cnode_ = nullptr;
  }

  virtual ~OperatorInfo() = default;

  Status set_is_parameter(const std::vector<bool> &is_parameter);
  Status SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                     const std::vector<size_t> &output_lengths);
  double GetOutputsTotalSize();
  // Set outputs dtype.
  // If only one output, outputs_type.size() is 1.
  // If output is tuple, outputs_type.size() is greater than 1.
  Status set_outputs_type(const std::vector<TypePtr> &outputs_type);
  const std::vector<TypePtr> &outputs_type() const { return outputs_type_; }
  virtual Status Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy);
  // only init the necessary parts
  virtual Status InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy);

  // Given the stage_id (which indicates the number of devices),
  // generate all strategies for this operator
  Status GenerateStrategies(int64_t stage_id);
  virtual std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) = 0;
  const OperatorCostPtr &operator_cost() const { return operator_cost_; }
  void set_cost(const OperatorCostPtr &cost) { operator_cost_ = cost; }
  virtual Status SetCostUnderStrategy(const StrategyPtr &strategy) = 0;

  virtual std::shared_ptr<Strategies> GenerateBatchStrategies();
  virtual void ReComputeBatchSplitFlagList();
  std::shared_ptr<Strategies> GenerateBatchStrategiesWithCheck();
  void ComputeBatchSplitFlagList();
  Shapes inputs_shape() const { return inputs_shape_; }

  double GetForwardMemoryCostFromCNode();
  // This is a common method for setting operator cost for a given strategy, in which the validity of this strategy
  // is checked
  Status SetCostUnderStrategyBase(const StrategyPtr &strategy);
  CostPtrList GetCostByStrategyPtr(const StrategyPtr &strategy);
  std::vector<std::shared_ptr<StrategyWithCost>> GetStrategyCost() { return strategy_cost_; }
  void SetStrategyCost(const std::vector<std::shared_ptr<StrategyWithCost>> &stra_cost);
  // In the training phase, when the input of a operator contains WEIGHT or a output from other operators involving
  // WEIGHT, then these input should stay in memory until it is used in the backward phase, which is kept in memory
  // at the end of forward phase.
  Status CalculateMemoryCost();
  // In the inference phase, the memory cost is incurred only when the operator is critical. The size is calculated
  // by the output
  Status CalculateMemoryCostForInference();
  virtual int64_t ComputeOpAndPrevEdgeParameterInvolved();

  ForwardOp forward_op() const { return forward_op_; }
  ForwardOp replace_op() const { return replace_op_; }
  OutPutInfoVector replace_op_info() const { return replace_op_info_; }
  virtual ReplaceGraphPtr replace_graph(const CNodePtr &) { return replace_graph_; }
  MirrorOps mirror_ops() const { return mirror_ops_; }
  Ops sub_ops() const { return sub_ops_; }
  VirtualDivOp virtual_div_op() const { return virtual_div_op_; }
  Shape dev_matrix_shape() const { return dev_matrix_shape_; }
  std::vector<TensorInfo> inputs_tensor_info() const { return inputs_tensor_info_; }
  void set_inputs_tensor_info(const std::vector<TensorInfo> &tensor_info) { inputs_tensor_info_ = tensor_info; }
  std::vector<TensorInfo> outputs_tensor_info() const { return outputs_tensor_info_; }
  std::vector<std::shared_ptr<StrategyWithCost>> strategy_cost() const { return strategy_cost_; }
  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  RankList stage_device_list() const { return stage_device_list_; }

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
  void SetSelectedStrategy(const StrategyPtr &s_strategy, size_t curr_depth);
  StrategyPtr selected_strategy() const { return selected_strategy_; }
  CostPtr selected_cost() const { return selected_cost_; }

  TensorLayout GetInputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t input_index);
  TensorLayout GetOutputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t output_index);
  StrategyPtr GetStrategyFromSWCByInputLayout(const TensorLayout &input_layout, size_t input_index);
  StrategyPtr GetStrategyFromSWCByOutputLayout(const TensorLayout &output_layout, size_t output_index);
  bool IsReshape() const;
  bool IsTmpIdentity() const;

  void set_swc_index(int64_t swc, int64_t depth);
  int64_t swc_index() const { return swc_index_; }
  // Approximate the list of available strategies
  void ApproximateStrategies();
  // Make the list of available strategies exact and re-init the related edges incident to this operator
  void ExactStrategiesAndRelatedEdges();
  bool is_strategy_cost_exact() const { return is_strategy_cost_exact_; }
  void SetIsStrategyCostExactTrue() { is_strategy_cost_exact_ = true; }
  void ClearStrategyCost() { strategy_cost_.clear(); }
  void CheckSelectedStrategy(const StrategyPtr &s_strategy);
  Status InitSelectedStrategy(const StrategyPtr &s_strategy) {
    set_auto_parallel(false);
    return Init(s_strategy, nullptr);
  }
  void set_input_value(const std::vector<ValuePtr> &input_value) { input_value_ = input_value; }
  const std::vector<ValuePtr> &input_value() const { return input_value_; }
  void set_outputs_dtype(const TypePtr &dtype) { outputs_dtype_ = dtype; }
  void set_cnode(const CNodePtr &cnode) {
    cnode_ = cnode;
    cnodes_.push_back(cnode);
  }
  std::vector<CNodePtr> cnodes();
  bool is_alive() const { return is_alive_; }
  void SetNotAlive() { is_alive_ = false; }
  StrategyPtr strategy() const { return strategy_; }
  StrategyPtr out_strategy() const { return out_strategy_; }
  void set_strategy(const StrategyPtr &strategy) { strategy_ = strategy; }
  void set_refkey_parameter_name(std::string p_name) { refkey_parameter_name_ = std::move(p_name); }
  const std::string &refkey_parameter_name() const { return refkey_parameter_name_; }
  // When the output of a Parameter (require_grad) being used by multiple operators, the Parameter's cost is calculated
  // multiple times. This method is to correct this, and makes the cost is calculated only once.
  Status CorrectMemoryCost(size_t input_index);
  int64_t is_output_parameter_involve() const { return is_output_parameter_involve_; }
  int64_t is_output_critical() const { return is_output_critical_; }
  void mark_output_critical() { is_output_critical_ = 1; }
  void mark_output_not_critical() { is_output_critical_ = 0; }
  int64_t used_devices() const { return used_devices_; }
  // needed by rec_parser
  void set_type(const std::string &type) { type_ = type; }
  const std::string &type() const { return type_; }
  void set_last_node_flag(const bool &is_last_node) { is_last_node_ = is_last_node; }
  const bool &is_last_node() const { return is_last_node_; }
  const mindspore::HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }
  void addAttr(const std::string &name, const ValuePtr &val) { attrs_[name] = val; }
  void set_stage_id(int32_t stage_id) { stage_id_ = stage_id; }
  int32_t stage_id() const { return stage_id_; }
  Status CreateGroupByTensorMap(const Shape &tensor_map, std::vector<Group> *group);
  Status CreateGroupForOptShard(TensorLayout *tensor_layout, std::vector<Group> *groups);
  virtual void ReplaceNodeInputOrAttrs() {}
  void set_auto_parallel(bool is_auto_parallel) { is_auto_parallel_ = is_auto_parallel; }

  // Key for user data.
  constexpr static char key[] = "OpInfo";

 protected:
  // needed by rec_parser
  std::string type_;
  bool is_last_node_ = false;
  virtual Status CheckStrategy(const StrategyPtr &strategy) = 0;
  virtual Status InferTensorMap() = 0;
  virtual Status InferForwardCommunication() = 0;
  virtual Status GetAttrs() = 0;
  virtual Status InferDevMatrixShape() = 0;
  virtual Status InferMirrorOps();
  virtual Status InferTensorInfo();
  virtual void InferReplaceOps() {}
  virtual Status CheckOutputStrategy(const StrategyPtr &out_strategy);
  Status CheckStrategyByVector(const Shapes &strategy, const Shapes &inputs_shape);
  Status CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape);
  void SetRepeatedCalcDevMatrix();
  void ResetTensorMapIfRepeatedCalc();
  Status CreateGroupByDim(size_t axis, std::vector<Group> *group);
  virtual Status InferAttrs();
  void ResetQueueMember();
  Status InitWithAutoRepeatCalc(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy);
  Status InitForCostModelWithAutoRepeatCalc(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy);
  Status InferRepeatedCalcInfo();
  Status InferVirtualDivOps();

  // Calculate the number of repeated calculations for the output by the number of devices and the output tensor map.
  // The tensor map of Outputs[0] is used by default. If there are multiple outputs, need to identify which output
  // is used for grad and overload the function. If the output is a scalar, need to override the function too.
  virtual Status InferAsLossDivisor();
  Status InferSliceShape(const Strategies &inputs_strategy, const Strategies &outputs_strategy,
                         Shapes *inputs_slice_shape, Shapes *outputs_slice_shape);
  void BreakingTiesForPreferringDataParallel(const StrategyPtr &stra, const CostPtr &cost) const;
  int64_t GetIntAttr(const std::string &attr_name);
  bool GetBoolAttr(const std::string &attr_name);
  float GetFloatAttr(const std::string &attr_name);
  std::string GetStringAttr(const std::string &attr_name);
  std::vector<int64_t> GetTupleIntAttr(const std::string &attr_name);
  void ReportError(const std::string &error_msg) const {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << error_msg;
    } else {
      MS_LOG(ERROR) << error_msg;
    }
  }

  std::string name_;
  Shapes inputs_shape_;
  Shapes outputs_shape_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  std::vector<ValuePtr> input_value_;
  TypePtr outputs_dtype_;

  int32_t stage_id_ = 0;
  StrategyPtr strategy_;
  StrategyPtr out_strategy_;
  std::vector<TensorInfo> inputs_tensor_info_;
  std::vector<TensorInfo> outputs_tensor_info_;
  Shape dev_matrix_shape_;  // if repeated calculation, it contains the repeated_calc_num_
  Shape out_dev_matrix_shape_;
  int64_t repeated_calc_num_ = 1;
  int64_t as_loss_divisor_ = 1;
  TensorMaps inputs_tensor_map_;
  TensorMaps outputs_tensor_map_;
  ForwardOp forward_op_;
  Ops sub_ops_;
  ForwardOp replace_op_;
  OutPutInfoVector replace_op_info_;
  ReplaceGraphPtr replace_graph_;
  MirrorOps mirror_ops_;
  VirtualDivOp virtual_div_op_;
  RankList stage_device_list_;  // the device list in this stage
  int64_t stage_device_size_ = 0;
  bool infer_attrs_completed_ = false;

  bool is_auto_parallel_ = false;  // false: semi_auto_parallel; true: auto_parallel
  // 'corrected_input_indices_' used to store the indices of input that have ALREADY been corrected.
  std::vector<size_t> corrected_input_indices_;
  // Given a parallelization strategy, there is a cost.
  std::vector<std::shared_ptr<StrategyWithCost>> strategy_cost_;
  // For each input in 'inputs_', there is a bool variable indicating whether that the corresponding input is parameter
  std::vector<bool> is_parameter_;
  // For each input in 'inputs_', a bool variable is true if the corresponding one is a parameter or a output of
  // pre-operator that has parameters as input.
  std::vector<bool> is_parameter_involve_;
  // If any input is parameter-involved, the output is parameter-involved. This variable is used in calculating
  // peak memory cost in the training phase.
  // -1: unset; 0: not parameter_involved; 1: parameter_involved
  int64_t is_output_parameter_involve_ = -1;
  // Whether this output is critical, which means that this output is included in calculating peak memory cost
  // in the inference phase.
  // -1 : unset; 0: not critical; 1: critical
  int64_t is_output_critical_ = -1;
  double outputs_total_size_ = 0.0;
  bool is_calculated_outputs_size_ = false;
  // for each input and output, the followings record the number of bytes of each element
  std::vector<size_t> inputs_type_lengths_;
  std::vector<size_t> outputs_type_lengths_;
  std::vector<std::shared_ptr<Edge>> prev_edges_;
  std::vector<std::shared_ptr<Edge>> succ_edges_;
  StrategyPtr selected_strategy_;
  int64_t selected_strategy_depth_ = -1;
  // Used in DP algorithm
  bool is_alive_;
  CostPtr selected_cost_;
  std::vector<bool> split_flag_list_;
  std::string refkey_parameter_name_;
  CNodePtr cnode_;
  std::vector<CNodePtr> cnodes_;
  int64_t used_devices_ = -1;
  // the repeated_calc_num_ will be inserted to the last dimension of dev matrix in default
  bool repeated_num_in_dev_matrix_right_ = true;
  // Whether the list of available strategies is exact or approximate
  bool is_strategy_cost_exact_ = true;

 private:
  OperatorCostPtr operator_cost_;
  std::vector<TypePtr> outputs_type_;
  int64_t swc_index_ = -1;
};

Shape GetSliceShape(const Shape &tensor_shape, const Dimensions &strategy);
Status CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape, bool);
Operator CreateVirtualDivOp(int64_t div_num);
Operator CreateAllReduceOp(const std::string &reduce_op, const std::string &group);
Operator CreateReduceScatterOp(const std::string &reduce_op, const std::string &group);
Operator CreateAllGatherOp(const std::string &group);
Operator CreateCastOp(TypePtr type);
Operator CreateDivOp(float scale);
Operator CreateMiniStepAllGatherOp(const std::string &group);
void AddCNodePrimAttr(const CNodePtr &comm_node, const std::string &attr_name, const ValuePtr &attr_val);
int32_t AddCommOpFusionType(const CNodePtr &comm_node, const AnfNodePtr &param_node);
Operator CreateMicroStepAllGatherOp(const std::string &group);
void AddCommOpMeanFlag(const CNodePtr &comm_node);
void AddCommOpParamFlag(const CNodePtr &comm_node);
Operator CreateGetTensorSliceOp(const TensorLayout &tensor_layout);
OperatorVector CreateMirrorOps(const std::string &group_name, size_t dev_num);
int64_t ComputeRepeatDeviceNumByTensorMap(const Shape &dev_matrix_shape, const Shape &tensor_map);
std::shared_ptr<Strategies> GenerateBatchStrategiesBySplitFlag(const Shapes &shapes,
                                                               const std::vector<bool> &split_flag_list);
std::string StrategyToString(const Strategies &strategy);
Status GenerateStrategiesForIndependentInputsBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_shape,
                                                  const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp_vector);
// generate strategies for that all inputs' dimensions are independent, such as: ([a, b, c, d])
Status GenerateStrategiesForIndependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                              const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp_vector);
// generate strategies for that inputs' dimension maybe dependent
Status GenerateStrategiesForDependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                            const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp);
// generate strategies for that have two inputs, and input0 or input1 maybe broadcast,
// and the corresponding dimensions that are not broadcast are all relevant dimensions
// such as: ([a, b, c, d], [a, b, c, d]) or ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
// or ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
// or ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesWithBroadcast(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                       std::vector<StrategyPtr> *sp_vector);
std::vector<ValuePtr> GetValueSequence(const ValuePtr &sequence);
ValuePtr MakeListValue(const std::vector<int64_t> &v);
ValuePtr MakeTupleListValue(const Shapes &v);
AnfNodePtr CreateValueTupleAnfNodePtr(const std::vector<int64_t> &value_tuple);
AnfNodePtr CreateTensorTupleAnfNodePtr(const tensor::TensorPtrList &tensor_tuple);

ForwardOp CreateReduceMeanForwardOp(const std::vector<Group> &forward_group, const TypePtr &dtype);
Operator CreateDivOpWithType(float divisor, const TypePtr &dtype);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_OPERATOR_INFO_H_
