/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/operator_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <numeric>

#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"
#include "utils/log_adapter.h"
#include "include/common/debug/anf_dump_utils.h"

namespace mindspore {
namespace parallel {
namespace {
struct InStrategyValueRegister {
  InStrategyValueRegister() noexcept {
    AnfDumpHandler::SetInStrategyValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputDim());
    });
  }
} in_regist;

struct InStrategyStageValueRegister {
  InStrategyStageValueRegister() noexcept {
    AnfDumpHandler::SetInStrategyStageValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputStage());
    });
  }
} in_stage_regist;

struct OutStrategyValueRegister {
  OutStrategyValueRegister() noexcept {
    AnfDumpHandler::SetOutStrategyValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->out_strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputDim());
    });
  }
} out_regist;
}  // namespace

std::string StrategyToString(const Strategies &strategy) {
  std::string strategy_str = "";
  strategy_str += "(";
  for (size_t i = 0; i < strategy.size(); ++i) {
    strategy_str += "(";
    for (size_t j = 0; j < strategy[i].size(); ++j) {
      strategy_str += std::to_string(strategy[i][j]);
      if (j != strategy[i].size() - 1) {
        strategy_str += ", ";
      }
    }
    strategy_str += ")";
    if (i != strategy.size() - 1) {
      strategy_str += ", ";
    }
  }
  if (strategy.size() == 1) {
    strategy_str += ",";
  }
  strategy_str += ")";
  return strategy_str;
}

Status OperatorInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy) {
    MS_LOG(ERROR) << name_ << ": It does not support to set output strategy now, please modify the shard set";
    return FAILED;
  }
  return SUCCESS;
}

Status OperatorInfo::CheckStrategyByVector(const Shapes &stra, const Shapes &inputs_shape) {
  size_t strategy_size = stra.size();
  size_t inputs_shape_size = inputs_shape.size();
  if (strategy_size != inputs_shape_size) {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", strategy size: " << strategy_size
                  << " is not equal to inputs size: " << inputs_shape_size;
    return FAILED;
  }

  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    Shape sub_input_shape = inputs_shape.at(i);
    size_t strategy_len = sub_strategy.size();
    size_t inputs_len = sub_input_shape.size();
    if (strategy_len != inputs_len) {
      MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", strategy len: " << strategy_len
                    << " is not equal to inputs len: " << inputs_len << ", index: " << i;
      return FAILED;
    }

    for (size_t j = 0; j < strategy_len; ++j) {
      int64_t strategy_value = sub_strategy.at(j);
      if (strategy_value < MIN_SLICE_NUM) {
        MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra)
                      << ", the value of strategy must be larger than 0, but get " << strategy_value;
        return FAILED;
      }

      int64_t shape_value = sub_input_shape.at(j);
      if ((shape_value % strategy_value) != 0) {
        MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", shape " << shape_value
                      << " cannot be divisible by strategy value " << strategy_value;
        return FAILED;
      }

      if ((LongToUlong(strategy_value) & LongToUlong(strategy_value - 1)) != 0) {
        if ((g_device_manager->DeviceNum() & (g_device_manager->DeviceNum() - 1)) != 0) {
          MS_LOG(INFO) << name_
                       << ": The device num is not the power of 2, thus do not check the strategy as power of 2";
          continue;
        }
        MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra)
                      << ", the value of strategy must be the power of 2, but get " << strategy_value;
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status OperatorInfo::CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  return CheckStrategyByVector(stra, inputs_shape);
}

void OperatorInfo::ResetQueueMember() {
  inputs_tensor_info_.clear();
  outputs_tensor_info_.clear();
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();
  dev_matrix_shape_.clear();
  forward_op_.clear();
  mirror_ops_.clear();
  sub_ops_.clear();
  replace_op_.clear();
  replace_op_info_.clear();
  virtual_div_op_.clear();
}

Status OperatorInfo::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }

  if (GetAttrs() != SUCCESS) {
    return FAILED;
  }
  infer_attrs_completed_ = true;
  return SUCCESS;
}

Status OperatorInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
    return FAILED;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    std::vector<Group> group;
    if (CreateGroupByTensorMap(inputs_tensor_map_[i], &group) != SUCCESS) {
      ReportError(name_ + ": Create group failed, the input index is " + std::to_string(i));
      mirror_ops_.clear();
      return FAILED;
    }

    OperatorVector mirror_op;
    if (group.empty()) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    group_is_empty = false;
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

Status OperatorInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }

  return SUCCESS;
}

Status OperatorInfo::InferRepeatedCalcInfo() {
  int64_t g_dev_list_size = stage_device_size_;
  int64_t dev_matrix_size =
    std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
  if (dev_matrix_size == 0) {
    MS_LOG(ERROR) << name_ << ": The dev matrix size is 0";
    return FAILED;
  }

  if (g_dev_list_size == dev_matrix_size) {
    repeated_calc_num_ = 1;
  } else if (g_dev_list_size % dev_matrix_size == 0) {
    repeated_calc_num_ = g_dev_list_size / dev_matrix_size;
  } else {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(strategy_->GetInputDim()) << ", it requires "
                  << dev_matrix_size << " devices, "
                  << "but the device number of this stage is " << g_dev_list_size << ", it can not be divisible by "
                  << dev_matrix_size;
    return FAILED;
  }
  return SUCCESS;
}

// If repeated calculation, set the repeated_calc_num as the last dimension of dev-matrix in default,
// because if the previous shard is (a, b), and the next shard is (a, 1), adding the repeated_calc_num
// to the last dimension of dev-matrix, there is no need to redistribution.
void OperatorInfo::SetRepeatedCalcDevMatrix() {
  if (repeated_calc_num_ <= 1) {
    return;
  }
  if (repeated_num_in_dev_matrix_right_) {
    dev_matrix_shape_.push_back(repeated_calc_num_);
  } else {
    (void)dev_matrix_shape_.insert(dev_matrix_shape_.cbegin(), repeated_calc_num_);
  }
}

// If repeated calculation, and the repeated_calc_num is inserted to the last dimension of the dev-matrix,
// the index value of tensor map needs to be increased by 1.
void OperatorInfo::ResetTensorMapIfRepeatedCalc() {
  if ((repeated_calc_num_ <= 1) || !repeated_num_in_dev_matrix_right_) {
    return;
  }

  MS_LOG(DEBUG) << name_ << ": the repeated calc num is " << repeated_calc_num_ << ", and reset the tensor maps";
  for (auto &tensor_map : inputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }

  for (auto &tensor_map : outputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }
}

// use for loss repeated calculation
Operator CreateVirtualDivOp(int64_t div_num) {
  OperatorName operator_name = VIRTUAL_DIV;
  ValuePtr attr0_value = MakeValue(div_num);
  Attr attr0 = std::make_pair(DIVISOR, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

Operator CreateDivOp(float scale) {
  OperatorName operator_name = REAL_DIV;
  OperatorAttrs operator_attrs;
  OperatorParams operator_param;
  constexpr size_t parameter_pos = 2;
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(scale);
  ValuePtr scale_value = MakeValue(tensor_ptr);
  (void)operator_param.emplace_back(std::make_pair(std::make_pair(Y, scale_value), parameter_pos));
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

static OperatorArgs CreateReduceCommunicationOpArgs(const std::string &reduce_op, const std::string &group) {
  ValuePtr attr0_value = MakeValue(reduce_op);
  ValuePtr attr1_value = MakeValue(group);
  Attr attr0 = std::make_pair(OP, attr0_value);
  Attr attr1 = std::make_pair(GROUP, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  return std::make_pair(operator_attrs, operator_param);
}

// use for forward all reduce
Operator CreateAllReduceOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = ALL_REDUCE;
  OperatorArgs operator_arg = CreateReduceCommunicationOpArgs(reduce_op, group);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create all reduce op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

Operator CreateReduceScatterOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = REDUCE_SCATTER;
  OperatorArgs operator_arg = CreateReduceCommunicationOpArgs(reduce_op, group);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create reduce scatter op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

Operator CreateCastOp(TypePtr type) {
  Param param_type = std::make_pair(std::make_pair(DTYPE, type), 2);
  OperatorAttrs attrs;
  OperatorParams params = {param_type};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op_cast = std::make_pair(CAST, args);
  return op_cast;
}

int32_t AddCommOpFusionType(const CNodePtr &comm_node, const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  MS_EXCEPTION_IF_NULL(param_node);
  ParameterPtr param;
  if (IsPrimitiveCNode(param_node, prim::kPrimReceive)) {
    param = param_node->user_data<AnfNode>(PIPELINE_PARAM)->cast<ParameterPtr>();
  } else {
    param = param_node->cast<ParameterPtr>();
  }
  MS_EXCEPTION_IF_NULL(param);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  auto param_info = param->param_info();
  if (!param_info) {
    MS_LOG(WARNING) << param->ToString() << "does not have parameter info.";
    return 0;
  }
  int32_t fusion_type = param_info->comm_fusion();
  attrs[FUSION] = MakeValue<int64_t>(fusion_type);
  (void)prim->SetAttrs(attrs);
  bool parallel_optimizer_comm_recompute = param_info->parallel_optimizer_comm_recompute();
  std::string instance_name = prim->instance_name();
  if (instance_name == PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE && parallel_optimizer_comm_recompute &&
      prim->name() == ALL_GATHER) {
    prim->set_attr(RECOMPUTE, MakeValue(true));
    prim->set_instance_name(PARALLEL_OPTIMIZER_ALLGATHER);
  }
  MS_LOG(INFO) << "Set comm fusion:" << param->param_info()->name() << "'s fusion type is " << fusion_type;
  return fusion_type;
}

void AddCommOpMeanFlag(const CNodePtr &comm_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  auto attrs = prim->attrs();
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  attrs[MEAN_FLAG] = MakeValue<bool>(mean_flag);
  (void)prim->SetAttrs(attrs);
}

void AddCNodePrimAttr(const CNodePtr &comm_node, const std::string &attr_name, const ValuePtr &attr_val) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  auto attrs = prim->attrs();
  attrs[attr_name] = attr_val;
  (void)prim->SetAttrs(attrs);
}

void AddCommOpParamFlag(const CNodePtr &comm_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[comm_node->input(1)];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimSend)) {
      auto prim = GetCNodePrimitive(comm_node);
      (void)prim->AddAttr(PARAMETER_MICRO, MakeValue(0));
      return;
    }
  }
}

Operator CreateAllGatherOp(const std::string &group) {
  OperatorName operator_name = ALL_GATHER;
  ValuePtr attr0_value = MakeValue(group);  // group
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create allgather op success, the group is " << group;
  return op;
}

Operator CreateMiniStepAllGatherOp(const std::string &group) {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();

  OperatorName operator_name = MINI_STEP_ALL_GATHER;
  ValuePtr attr0_value = MakeValue(group);  // group
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  ValuePtr attr1_value = MakeValue(grad_accumulation_step);  // grad_accumulation_step
  Attr attr1 = std::make_pair(GRAD_ACCUMULATION_STEP, attr1_value);
  ValuePtr attr2_value = MakeValue(mean_flag);  // mean_flag
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create MINI_STEP_ALL_GATHER success, the group is " << group;
  return op;
}

Operator CreateMicroStepAllGatherOp(const std::string &group) {
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  OperatorName operator_name = MICRO_STEP_ALL_GATHER;
  ValuePtr attr0_value = MakeValue(group);  // group
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  ValuePtr attr1_value = MakeValue(mean_flag);  // mean_flag
  Attr attr1 = std::make_pair(MEAN_FLAG, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create MICRO_STEP_ALL_GATHER success, the group is " << group;
  return op;
}

// use for get tensor slice
Operator CreateGetTensorSliceOp(const TensorLayout &tensor_layout) {
  Shape tensor_map = tensor_layout.tensor_map().array();
  Shape dev_matrix_shape = tensor_layout.device_arrangement().array();
  OperatorName operator_name = GET_TENSOR_SLICE;

  OperatorAttrs attrs;
  ValuePtr dev_mat_value = MakeValue(dev_matrix_shape);
  Param dev_mat_param = std::make_pair(std::make_pair(DEV_MAT, dev_mat_value), 2);
  ValuePtr tensor_map_value = MakeValue(tensor_map);
  Param tensor_map_param = std::make_pair(std::make_pair(TENSOR_MAP, tensor_map_value), 3);
  OperatorParams params = {dev_mat_param, tensor_map_param};
  OperatorArgs operator_arg = std::make_pair(attrs, params);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create get tensor slice op success, the dev mat and tensor map is "
               << ShapeToString(dev_matrix_shape) << ", " << ShapeToString(tensor_map);
  return op;
}

OperatorVector CreateMirrorOps(const std::string &group_name, size_t dev_num) {
  if (dev_num == 0) {
    MS_LOG(EXCEPTION) << "Invalid dev num: " << dev_num;
  }
  OperatorVector op_for_weight;
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  ValuePtr attr0_value = MakeValue(group_name);
  ValuePtr attr1_value = MakeValue(SizeToLong(dev_num));
  ValuePtr attr2_value = MakeValue(mean_flag);

  Attr attr0 = std::make_pair(GROUP, attr0_value);
  Attr attr1 = std::make_pair(DEV_NUM, attr1_value);
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);

  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorName operator_name;
  if (grad_accumulation_step > 1) {
    operator_name = MIRROR_MINI_STEP_OPERATOR;
    ValuePtr attr3_value = MakeValue(grad_accumulation_step);
    Attr attr3 = std::make_pair(GRAD_ACCUMULATION_STEP, attr3_value);
    operator_attrs.push_back(attr3);
    MS_LOG(INFO) << "The grad accumulation step is " << grad_accumulation_step << ", use mini step mirror";
  } else if (split_stage_num > 1) {
    operator_name = MIRROR_MICRO_STEP_OPERATOR;
  } else {
    operator_name = MIRROR_OPERATOR;
  }

  OperatorParams operator_param;
  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_args);

  op_for_weight.push_back(op);
  MS_LOG(INFO) << "The group name is " << group_name << ", the dev num is " << dev_num << ", the mean flag is "
               << mean_flag;
  return op_for_weight;
}

Status OperatorInfo::CreateGroupByTensorMap(const Shape &tensor_map, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }
  if (is_auto_parallel_) {
    if (g_device_manager->CheckDeviceList(group_devices) != SUCCESS) {
      MS_LOG(INFO) << name_ << ": Try to create communication group : " << group_devices
                   << " failed in auto parallel mode, "
                      "this error can be ignored in parallel strategies searching step";
      return FAILED;
    }
    return SUCCESS;
  }

  Group g;
  if (g_device_manager->CreateGroup(group_devices, &g) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create communication group by tensor_map failed, the rank_list is: " << group_devices
                  << ", the input strategy is " << strategy_->GetInputDim()
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }
  group->push_back(g);
  return SUCCESS;
}

Status OperatorInfo::CreateGroupForOptShard(TensorLayout *tensor_layout, std::vector<Group> *groups) {
  if (groups == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  Shape tensor_map = tensor_layout->origin_tensor_map().array();
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }
  int64_t optimizer_weight_shard_size = ParallelContext::GetInstance()->optimizer_weight_shard_size();
  if (optimizer_weight_shard_size != -1) {
    // not fully use opt shard
    int64_t index = std::find(group_devices.begin(), group_devices.end(), rank) - group_devices.begin();
    int64_t repeated_size = SizeToLong(group_devices.size());
    if (repeated_size % optimizer_weight_shard_size != 0) {
      MS_LOG(WARNING) << name_ << ": Parallel optimizer: optimizer_weight_shard_size " << optimizer_weight_shard_size
                      << " can not be applied. The repeated size is " << repeated_size;
      return FAILED;
    }
    repeated_size = repeated_size / optimizer_weight_shard_size;
    // create allgather group
    // eg: optimizer_weight_shard_size = 2, [0, 8, 16, 24] -> [0, 8], [16, 24]
    RankList new_group_devices(
      group_devices.begin() + index / optimizer_weight_shard_size * optimizer_weight_shard_size,
      group_devices.begin() + (index / optimizer_weight_shard_size + 1) * optimizer_weight_shard_size);
    Group allgather_group;
    if (g_device_manager->CreateGroup(new_group_devices, &allgather_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for allgather in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(allgather_group);
    tensor_layout->set_opt_shard_group(allgather_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create allgather group " << allgather_group.name();
    // create mirror group
    // eg: optimizer_weight_shard_size = 2, [0, 8, 16, 24] -> [0, 16], [8, 24]
    int64_t device_num = g_device_manager->stage_device_num();
    Shape dev_mat = {repeated_size, device_num / repeated_size};
    DeviceMatrix temp_dev_matrix(rank, stage_device_list_, dev_mat);
    RankList mirror_group_devices;
    if (temp_dev_matrix.GetDevicesAlongDim(0, &mirror_group_devices) != SUCCESS) {
      return FAILED;
    }
    Group mirror_group;
    if (g_device_manager->CreateGroup(mirror_group_devices, &mirror_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for mirror in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(mirror_group);
    tensor_layout->set_opt_shard_mirror_group(mirror_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create mirror group " << mirror_group.name();
  } else {
    // fully use opt shard
    // create allgather group
    Group allgather_group;
    if (g_device_manager->CreateGroup(group_devices, &allgather_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for allgather in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(allgather_group);
    tensor_layout->set_opt_shard_group(allgather_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create allgather group " << allgather_group.name();
  }
  // save in tensor_layout for strategy ckpt
  auto integrated_save = ParallelContext::GetInstance()->optimizer_weight_shard_aggregated_save();
  if (!integrated_save) {
    tensor_layout->set_opt_weight_shard_size(LongToInt(optimizer_weight_shard_size));
    int64_t opt_weight_shard_step =
      (group_devices.back() - group_devices.front()) / (SizeToLong(group_devices.size()) - 1);
    tensor_layout->set_opt_weight_shard_step(LongToInt(opt_weight_shard_step));
    MS_LOG(INFO) << name_ << "Parallel optimizer, save opt_weight_shard_step " << opt_weight_shard_step
                 << " in strategy ckpt";
  }
  return SUCCESS;
}

Status OperatorInfo::CreateGroupByDim(size_t axis, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(axis), &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }
  if (is_auto_parallel_) {
    if (g_device_manager->CheckDeviceList(group_devices) != SUCCESS) {
      MS_LOG(INFO) << name_ << ": Try to create communication group : " << group_devices
                   << " failed in auto parallel mode, "
                      "this error can be ignored in parallel strategies searching step";
      return FAILED;
    }
    return SUCCESS;
  }
  Group g;
  if (g_device_manager->CreateGroup(group_devices, &g) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create communication group by dim failed, the rank_list is: " << group_devices
                  << ", the input strategy is " << strategy_->GetInputDim()
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }
  group->push_back(g);
  return SUCCESS;
}

Shape GetSliceShape(const Shape &tensor_shape, const Dimensions &strategy) {
  Shape slice_shape;
  if (std::any_of(strategy.begin(), strategy.end(), [](int64_t value) { return value <= 0; })) {
    MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategy) << ", the element is less than or equal to 0";
    return slice_shape;
  }
  for (size_t i = 0; i < strategy.size(); ++i) {
    slice_shape.push_back(tensor_shape.at(i) / strategy.at(i));
  }
  return slice_shape;
}

Status InferSliceShapeByStrategy(const Strategies &strategys, const Shapes &shapes, Shapes *slice_shapes) {
  if (slice_shapes == nullptr) {
    MS_LOG(ERROR) << "The slice_shapes is null.";
    return FAILED;
  }
  if (strategys.size() != shapes.size()) {
    MS_LOG(ERROR) << "Strategy size " << strategys.size() << " not equal to shape size " << shapes.size();
    return FAILED;
  }

  for (size_t i = 0; i < strategys.size(); ++i) {
    if (strategys.at(i).size() != shapes.at(i).size()) {
      MS_LOG(ERROR) << "Strategy dimension " << strategys.at(i).size() << " not equal to shape dimension "
                    << shapes.at(i).size();
      slice_shapes->clear();
      return FAILED;
    }

    for (size_t j = 0; j < shapes.at(i).size(); ++j) {
      if (strategys.at(i).at(j) <= 0) {
        MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategys[i])
                      << " the element is less than or equal to 0.";
        slice_shapes->clear();
        return FAILED;
      }
      if (shapes.at(i).at(j) % strategys.at(i).at(j) != 0) {
        MS_LOG(ERROR) << "Shape cannot be divisible by strategy, " << shapes.at(i).at(j) << " : "
                      << strategys.at(i).at(j);
        slice_shapes->clear();
        return FAILED;
      }
    }
    Shape slice_shape = GetSliceShape(shapes.at(i), strategys.at(i));
    slice_shapes->push_back(slice_shape);
  }

  return SUCCESS;
}

Status OperatorInfo::InferSliceShape(const Strategies &inputs_strategy, const Strategies &outputs_strategy,
                                     Shapes *inputs_slice_shape, Shapes *outputs_slice_shape) {
  if (inputs_slice_shape == nullptr || outputs_slice_shape == nullptr) {
    MS_LOG(ERROR) << name_ << ": The slice_shape is null.";
    return FAILED;
  }

  if (InferSliceShapeByStrategy(inputs_strategy, inputs_shape_, inputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer inputs slice shape error.";
    return FAILED;
  }

  if (InferSliceShapeByStrategy(outputs_strategy, outputs_shape_, outputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer outputs slice shape error.";
    inputs_slice_shape->clear();
    return FAILED;
  }

  return SUCCESS;
}

Status OperatorInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status OperatorInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    ReportError(name_ + " : Init for cost model failed.");
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

// auto insert repeated_calculation_num for dev_matrix_shape when repeated_calculation_num > 1
Status OperatorInfo::InitForCostModelWithAutoRepeatCalc(const StrategyPtr &in_strategy,
                                                        const StrategyPtr &out_strategy) {
  if (in_strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null, the inputs shape is " << inputs_shape_;
    return FAILED;
  }

  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  // must be after InferAttrs()
  if (CheckStrategy(in_strategy) != SUCCESS) {
    FILTER_LOG(is_auto_parallel_) << name_ << ": CheckStrategy failed.";
    return FAILED;
  }
  strategy_ = in_strategy;

  if (out_strategy && CheckOutputStrategy(out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": The output strategy is invalid";
    return FAILED;
  }
  out_strategy_ = out_strategy;

  // need to clear queues before Init(),
  // because Init() may be called multiple times by cost model
  ResetQueueMember();

  if (InferDevMatrixShape() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferDevMatrixShape failed.";
    return FAILED;
  }

  used_devices_ = std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());

  // must be after InferDevMatrixShape
  if (InferRepeatedCalcInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferRepeatedCalcInfo failed.";
    return FAILED;
  }

  // if repeated calculation, need to set the repeated_calc_num as the last dimension of dev-matrix for layout
  SetRepeatedCalcDevMatrix();

  if (InferTensorMap() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorMap failed.";
    return FAILED;
  }

  ResetTensorMapIfRepeatedCalc();

  if (InferTensorInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorInfo failed.";
    return FAILED;
  }
  auto stage_dev_num = LongToSize(g_device_manager->stage_device_num());
  if ((stage_dev_num & (stage_dev_num - 1)) == 0) {
    return SUCCESS;
  }
  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferForwardCommunication failed in auto parallel searching strategies step.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferMirrorOps failed in auto parallel searching strategies step.";
    return FAILED;
  }
  return SUCCESS;
}

Status OperatorInfo::InitWithAutoRepeatCalc(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (in_strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The input strategy is null.";
    return FAILED;
  }

  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    return FAILED;
  }

  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }

  if (InferVirtualDivOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }

  InferReplaceOps();
  return SUCCESS;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAliveSuccEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) != std::string::npos)) {
      ret.push_back(edge);
    } else if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(CAST) != std::string::npos)) {
      // CAST is ordered in front of L2NORMALIZE
      ret.push_back(edge);
    }
  }
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) == std::string::npos) &&
        (edge->next_operator()->name().find(CAST) == std::string::npos)) {
      ret.push_back(edge);
    }
  }
  return ret;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAlivePrevEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator()->is_alive()) {
      ret.push_back(edge);
    }
  }
  return ret;
}

void OperatorInfo::ReplacePreEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdge: the op is null.";
    return;
  }
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplaceSuccEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdge: the op is null.";
    return;
  }
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplacePreEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdges: the op is null.";
    return;
  }
  std::vector<std::shared_ptr<Edge>> update_pre_edges;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() != op) {
      update_pre_edges.push_back(edge);
    }
  }
  update_pre_edges.push_back(new_edge);
  prev_edges_ = update_pre_edges;
}

void OperatorInfo::ReplaceSuccEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdges: the op is null";
    return;
  }
  std::vector<std::shared_ptr<Edge>> update_pre_edges;
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() != op) {
      update_pre_edges.push_back(edge);
    }
  }
  update_pre_edges.push_back(new_edge);
  succ_edges_ = update_pre_edges;
}

std::shared_ptr<Strategies> OperatorInfo::GenerateBatchStrategiesWithCheck() {
  std::shared_ptr<Strategies> batch_strategy = GenerateBatchStrategies();
  if (batch_strategy->size() != inputs_shape_.size()) {
    MS_LOG(WARNING) << "The inputs size:" << inputs_shape_.size()
                    << " is not equal to the generated batch parallel strategies size:" << batch_strategy->size();
    return batch_strategy;
  }
  int64_t shard_size = g_device_manager->stage_device_num();
  std::vector<std::pair<size_t, size_t>> changed_pos;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    auto stra = batch_strategy->at(i);
    auto input_shape = inputs_shape_.at(i);
    if (stra.size() != input_shape.size()) {
      MS_LOG(WARNING) << "The " << i << " input size:" << input_shape.size() << " is not equal to the " << i
                      << " generated batch parallel strategy size:" << stra.size();
      return batch_strategy;
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape[j] <= 0) {
        return batch_strategy;
      }
      if (stra[j] == 1) {
        continue;
      }
      if (stra[j] != g_device_manager->stage_device_num()) {
        MS_LOG(WARNING) << "The batch parallel value is not equal to device num, skip adjust it.";
        return batch_strategy;
      }
      shard_size = std::gcd(input_shape[j], shard_size);
      changed_pos.push_back({i, j});
    }
  }
  for (auto &pair : changed_pos) {
    batch_strategy->at(pair.first).at(pair.second) = shard_size;
  }
  return batch_strategy;
}

std::shared_ptr<Strategies> GenerateBatchStrategiesBySplitFlag(const Shapes &shapes,
                                                               const std::vector<bool> &split_flag_list) {
  if (shapes.size() != split_flag_list.size()) {
    MS_LOG(ERROR) << "Split_flag_list do not have the same size as inputs shape, " << split_flag_list.size() << " : "
                  << shapes.size();
    return nullptr;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  Strategies strategy_v;
  for (size_t i = 0; i != shapes.size(); i++) {
    if (shapes[i].empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      Dimensions empty_element;
      strategy_v.push_back(empty_element);
    } else {
      Dimensions element(shapes[i].size(), 1);
      if (split_flag_list[i]) {
        element[0] = dev_num;
      }
      strategy_v.push_back(element);
    }
  }
  return std::make_shared<Strategies>(strategy_v);
}

void OperatorInfo::ReComputeBatchSplitFlagList() {
  if (!inputs_shape_.empty()) {
    split_flag_list_[0] = true;
  }
}

void OperatorInfo::ComputeBatchSplitFlagList() {
  split_flag_list_.clear();
  for (auto iter = inputs_shape_.begin(); iter != inputs_shape_.end(); ++iter) {
    split_flag_list_.push_back(false);
  }
  ReComputeBatchSplitFlagList();
}

// This is a common method for checking whether the generated strategy has the correct number of devuces.
Status PrepareStrategyBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_partitions, StrategyPtr *const sp) {
  if (sp == nullptr) {
    MS_LOG(ERROR) << "The strategy is null.";
    return FAILED;
  }
  int64_t product = 1;

  for (auto &input_partition : inputs_partitions) {
    product *= std::accumulate(input_partition.begin(), input_partition.end(), 1, std::multiplies<int64_t>());
  }
  const auto fully_use_device = CostModelContext::GetInstance()->fully_use_device();
  if (!fully_use_device) {
    if (LongToSize(product) > dev_num) {
      return FAILED;
    }
  } else {
    if ((product != 1) && (LongToSize(product) != dev_num)) {
      return FAILED;
    }
  }
  Strategies stras(inputs_partitions);
  (*sp) = std::make_shared<Strategy>(stage_id, stras);
  return SUCCESS;
}

std::shared_ptr<Strategies> OperatorInfo::GenerateBatchStrategies() {
  if (inputs_shape_.empty() && InferAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer attrs failed";
  }
  ComputeBatchSplitFlagList();
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

// generate strategies for that each dimension of input0 and input1 is relevant, such as: ([a, b, c, d], [a, b, c, d])
Status GenerateStrategiesForTwoEqualInputs(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs size is wrong.";
    return FAILED;
  }

  if ((inputs_shape[0].size() != inputs_shape[1].size()) ||
      (splittable_inputs[0].size() != splittable_inputs[1].size())) {
    MS_LOG(ERROR) << "The size of two inputs are not equal.";
    return FAILED;
  }

  Shapes input0_shape = {inputs_shape[0]};
  Shapes input0_splittable = {splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, input0_shape, input0_splittable, sp_vector) != SUCCESS) {
    return FAILED;
  }

  for (auto &sp : *sp_vector) {
    sp->ExpandInputDimFromOneToTwo();
  }

  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input0 needs to broadcast
// such as: ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
Status GenerateStrategiesForBroadcastLeft(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() >= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input0 the same as input1
  Shapes tmp_inputs_shape = {inputs_shape[1], inputs_shape[1]};
  Shapes tmp_splittable_inputs = {splittable_inputs[1], splittable_inputs[1]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input0
  for (auto &sp : *sp_vector) {
    Strategies tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    size_t size_diff = inputs_shape[1].size() - inputs_shape[0].size();

    // erase the unnecessary part
    (void)input0_strategy.erase(input0_strategy.cbegin(),
                                input0_strategy.cbegin() + static_cast<different_type>(size_diff));

    // handle the case likes ([1, c, d], [a, b, c, d])
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input0_strategy);       // input0
    tmp_strategy.push_back(sp->GetInputDim()[1]);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input1 needs to broadcast
// such as: ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
Status GenerateStrategiesForBroadcastRight(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() <= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input1 the same as input0
  Shapes tmp_inputs_shape = {inputs_shape[0], inputs_shape[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input1
  for (auto &sp : *sp_vector) {
    Strategies tmp_strategy;
    tmp_strategy.push_back(sp->GetInputDim()[0]);  // input0

    Dimensions input1_strategy = sp->GetInputDim()[1];
    size_t size_diff = inputs_shape[0].size() - inputs_shape[1].size();

    // erase the unnecessary part
    (void)input1_strategy.erase(input1_strategy.cbegin(),
                                input1_strategy.cbegin() + static_cast<different_type>(size_diff));

    // handle the case likes ([a, b, c, d], [1, c, d])
    for (size_t i = 0; i < inputs_shape[1].size(); ++i) {
      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have same size, and input0 or input1 needs to broadcast
// such as: ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesForBroadcastBoth(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() != inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // step1: ([a, 1], [1, b]) -> [a, b]
  Shape max_shape, splittable_vector;
  for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
    if (inputs_shape[0][i] >= inputs_shape[1][i]) {
      max_shape.push_back(inputs_shape[0][i]);
      splittable_vector.push_back(splittable_inputs[0][i]);
    } else {
      max_shape.push_back(inputs_shape[1][i]);
      splittable_vector.push_back(splittable_inputs[1][i]);
    }
  }

  // step2: ([a, 1], [1, b]) -> generate strategy for ([a, b], [a, b])
  Shapes tmp_inputs_shape = {max_shape, max_shape};
  Shapes tmp_splittable_inputs = {splittable_vector, splittable_vector};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // step3: reset the strategy if the dimension is 1
  for (auto &sp : *sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      }

      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      }
    }
    sp->ResetInputs({input0_strategy, input1_strategy});
  }

  return SUCCESS;
}

Status GenerateStrategiesForIndependentInputsBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_shape,
                                                  const Shapes &splittable_inputs,
                                                  std::vector<StrategyPtr> *sp_vector) {
  Shape combined_inputs_shape, combined_splittable_inputs, combined_partitions;
  for (size_t j = 0; j < inputs_shape.size(); ++j) {
    (void)combined_inputs_shape.insert(combined_inputs_shape.cend(), inputs_shape[j].cbegin(), inputs_shape[j].cend());
    (void)combined_splittable_inputs.insert(combined_splittable_inputs.cend(), splittable_inputs[j].cbegin(),
                                            splittable_inputs[j].cend());
  }
  std::function<void(uint64_t, size_t)> recursive = [&stage_id, &dev_num, &sp_vector, &combined_inputs_shape,
                                                     &combined_splittable_inputs, &combined_partitions, &recursive,
                                                     &inputs_shape](uint64_t current_index, size_t n) {
    if (current_index == combined_inputs_shape.size()) {
      MS_LOG(DEBUG) << "The value of combined_splittable_inputs.size is: " << combined_splittable_inputs.size();
      Shapes inputs_partitions;
      size_t global_index = 0;
      for (auto &shape : inputs_shape) {
        Shape tmp_partition;
        for (size_t j = 0; j < shape.size(); ++j) {
          tmp_partition.push_back(combined_partitions[global_index]);
          global_index++;
        }
        inputs_partitions.push_back(tmp_partition);
      }
      StrategyPtr sp;
      if (PrepareStrategyBase(stage_id, dev_num, inputs_partitions, &sp) == SUCCESS) {
        sp_vector->push_back(sp);
      }
      return;
    } else {
      MS_LOG(DEBUG) << "The value of sp_vector size is " << sp_vector->size();
      if (combined_splittable_inputs[current_index] == 0) {
        combined_partitions.push_back(MIN_SLICE_NUM);
        recursive(current_index + 1, n / MIN_SLICE_NUM);
        combined_partitions.pop_back();
      } else if (combined_splittable_inputs[current_index] == 1) {
        for (uint64_t i = 1; i <= n; i *= 2) {
          if (n % i == 0 && LongToSize(combined_inputs_shape[current_index]) % i == 0) {
            combined_partitions.push_back(i);
            recursive(current_index + 1, n / i);
            combined_partitions.pop_back();
          }
        }
      }
    }
  };
  recursive(0, dev_num);
  if (sp_vector->empty()) {
    MS_LOG(EXCEPTION) << "No available strategy for current OperatorInfo.";
  }
  return SUCCESS;
}

// 'splittable_inputs' has the same dimensions as 'inputs_shape_'. '0' in 'splittable_inputs' means that
// the corresponding dimension is unsplittable, '1' in 'splittable_inputs' means that the corresponding
// dimension is splittable. 'inputs_partitions' is the result of partitions.
// NOTE: This implementation would partition all splittable dimensions in all inputs. Some operators requiring
// specific dimensions in inputs have the identical partition should have individual implementation.
Status GenerateStrategiesForIndependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                              const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }
  if (splittable_inputs.size() != inputs_shape.size()) {
    MS_LOG(ERROR) << "Splittable_inputs do not have the same input number of inputs shape, " << splittable_inputs.size()
                  << " : " << inputs_shape.size();
    return FAILED;
  }
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  auto dev_num_2_power = (dev_num & (dev_num - 1));
  if (dev_num_2_power == 0) {
    return GenerateStrategiesForIndependentInputsBase(stage_id, dev_num, inputs_shape, splittable_inputs, sp_vector);
  }
  auto dev_num_not_2_power = dev_num / (dev_num - dev_num_2_power);
  std::vector<StrategyPtr> sp_vector_2_power_part;
  if (GenerateStrategiesForIndependentInputsBase(stage_id, dev_num - dev_num_2_power, inputs_shape, splittable_inputs,
                                                 &sp_vector_2_power_part) != SUCCESS) {
    MS_LOG(ERROR) << "Generate strategy in the power of 2 devices part failed.";
    return FAILED;
  }
  // Handle the not power of 2 part.
  for (auto &stra : sp_vector_2_power_part) {
    auto stra_arrays = stra->GetInputDim();
    size_t stras_size = stra_arrays.size();
    for (size_t i = 0; i < stras_size; ++i) {
      auto split_input = splittable_inputs[i];
      size_t stra_size = stra_arrays[i].size();
      for (size_t j = 0; j < stra_size; ++j) {
        if (split_input[j] == 0) {
          continue;
        }
        auto new_stra_arrays{stra_arrays};
        new_stra_arrays[i][j] = new_stra_arrays[i][j] * UlongToLong(dev_num_not_2_power);
        // discard invalid strategy
        if (inputs_shape[i][j] % new_stra_arrays[i][j] != 0) {
          continue;
        }
        StrategyPtr new_stra = std::make_shared<Strategy>(stage_id, new_stra_arrays);
        sp_vector->push_back(new_stra);
      }
    }
  }
  // add the repeated strategy
  auto repeated_stra_arrays{splittable_inputs};
  for (auto &stra_array : repeated_stra_arrays) {
    std::fill(stra_array.begin(), stra_array.end(), 1);
  }
  StrategyPtr repeated_stra = std::make_shared<Strategy>(stage_id, repeated_stra_arrays);
  sp_vector->push_back(repeated_stra);
  return SUCCESS;
}

// 'splittable_inputs' has the same dimensions as 'inputs_shape_'. '0' in 'splittable_inputs' means that
// the corresponding dimension is unsplittable, otherwise means that the corresponding dimension is splittable.
// In particular, if the same dimensions exist in 'splittable_inputs',
// the corresponding dimensions in the strategy are the same.
// 'sp' is the result of partitions.
Status GenerateStrategiesForDependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                            const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp) {
  if (inputs_shape.size() != splittable_inputs.size()) {
    MS_LOG(EXCEPTION) << "Size of inputs_shape and splittable_inputs are not equal.";
  }

  std::unordered_map<int64_t, int64_t> mp;
  for (size_t i = 0; i < inputs_shape.size(); ++i) {
    auto input_shape = inputs_shape[i];
    auto splittable_input = splittable_inputs[i];
    for (size_t j = 0; j < input_shape.size(); ++j) {
      int64_t indice = splittable_input[j];
      int64_t shape = input_shape[j];
      if (splittable_input[j] == 0) {
        continue;
      }
      if (mp.find(indice) == mp.end()) {
        mp[indice] = shape;
      } else {
        mp[indice] = std::gcd(mp[indice], shape);
      }
    }
  }

  std::unordered_map<int64_t, size_t> indices_mp;
  Shape tmp_input_shape;
  Shapes tmp_splittable_inputs = {Shape(mp.size(), 1)};

  for (const auto &item : mp) {
    indices_mp[item.first] = tmp_input_shape.size();
    tmp_input_shape.push_back(item.second);
  }
  Shapes tmp_inputs_shape = {tmp_input_shape};
  std::vector<StrategyPtr> tmp_sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, &tmp_sp_vector) !=
      SUCCESS) {
    return FAILED;
  }

  (void)std::transform(tmp_sp_vector.begin(), tmp_sp_vector.end(), std::back_inserter(*sp),
                       [stage_id, &indices_mp, &splittable_inputs](const StrategyPtr &sp) {
                         auto sp_strategies = sp->GetInputDim();
                         auto sp_sub_strategy = sp_strategies.at(0);
                         Strategies strategies(splittable_inputs);
                         for (size_t i = 0; i < strategies.size(); ++i) {
                           for (size_t j = 0; j < strategies[i].size(); ++j) {
                             if (splittable_inputs[i][j] == 0) {
                               strategies[i][j] = 1;
                             } else {
                               strategies[i][j] = sp_sub_strategy[indices_mp[splittable_inputs[i][j]]];
                             }
                           }
                         }
                         return std::make_shared<Strategy>(stage_id, strategies);
                       });
  return SUCCESS;
}

// generate strategies for that have two inputs, and input0 or input1 maybe broadcast,
// and the corresponding dimensions that are not broadcast are all relevant dimensions
// such as: ([a, b, c, d], [a, b, c, d]) or ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
// or ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
// or ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesWithBroadcast(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                       std::vector<StrategyPtr> *sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs' size is wrong.";
    return FAILED;
  }

  if (inputs_shape[0] == inputs_shape[1]) {
    // element wise operation([a, b, c, d], [a, b, c, d]), so input0's strategy is equal to input1's strategy
    if (GenerateStrategiesForTwoEqualInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForTwoEqualInputs success.";
  } else if (inputs_shape[0].empty() || inputs_shape[1].empty()) {
    // ([a, b, c, d], []) or ([], [a, b, c, d])
    if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "Generate strategies for scalar case failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "Generate strategies for scalar case success.";
  } else if (inputs_shape[0].size() > inputs_shape[1].size()) {
    // ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
    if (GenerateStrategiesForBroadcastRight(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastRight failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastRight success.";
  } else if (inputs_shape[0].size() < inputs_shape[1].size()) {
    // ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
    if (GenerateStrategiesForBroadcastLeft(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastLeft failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastLeft success.";
  } else {  // same size, but different value
    // ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
    if (GenerateStrategiesForBroadcastBoth(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastBoth failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastBoth success.";
  }
  return SUCCESS;
}

Status OperatorInfo::SetCostUnderStrategyBase(const StrategyPtr &strategy) {
  if (InitForCostModel(strategy, nullptr) == FAILED) {
    MS_LOG(DEBUG) << name_ << ": Initialization under the strategy failed.";
    return FAILED;
  }
  int64_t stage_id = strategy->GetInputStage();
  double computation_cost =
    operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  double communication_cost = operator_cost()->GetCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
  std::shared_ptr<Cost> result = std::make_shared<Cost>(computation_cost, communication_cost);
  result->communication_without_parameter_ =
    operator_cost()->GetForwardCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ + gamma * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPreferringDataParallel(strategy, result);
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);
  result->communication_forward_ = result->communication_without_parameter_;

  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  (void)strategy_cost_.emplace_back(swc);

  return SUCCESS;
}

CostPtrList OperatorInfo::GetCostByStrategyPtr(const StrategyPtr &strategy) {
  auto target = std::find_if(
    strategy_cost_.begin(), strategy_cost_.end(),
    [&](const std::shared_ptr<StrategyWithCost> &stra_cost) { return stra_cost->strategy_ptr == strategy; });
  if (target == strategy_cost_.end()) {
    MS_LOG(EXCEPTION) << "There is no StrategyWithCost with a strategy";
  }
  return (*target)->cost_list;
}

TensorLayout OperatorInfo::GetInputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t input_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) { return swc->strategy_ptr->IsEqual(stra); };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    const auto &input_info = (*it)->inputs_ptr[input_index];
    return std::move(input_info.tensor_layout());
  }
  TensorLayout empty;
  return empty;
}

TensorLayout OperatorInfo::GetOutputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t output_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) { return swc->strategy_ptr->IsEqual(stra); };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    const auto &output_info = (*it)->outputs_ptr[output_index];
    return std::move(output_info.tensor_layout());
  }
  TensorLayout empty;
  return empty;
}

StrategyPtr OperatorInfo::GetStrategyFromSWCByInputLayout(const TensorLayout &input_layout, size_t input_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->inputs_ptr[input_index].tensor_layout() == input_layout;
  };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    return (*it)->strategy_ptr;
  }
  return nullptr;
}

StrategyPtr OperatorInfo::GetStrategyFromSWCByOutputLayout(const TensorLayout &output_layout, size_t output_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->outputs_ptr[output_index].tensor_layout() == output_layout;
  };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    return (*it)->strategy_ptr;
  }
  return nullptr;
}

bool OperatorInfo::IsReshape() const {
  if (name_.find(RESHAPEINFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsTmpIdentity() const {
  if (name_.find(IDENTITY_INFO) != std::string::npos) {
    return true;
  }
  return false;
}

// Keep at most (1.0 / epsilon) number of available strategies for each operator.
void OperatorInfo::ApproximateStrategies() {
  auto enable_approxi = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (!enable_approxi) {
    return;
  }
  MS_LOG(INFO) << name_ << ": Approximating strategy-cost";
  auto epsilon = CostModelContext::GetInstance()->dp_algo_approxi_epsilon();
  auto target_num = static_cast<size_t>(std::ceil(1.0 / epsilon));
  if (strategy_cost_.size() <= target_num) {
    MS_LOG(INFO) << name_ << "'s strategy number is: " << strategy_cost_.size()
                 << ", no greater than target-num: " << target_num;
    return;
  }
  std::vector<std::shared_ptr<StrategyWithCost>> ret;
  auto &origin_stra_cost = strategy_cost_;
  auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  auto beta = CostModelContext::GetInstance()->costmodel_beta();
  // sort
  std::sort(
    origin_stra_cost.begin(), origin_stra_cost.end(),
    [&alpha, &beta](const std::shared_ptr<StrategyWithCost> &s1, const std::shared_ptr<StrategyWithCost> &s2) {
      if (alpha * s1->cost_list[0]->computation_cost_ + beta * s1->cost_list[0]->communication_with_partial_para_ <
          alpha * s2->cost_list[0]->computation_cost_ + beta * s2->cost_list[0]->communication_with_partial_para_) {
        return true;
      }
      return false;
    });
  size_t step_length = origin_stra_cost.size() / target_num;
  for (size_t i = 0; ret.size() < target_num && static_cast<size_t>(i * step_length) < origin_stra_cost.size(); ++i) {
    ret.push_back(origin_stra_cost[static_cast<size_t>(i * step_length)]);
  }

  strategy_cost_ = ret;
  is_strategy_cost_exact_ = false;
}

void OperatorInfo::ExactStrategiesAndRelatedEdges() {
  if (is_strategy_cost_exact()) {
    return;
  }
  ClearStrategyCost();
  if (GenerateStrategies(0) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Strategy search failed.";
  }
  SetIsStrategyCostExactTrue();
  // re-init the previous edges
  for (auto &prev_edge : prev_edges()) {
    if (prev_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": Edge: " << prev_edge->edge_name() << " cost init failed.";
    }
  }
  // re-init the successive edges
  for (auto &next_edge : succ_edges()) {
    if (next_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": Edge: " << next_edge->edge_name() << " cost init failed.";
    }
  }
}

int64_t OperatorInfo::ComputeOpAndPrevEdgeParameterInvolved() {
  if (is_output_parameter_involve_ != -1) {
    return is_output_parameter_involve_;
  }
  is_parameter_involve_ = is_parameter_;
  const auto &prev_edges = this->GetAlivePrevEdges();
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto prev_op_para = p_edge->prev_operator()->ComputeOpAndPrevEdgeParameterInvolved();
    if (input_index >= is_parameter_involve_.size()) {
      MS_LOG(EXCEPTION) << name_ << " has input length: " << is_parameter_involve_.size()
                        << ", but got wrong input_index: " << input_index;
    }
    if (prev_op_para == 0) {
      is_parameter_involve_[input_index] = false;
    } else if (prev_op_para == 1) {
      is_parameter_involve_[input_index] = true;
    } else {
      MS_LOG(EXCEPTION) << name_ << " got wrong value: " << prev_op_para << ", input_index: " << input_index;
    }
    p_edge->set_parameter_involve(prev_op_para);
  }
  if (std::any_of(is_parameter_involve_.begin(), is_parameter_involve_.end(), [](bool value) { return value; })) {
    // If anyone of the input is a parameter_involved, the output is parameter_involved.
    is_output_parameter_involve_ = 1;
  } else {
    is_output_parameter_involve_ = 0;
  }
  // Set 'is_parameter_involve_' and 'is_output_parameter_involve_' into operatorCost, which are used in
  // calculating 'inputs_in_memory' and 'output_in_memory', respectively.
  operator_cost()->set_is_parameter_involve(is_parameter_involve_);
  operator_cost()->set_output_parameter_involve(is_output_parameter_involve_);
  // Calculating 'output_in_memory'
  operator_cost()->CalculateOutputInMemory();
  // Calculating 'inputs_in_memory'
  std::map<size_t, bool> input_in_memory;
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto is_in_mem = p_edge->prev_operator()->operator_cost()->is_output_in_memory();
    (void)input_in_memory.emplace(std::make_pair(input_index, is_in_mem));
  }
  operator_cost()->CalculateInputsInMemory(input_in_memory);

  return is_output_parameter_involve_;
}

Status OperatorInfo::set_is_parameter(const std::vector<bool> &is_parameter) {
  if (is_parameter.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": Is_parameter: " << is_parameter.size()
                  << " do not have the same number of inputs_shape_: " << inputs_shape_.size();
    return FAILED;
  }
  is_parameter_ = is_parameter;
  operator_cost()->set_is_parameter(is_parameter);
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCost() {
  if (is_parameter_involve_.size() != is_parameter_.size()) {
    MS_LOG(ERROR) << name_ << ": the size of 'is_parameter_' is " << is_parameter_.size()
                  << " does not have the same number of the size of 'is_parameter_involve_'."
                  << is_parameter_involve_.size();
    return FAILED;
  }
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCost(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCostForInference() {
  // First, set the 'is_outputs_critical_' flag into OperatorCost.
  if (is_output_critical_ == -1) {
    MS_LOG(EXCEPTION) << name_ << ": The critical flag is not set.";
  }
  operator_cost()->set_output_critical(is_output_critical_);
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCostForInference(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CorrectMemoryCost(size_t input_index) {
  for (auto &swc : strategy_cost_) {
    double parameter_mem_cost = ListProduct(swc->inputs_ptr[input_index].slice_shape()) *
                                static_cast<double>(operator_cost()->inputs_type_lengths()[input_index]);
    swc->cost_list[0]->memory_with_reuse_ -= parameter_mem_cost;
    if (swc->cost_list[0]->memory_with_reuse_ < 0) {
      MS_LOG(WARNING) << name_ << ": The memory cost after correction is: " << swc->cost_list[0]->memory_with_reuse_
                      << ", the parameter memory cost is: " << parameter_mem_cost;
      swc->cost_list[0]->memory_with_reuse_ = 0;
    }
  }
  return SUCCESS;
}

int64_t ComputeRepeatDeviceNumByTensorMap(const Shape &dev_matrix_shape, const Shape &tensor_map) {
  int64_t ret = -1;

  // The number of repetitions is equal to the number of all devices divided by the number of devices use for
  // tensor map.
  int64_t device_num = std::accumulate(dev_matrix_shape.begin(), dev_matrix_shape.end(), 1, std::multiplies<int64_t>());
  for (auto &element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (LongToSize(element) >= dev_matrix_shape.size())) {
      MS_LOG(ERROR) << "Invalid tensor map: " << ShapeToString(tensor_map) << ", the dev matrix shape is "
                    << ShapeToString(dev_matrix_shape);
      return ret;
    } else {
      size_t index = dev_matrix_shape.size() - LongToSize(element) - 1;
      if (dev_matrix_shape[index] <= 0) {
        MS_LOG(ERROR) << "Invalid dev matrix shape: " << ShapeToString(dev_matrix_shape);
        return ret;
      }
      device_num /= dev_matrix_shape[index];
    }
  }

  return device_num;
}

Status OperatorInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_.size() > 1) {
    MS_LOG(ERROR) << name_ << ": The output size is " << outputs_tensor_map_.size()
                  << ", need to override this function ";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  if (out_dev_matrix_shape_.empty()) {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

// If the operator is used as a loss, a div node is inserted for the grad of all its inputs.
Status OperatorInfo::InferVirtualDivOps() {
  if (InferAsLossDivisor() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAsLossDivisor failed.";
    return FAILED;
  }

  if (as_loss_divisor_ <= 0) {
    MS_LOG(ERROR) << name_ << ": Invalid loss divisor: " << as_loss_divisor_;
    return FAILED;
  } else if (as_loss_divisor_ == 1) {
    MS_LOG(INFO) << name_ << ": The loss divisor is 1, no need to create virtual div op.";
    return SUCCESS;
  }

  virtual_div_op_.clear();
  // if loss is repeated calculation, insert div op
  Operator op = CreateVirtualDivOp(as_loss_divisor_);
  virtual_div_op_.push_back(op);
  return SUCCESS;
}

Status OperatorInfo::SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                                 const std::vector<size_t> &output_lengths) {
  if (input_lengths.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": Input_lengths: " << input_lengths.size()
                  << " do not have the same number of inputs shape: " << inputs_shape_.size();
    return FAILED;
  }
  if (output_lengths.size() != outputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": Output_lengths: " << output_lengths.size()
                  << " do not have the same number of outputs shape: " << outputs_shape_.size();
    return FAILED;
  }
  inputs_type_lengths_ = input_lengths;
  outputs_type_lengths_ = output_lengths;
  operator_cost()->SetInputAndOutputTypeLength(input_lengths, output_lengths);
  return SUCCESS;
}

double OperatorInfo::GetOutputsTotalSize() {
  if (is_calculated_outputs_size_) {
    return outputs_total_size_;
  }
  if (outputs_type_lengths_.size() != outputs_shape_.size()) {
    MS_LOG(EXCEPTION) << name_ << ": Output_lengths: " << outputs_type_lengths_.size()
                      << " do not have the same number of outputs shape: " << outputs_shape_.size();
  }
  double sum = 0.0;
  for (size_t i = 0; i < outputs_type_lengths_.size(); ++i) {
    auto size = std::accumulate(outputs_shape_[i].begin(), outputs_shape_[i].end(), static_cast<double>(1.0),
                                std::multiplies<double>());
    sum += size * static_cast<double>(outputs_type_lengths_[i]);
  }
  is_calculated_outputs_size_ = true;
  outputs_total_size_ = sum;
  return outputs_total_size_;
}

Status OperatorInfo::set_outputs_type(const std::vector<TypePtr> &outputs_type) {
  if (outputs_type.size() != outputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": Outputs type: " << outputs_type.size()
                  << " do not have the same number of outputs shape: " << outputs_shape_.size();
    return FAILED;
  }
  outputs_type_ = outputs_type;
  return SUCCESS;
}

void OperatorInfo::BreakingTiesForPreferringDataParallel(const StrategyPtr &stra, const CostPtr &cost) const {
  if (!stra->GetInputDim().empty() && !stra->GetInputDim()[0].empty()) {
    if (stra->GetInputDim()[0][0] == stage_device_size_) {
      if (cost->computation_cost_ > 1.0) {
        cost->computation_cost_ -= 1.0;
      }
      if (cost->communication_cost_ > 1.0) {
        cost->communication_cost_ -= 1.0;
      }
      if (cost->communication_with_partial_para_ > 1.0) {
        cost->communication_with_partial_para_ -= 1.0;
      }
      if (cost->communication_without_parameter_ > 1.0) {
        cost->communication_without_parameter_ -= 1.0;
      }
    }
  }
}

void OperatorInfo::SetSelectedStrategy(const StrategyPtr &s_strategy, size_t curr_depth) {
  MS_EXCEPTION_IF_NULL(s_strategy);
  if ((selected_strategy_depth_ != -1) && (SizeToLong(curr_depth) > selected_strategy_depth_)) {
    MS_LOG(INFO) << name_ << " has already been set strategy.";
    return;
  }
  MS_LOG(INFO) << name_ << ": Set strategy " << s_strategy->ToString();
  selected_strategy_ = s_strategy;
  selected_strategy_depth_ = SizeToLong(curr_depth);
}

void OperatorInfo::set_swc_index(int64_t swc, int64_t depth) {
  MS_LOG(INFO) << name_ << ": Set SWC index: " << swc;
  selected_strategy_depth_ = depth;
  swc_index_ = swc;
}

std::vector<CNodePtr> OperatorInfo::cnodes() { return cnodes_; }

double OperatorInfo::GetForwardMemoryCostFromCNode() {
  return operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, 0);
}

void OperatorInfo::CheckSelectedStrategy(const StrategyPtr &s_strategy) {
  MS_EXCEPTION_IF_NULL(s_strategy);
  if (!s_strategy->IsEqual(selected_strategy_)) {
    MS_LOG(INFO) << name_
                 << "'s strategy may cause suboptimal, the determined strategy: " << selected_strategy_->ToString()
                 << "The minimal strategy: " << s_strategy->ToString();
  }
}

void OperatorInfo::SetStrategyCost(const std::vector<std::shared_ptr<StrategyWithCost>> &stra_cost) {
  strategy_cost_ = stra_cost;
}

Status OperatorInfo::GenerateStrategies(int64_t stage_id) {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer attrs failed";
    return FAILED;
  }

  std::vector<StrategyPtr> sp_vector = GenerateOpStrategies(stage_id);

  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated the " << GetSerialNumberString(success)
                   << " strategy: " << sp->ToString();
    } else {
      MS_LOG(INFO) << name_ << ": SetCostUnderStrategy failed, the strategy is " << sp->ToString();
    }
  }
  return SUCCESS;
}

int64_t OperatorInfo::GetIntAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << name_ << ": The value of " << attr_name << " is not int";
  }

  return attr_iter->second->cast<Int64ImmPtr>()->value();
}

bool OperatorInfo::GetBoolAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << name_ << ": The value of " << attr_name << " is not int";
  }

  return attr_iter->second->cast<BoolImmPtr>()->value();
}

std::string OperatorInfo::GetStringAttr(const std::string &attr_name) {
  std::string string_attr;
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << name_ << ": The value of " << attr_name << " is not string";
  }

  string_attr = attr_iter->second->cast<StringImmPtr>()->value();
  return string_attr;
}

std::vector<int64_t> OperatorInfo::GetTupleIntAttr(const std::string &attr_name) {
  std::vector<int64_t> tuple_attr;
  auto tuple_attr_iter = attrs_.find(attr_name);
  if (tuple_attr_iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(tuple_attr_iter->second);
  tuple_attr = GetValue<std::vector<int64_t>>(tuple_attr_iter->second);

  return tuple_attr;
}

float OperatorInfo::GetFloatAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<FP32Imm>()) {
    MS_LOG(EXCEPTION) << name_ << ": The value of " << attr_name << " is not float";
  }

  return attr_iter->second->cast<FP32ImmPtr>()->value();
}

std::vector<ValuePtr> GetValueSequence(const ValuePtr &sequence) {
  MS_EXCEPTION_IF_NULL(sequence);
  std::vector<ValuePtr> ret;
  if (!sequence->isa<ValueTuple>() && !sequence->isa<ValueList>()) {
    MS_LOG(ERROR) << "The arg is not value tuple or value list";
    return ret;
  }

  if (sequence->isa<ValueTuple>()) {
    auto val_tuple = sequence->cast<ValueTuplePtr>();
    return val_tuple->value();
  }
  auto val = sequence->cast<ValueListPtr>();
  return val->value();
}

ValuePtr MakeListValue(const std::vector<int64_t> &v) {
  std::vector<ValuePtr> list;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(list), [](int64_t ele) { return MakeValue(ele); });
  return std::make_shared<ValueSequence>(list);
}

ValuePtr MakeTupleListValue(const Shapes &v) {
  std::vector<ValuePtr> tuple;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(tuple),
                       [](const std::vector<int64_t> &list) { return MakeListValue(list); });
  return std::make_shared<ValueTuple>(tuple);
}

AnfNodePtr CreateValueTupleAnfNodePtr(const std::vector<int64_t> &value_tuple) {
  auto value_ptr = MakeValue(value_tuple)->cast<ValueTuplePtr>();
  auto value_node = NewValueNode(value_ptr);
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr CreateTensorTupleAnfNodePtr(const tensor::TensorPtrList &tensor_tuple) {
  auto tensor_ptr = MakeValue(tensor_tuple)->cast<ValueTuplePtr>();
  auto tensor_node = NewValueNode(tensor_ptr);
  return tensor_node->cast<AnfNodePtr>();
}

Operator CreateDivOpWithType(float divisor, const TypePtr &dtype) {
  OperatorName operator1_name = REAL_DIV;
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(divisor, dtype);
  ValuePtr op1_param_value = MakeValue(tensor_ptr);
  Attr op1_param = std::make_pair("divisor", op1_param_value);
  OperatorParams operator1_params = {std::make_pair(op1_param, 2)};
  OperatorAttrs operator1_attrs;
  OperatorArgs operator1_args = std::make_pair(operator1_attrs, operator1_params);
  Operator div_op = std::make_pair(operator1_name, operator1_args);
  return div_op;
}

ForwardOp CreateReduceMeanForwardOp(const std::vector<Group> &forward_group, const TypePtr &dtype) {
  // Create AllReduceSum op
  Operator op0 = CreateAllReduceOp(REDUCE_OP_SUM, forward_group[0].name());
  std::string group_name = forward_group[0].name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name;

  // Create RealDiv op
  std::vector<Device> device_list = forward_group[0].GetDevicesList();
  auto divisor = SizeToFloat(device_list.size());
  Operator op1 = CreateDivOpWithType(divisor, dtype);
  std::string dtype_name = dtype->ToString();
  MS_LOG(INFO) << "The divisor of Div op is " << device_list.size() << ", the dtype is " << dtype_name;

  return {op0, op1};
}
}  // namespace parallel
}  // namespace mindspore
