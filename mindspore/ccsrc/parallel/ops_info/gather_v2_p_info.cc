/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "parallel/ops_info/gather_v2_p_info.h"

#include <vector>
#include <numeric>
#include <functional>
#include <utility>

#include "parallel/device_matrix.h"
#include "parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status GatherV2PInfo::GetAttrs() {
  // get axis, the third input is the axis, is a ValueNode
  if (input_value_.at(2) == nullptr) {
    MS_LOG(ERROR) << name_ << ": the third input value is nullptr, is not a ValueNode!";
    return FAILED;
  }
  auto axis = GetValue<int>(input_value_.at(2));
  // if axis is negative then convert it to positive
  auto params_shape = inputs_shape_.at(0);
  if (params_shape.size() == 0) {
    MS_LOG(ERROR) << name_ << ": params can not be a scalar!";
    return FAILED;
  }
  if (axis < 0) {
    axis += SizeToInt(inputs_shape_[0].size());
  }
  axis_ = axis;

  return SUCCESS;
}

Status GatherV2PInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }

  // only support 1-dim and 2-dim param
  if (inputs_shape_.at(0).size() != 1 && inputs_shape_.at(0).size() != 2) {
    MS_LOG(ERROR) << name_ << ": Don't support param dim " << inputs_shape_.at(0).size();
    return FAILED;
  }

  // don't support scalar index
  if (inputs_shape_.at(1).size() == 0) {
    MS_LOG(ERROR) << name_ << ": Don't support scalar index.";
    return FAILED;
  }

  // axis=0, index_shape(0)%param_strategy(0) must be 0
  Shape index_shape = inputs_shape_.at(1);
  auto param_strategy = strategy->GetInputDim().at(0);
  if ((axis_ == 0) && (index_shape.at(0) % param_strategy.at(0) != 0)) {
    MS_LOG(ERROR) << name_ << ": index_shape(0) can't be divided by param_strategy(0).";
    return FAILED;
  }

  // axis != 0, param_shape(0)%(param_strategy(0)*param_strategy(axis)) must be 0
  Shape param_shape = inputs_shape_.at(0);
  if (axis_ != 0 && param_shape.at(0) % (param_strategy.at(0) * param_strategy.at(IntToSize(axis_))) != 0) {
    MS_LOG(ERROR) << name_ << ": index_shape(0) can't be divided by (param_strategy(0)*param_strategy(axis)).";
    return FAILED;
  }

  // param_strategy(axis) != 1, index can't be splited
  auto index_strategy = strategy->GetInputDim().at(1);
  auto product_i = std::accumulate(index_strategy.begin(), index_strategy.end(), 1, std::multiplies<int>());
  if ((param_strategy.at(IntToSize(axis_)) != 1) && (product_i != 1)) {
    MS_LOG(ERROR) << name_ << ": param is splited at dim (axis)" << axis_ << " ,index can't be splited.";
    return FAILED;
  }

  // param_strategy(axis) != 1, Don't support repeated calc
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  auto product_p = std::accumulate(param_strategy.begin(), param_strategy.end(), 1, std::multiplies<int>());
  if (IntToSize(product_p) != dev_num && param_strategy.at(IntToSize(axis_)) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. Don't support repeated calc.";
    return FAILED;
  }

  return SUCCESS;
}

Status GatherV2PInfo::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_a_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_a_group;
  if (CreateGroupByTensorMap(input_a_tensor_map, &input_a_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group for input a failed.";
    return FAILED;
  }

  OperatorVector op_for_input_a, op_for_input_b, op_for_axis;
  if (input_a_group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror group is empty.";
    return SUCCESS;
  } else {
    op_for_input_a = CreateMirrorOps(input_a_group[0].name(), input_a_group[0].GetDevNum());
    MS_LOG(INFO) << name_ << " : Create the mirror ops for input a success, group is " << input_a_group[0].name();
  }

  mirror_ops_.push_back(op_for_input_a);
  mirror_ops_.push_back(op_for_input_b);
  mirror_ops_.push_back(op_for_axis);

  return SUCCESS;
}

Status GatherV2PInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  out_dev_matrix_shape_.clear();
  // infer input dev_matrix_shape
  auto param_strategy = strategy_->GetInputDim().at(0);
  auto index_strategy = strategy_->GetInputDim().at(1);
  dev_matrix_shape_ = param_strategy;

  // param_strategy(axis)!=1,
  if (param_strategy.at(IntToSize(axis_)) != 1) {
    std::reverse(dev_matrix_shape_.begin(), dev_matrix_shape_.end());
  } else {
    dev_matrix_shape_.insert(dev_matrix_shape_.end(), index_strategy.begin(), index_strategy.end());
  }

  // infer out dev_matrix_shape
  // axis!=0, split axis
  if (axis_ != 0 && param_strategy.at(IntToSize(axis_)) != 1) {
    out_dev_matrix_shape_.push_back(param_strategy.at(0) * param_strategy.at(IntToSize(axis_)));
    for (size_t i = 1; i < param_strategy.size(); ++i) {
      if (i == IntToSize(axis_)) {
        out_dev_matrix_shape_.push_back(1);
      } else {
        out_dev_matrix_shape_.push_back(param_strategy.at(i));
      }
    }
  } else {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  auto product_out =
    std::accumulate(out_dev_matrix_shape_.begin(), out_dev_matrix_shape_.end(), 1, std::multiplies<int>());
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  if (product_out == 1) {
    out_dev_matrix_shape_.insert(out_dev_matrix_shape_.begin(), dev_num);
  }

  return SUCCESS;
}

Status GatherV2PInfo::InferTensorMap() {
  // infer input tensor map
  // param_strategy(axis) != 1
  size_t param_size = inputs_shape_.at(0).size();
  size_t index_size = inputs_shape_.at(1).size();
  size_t total_size = dev_matrix_shape_.size();
  std::vector<int32_t> tensor_map_index;
  std::vector<int32_t> tensor_map_params;
  auto param_strategy = strategy_->GetInputDim().at(0);
  if (param_strategy.at(IntToSize(axis_)) != 1) {
    tensor_map_index.insert(tensor_map_index.begin(), index_size, -1);
    for (size_t i = 0; i < param_size; ++i) {
      tensor_map_params.push_back(SizeToInt(i));
    }
  } else {
    // param_strategy(axis) == 1
    for (size_t i = 0; i < param_size; ++i) {
      tensor_map_params.push_back(SizeToInt(total_size - i - 1));
    }
    for (size_t i = 0; i < index_size; ++i) {
      tensor_map_index.push_back(SizeToInt(index_size - i - 1));
    }
  }

  // infer output tensor map
  std::vector<int32_t> tensor_map_out;
  if (param_strategy.at(IntToSize(axis_)) == 1) {
    // param_strategy(axis) == 1
    for (size_t i = 0; i < param_size; ++i) {
      if (i == IntToSize(axis_)) {
        for (size_t j = 0; j < index_size; ++j) {
          tensor_map_out.push_back(SizeToInt(index_size - j - 1));
        }
      } else {
        tensor_map_out.push_back(SizeToInt(total_size - i - 1));
      }
    }
  } else {
    // param_strategy(axis) != 1
    if (axis_ == 0) {
      tensor_map_out.insert(tensor_map_out.end(), 0);
      tensor_map_out.insert(tensor_map_out.end(), index_size - 1, -1);
      for (size_t i = 1; i < param_size; ++i) {
        tensor_map_out.push_back(i);
      }
    } else {
      for (size_t i = 0; i < param_size; ++i) {
        if (i == IntToSize(axis_)) {
          tensor_map_out.insert(tensor_map_out.end(), index_size, -1);
        } else {
          tensor_map_out.push_back(SizeToInt(param_size - i - 1));
        }
      }
    }
  }

  inputs_tensor_map_.emplace_back(std::move(tensor_map_params));
  inputs_tensor_map_.emplace_back(std::move(tensor_map_index));
  outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

Status GatherV2PInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape input_index_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);
  // infer tensor layout
  TensorLayout input_tensor_layout, input_index_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(0), input_shape) != SUCCESS) ||
      (input_index_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(1), input_index_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(out_dev_matrix_shape_, outputs_tensor_map_.at(0), output_shape) !=
       SUCCESS)) {
    return FAILED;
  }
  // infer tensor info
  TensorInfo input_tensor_info(input_tensor_layout);
  TensorInfo input_index_info(input_index_layout);
  TensorInfo output_tensor_info(output_tensor_layout);

  inputs_tensor_info_.push_back(input_tensor_info);
  inputs_tensor_info_.push_back(input_index_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status GatherV2PInfo::InferBias() {
  CheckGlobalDeviceManager();
  int32_t rank = g_device_manager->global_rank();
  auto input_shape = inputs_shape_.at(0);
  auto params_strategy = strategy_->GetInputDim().at(0);
  // params_size=1, axis=0
  if ((input_shape.size() == 1) && (axis_ == 0)) {
    slice_size_ = input_shape.at(0) / params_strategy.at(0);
    bias_ = rank * slice_size_;
    return SUCCESS;
  }
  // params_size=2, axis=0
  if ((input_shape.size() == 2) && (axis_ == 0)) {
    slice_size_ = input_shape.at(0) / params_strategy.at(0);
    bias_ = rank / params_strategy.at(1) * slice_size_;
    return SUCCESS;
  }
  // params_size=2, axis=1
  if ((input_shape.size() == 2) && (axis_ == 1)) {
    slice_size_ = input_shape.at(1) / params_strategy.at(1);
    bias_ = rank % params_strategy.at(1) * slice_size_;
    return SUCCESS;
  }
  MS_LOG(ERROR) << name_ << ": Don't support params_size:" << input_shape.size() << " axis:" << axis_;
  return FAILED;
}

Status GatherV2PInfo::InferGroup() {
  std::vector<Group> group_list;
  auto param_strategy = strategy_->GetInputDim().at(0);
  size_t dim = IntToSize(axis_);
  if (param_strategy.at(IntToSize(axis_)) != 1 && inputs_shape_.at(0).size() == 2) {
    dim = (axis_ + 1) % 2;
  }
  if (CreateGroupByDim(dim, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed.";
    return FAILED;
  }

  group_ = group_list.at(0);
  return SUCCESS;
}

Status GatherV2PInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph();
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Bias failed.";
    return FAILED;
  }
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto gather_v2 =
    gen_g.PushBack({gen_g.NewOpInst(GATHERV2), gen_g.virtual_input_node(), minimum, CreatInt32Imm(axis_)});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gather_v2});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype});
  auto expand_dims = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), cast, CreatInt32Imm(axis_ - 1)});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gather_v2, expand_dims});
  // don't need expandim,if param_size = 1,
  if (inputs_shape_.at(0).size() == 1) {
    mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gather_v2, cast});
  }
  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Group failed.";
    return FAILED;
  }
  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_.name()));
  OperatorAttrs attrs = {attr_op, attr_group};
  auto reduce_scatter = gen_g.PushBack({gen_g.NewOpInst(REDUCE_SCATTER, attrs), mul});
  std::vector<std::pair<AnfNodePtr, int>> input_nodes = {std::make_pair(sub, 2), std::make_pair(gather_v2, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int>>, AnfNodePtr>>(
    std::make_pair(input_nodes, reduce_scatter));

  return SUCCESS;
}

ReplaceGraphPtr GatherV2PInfo::replace_graph(const CNodePtr &cnode) {
  auto param_strategy = strategy_->GetInputDim().at(0);
  if (param_strategy.at(IntToSize(axis_)) != 1 && ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": ComputeReplaceGraph failed.";
    return nullptr;
  }
  return replace_graph_;
}

Status GatherV2PInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status GatherV2PInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }
  auto param_strategy = strategy_->GetInputDim().at(0);
  // cost model set axis and strategy
  auto gatherv2_2cost = std::dynamic_pointer_cast<GatherV2PCost>(operator_cost());
  gatherv2_2cost->set_axis(axis_);
  gatherv2_2cost->set_strategy(param_strategy);
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status GatherV2PInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status GatherV2PInfo::GenerateStrategies(int32_t stage_id) {
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies for independent inputs() failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> GatherV2PInfo::GenerateBatchStrategies() {
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  Dimensions param_strategy(inputs_shape_[0].size(), 1);
  Dimensions index_strategy;
  index_strategy.push_back(SizeToInt(dev_num));
  for (size_t i = 1; i < inputs_shape_[1].size(); i++) {
    index_strategy.push_back(1);
  }
  std::vector<Dimensions> strategy_v = {param_strategy, index_strategy};
  return std::make_shared<std::vector<std::vector<int32_t>>>(strategy_v);
}
}  // namespace parallel
}  // namespace mindspore
