/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <utility>
#include <algorithm>
#include <functional>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/crop_and_resize_info.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
Status CropAndResizeInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  auto x_strategy = strategies.at(0);
  auto boxes_strategy = strategies.at(1);
  auto index_strategy = strategies.at(2);

  if (x_strategy[1] != 1 || x_strategy[2] != 1) {
    MS_LOG(ERROR) << name_ << ": It is not support the H/W dimension of input images, "
                  << "inputs_strategy[0][1] and inputs_strategy[0][2] must be 1. "
                  << "But got strategy " << StrategyToString(strategies);
    return FAILED;
  }
  if (boxes_strategy[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The value of inputs_strategy[1][1] must be 1, "
                  << "but got strategy: " << StrategyToString(strategies);
    return FAILED;
  }
  if (boxes_strategy[0] != index_strategy[0]) {
    MS_LOG(ERROR) << name_ << ": The value of inputs_strategy[1][0] and inputs[2][0] must be equal, "
                  << "but got strategy " << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status CropAndResizeInfo::CheckStrategyForDynamicShape(const StrategyPtr &strategy) {
  Strategies strategies = strategy->GetInputDim();
  auto x_strategy = strategies[0];
  if (x_strategy[0] != 1 && inputs_shape_[0][0] == -1) {
    MS_LOG(ERROR) << name_ << ": the dim 0 of first input can not be split if it's dynamic shape, the strategy is "
                  << ShapesToString(strategies) << ", the inputs' shape: " << ShapesToString(inputs_shape_);
    return FAILED;
  }
  return SUCCESS;
}

Status CropAndResizeInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  auto strategies = strategy_->GetInputDim();
  auto x_strategy = strategies.at(0);
  auto boxes_strategy = strategies.at(1);
  dev_matrix_shape_ = {x_strategy[0], x_strategy[3], boxes_strategy[0]};
  return SUCCESS;
}

Status CropAndResizeInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape x_map = {2, -1, -1, 1};
  Shape boxes_map = {0, -1};
  Shape index_map = {0};
  Shape output_map = {0, -1, -1, 1};
  (void)inputs_tensor_map_.emplace_back(std::move(x_map));
  (void)inputs_tensor_map_.emplace_back(std::move(boxes_map));
  (void)inputs_tensor_map_.emplace_back(std::move(index_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_map));
  return SUCCESS;
}

Status CropAndResizeInfo::InferBias() {
  MS_EXCEPTION_IF_NULL(strategy_);
  CheckGlobalDeviceManager();

  int64_t rank = g_device_manager->rank_index_in_stage();
  auto strategies = strategy_->GetInputDim();
  auto x_strategy = strategies.at(0);
  Shape x_shape = inputs_shape_.at(0);
  MS_EXCEPTION_IF_ZERO("x_strategy[0]", x_strategy[0]);
  if (x_shape[0] % x_strategy[0] != 0) {
    return FAILED;
  }
  int64_t dev_accu =
    std::accumulate(dev_matrix_shape_.begin() + 1, dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
  MS_EXCEPTION_IF_ZERO("dev_accu", dev_accu);
  slice_size_ = x_shape[0] / x_strategy[0];
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    bias_ = rank / dev_matrix_shape_[2] / dev_matrix_shape_[3] % dev_matrix_shape_[1] * slice_size_;
  } else {
    bias_ = rank / dev_accu * slice_size_;
  }

  MS_LOG(INFO) << "Sharding batch, the rank is " << rank << ", slice size is " << slice_size_ << ", bias is " << bias_;
  return SUCCESS;
}

Status CropAndResizeInfo::InferGroup() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  size_t dim = 0;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    ++dim;
  }

  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(dim), &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed.";
    return FAILED;
  }
  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The group is empty.";
    return SUCCESS;
  }

  MS_LOG(INFO) << name_ << ": The group rank is " << group_devices;
  if (g_device_manager->CreateGroup(group_devices, &group_) != SUCCESS) {
    MS_LOG(ERROR) << "The node " << cnode_->fullname_with_scope() << " create sync allreduce failed";
    return FAILED;
  }
  return SUCCESS;
}

Status CropAndResizeInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer bias failed.";
    return FAILED;
  }
  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Group failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), minimum, sub});
  auto cast_equal =
    gen_g.PushBack({gen_g.NewOpInst(CAST), equal, CreatInt64Imm(static_cast<int64_t>(kNumberTypeFloat32))});
  auto crop_and_resize = gen_g.PushBack({gen_g.NewOpInst(CROP_AND_RESIZE), gen_g.virtual_input_node(),
                                         gen_g.virtual_input_node(), minimum, gen_g.virtual_input_node()});
  auto expand_dims_0 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), cast_equal, CreatInt64Imm(-1)});
  auto expand_dims_1 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), expand_dims_0, CreatInt64Imm(-1)});
  auto expand_dims_2 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), expand_dims_1, CreatInt64Imm(-1)});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), crop_and_resize, expand_dims_2});

  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_.name()));
  OperatorAttrs attrs = {attr_op, attr_group};
  AnfNodePtr reduce_op = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(sub, 3), std::make_pair(crop_and_resize, 1),
                                                             std::make_pair(crop_and_resize, 2),
                                                             std::make_pair(crop_and_resize, 4)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, reduce_op));
  return SUCCESS;
}

Status CropAndResizeInfo::InitForCostModel(const StrategyPtr &strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy, out_strategy) != SUCCESS) {
    MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  constexpr size_t CROP_SIZE_INDEX = 3;
  auto crop_size = GetValue<std::vector<int64_t>>(input_value_[CROP_SIZE_INDEX]);
  auto strategies = strategy_->GetInputDim();
  auto x_strategy = strategies.at(0);
  auto crop_and_resize_cost = std::dynamic_pointer_cast<CropAndResizeCost>(operator_cost());
  crop_and_resize_cost->set_crop_size(crop_size);
  crop_and_resize_cost->set_strategy(x_strategy);
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

ReplaceGraphPtr CropAndResizeInfo::replace_graph(const CNodePtr &cnode) {
  auto strategies = strategy_->GetInputDim();
  auto x_strategy = strategies.at(0);
  if (x_strategy[0] != 1 && ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

std::vector<StrategyPtr> CropAndResizeInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape x_split = {1, 0, 0, 1};
  Shape boxes_split = {1, 0};
  Shapes splittable_inputs = {x_split, boxes_split};
  Shapes sub_inputs_shape(inputs_shape_.begin(), inputs_shape_.begin() + 2);

  std::vector<StrategyPtr> sp_vector;
  std::vector<StrategyPtr> sub_sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, sub_inputs_shape, splittable_inputs, &sub_sp_vector) !=
      SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  (void)std::transform(sub_sp_vector.begin(), sub_sp_vector.end(), std::back_inserter(sp_vector),
                       [stage_id](const StrategyPtr &sp) {
                         auto strategies = sp->GetInputDim();
                         int64_t boxes_shard_num = strategies[1][0];
                         std::vector<int64_t> index_strategy = {boxes_shard_num};
                         (void)strategies.emplace_back(std::move(index_strategy));
                         return std::make_shared<Strategy>(stage_id, strategies);
                       });
  return sp_vector;
}

Status CropAndResizeInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_crop_size;
  (void)mirror_ops_.emplace_back(std::move(op_for_crop_size));
  return SUCCESS;
}

REGISTER(CropAndResizeInfo);
}  // namespace parallel
}  // namespace mindspore
