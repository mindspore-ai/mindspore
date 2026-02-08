/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/ops_info/quant_info.h"

#include <utility>
#include <algorithm>

#include "include/utils/parallel_context.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
namespace {
const int ANTIQUANT_INPUT_X = 0;
const int ANTIQUANT_INPUT_SCALE = 1;
const int ANTIQUANT_INPUT_OFFSET = 2;
const int ANTIQUANT_INPUT_SIZE = 3;
}  // namespace
Status FakeQuantPerLayerInfo::GetAttrs() {
  if (inputs_shape_[1].size() != 1 || inputs_shape_[2].size() != 1) {
    MS_LOG(ERROR) << name_ << ": only support that both shape of min and max are 1, but the shape of min is "
                  << inputs_shape_[1] << ", and the shape of max is " << inputs_shape_[2];
    return FAILED;
  }

  if (inputs_shape_[1][0] != 1 || inputs_shape_[2][0] != 1) {
    MS_LOG(ERROR) << name_ << ": only support that both shape of min and max are 1, but the shape of min is "
                  << inputs_shape_[1] << ", and the shape of max is " << inputs_shape_[2];
    return FAILED;
  }
  return SUCCESS;
}

Status FakeQuantPerLayerInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FakeQuantPerLayerInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    return SUCCESS;
  }
  dev_matrix_shape_ = strategies[0];

  MS_LOG(INFO) << name_ << ": dev matrix is: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status FakeQuantPerLayerInfo::InferTensorMap() {
  Shape sub_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    sub_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(sub_tensor_map);
  inputs_tensor_map_.push_back({-1});
  inputs_tensor_map_.push_back({-1});
  (void)outputs_tensor_map_.emplace_back(std::move(sub_tensor_map));
  return SUCCESS;
}

std::vector<StrategyPtr> FakeQuantPerLayerInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The strategy is null or empty";
    }
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Strategies tmp_strategy = {first_input_strategy, {1}, {1}};
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status FakeQuantPerChannelInfo::GetAttrs() {
  channel_axis_ = GetIntAttr(CHANNEL_AXIS);
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": the size of inputs shape must be 3, but got " << inputs_shape_.size();
    return FAILED;
  }
  int64_t dim = SizeToLong(inputs_shape_[0].size());
  if ((channel_axis_ > dim) || (channel_axis_ < -dim)) {
    MS_LOG(ERROR) << name_ << ": the dim is " << dim << ", but got the invalid channel axis: " << channel_axis_;
    return FAILED;
  }

  if (channel_axis_ < 0) {
    channel_axis_ += dim;
  }
  MS_LOG(INFO) << name_ << ": the channel axis is " << channel_axis_;

  if (inputs_shape_[1].size() != 1 || inputs_shape_[2].size() != 1) {
    MS_LOG(ERROR) << name_ << ": only support that both dimension of min and max are 1, but the shape of min is "
                  << inputs_shape_[1] << ", and the shape of max is " << inputs_shape_[2];
    return FAILED;
  }
  return SUCCESS;
}

Status FakeQuantPerChannelInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // The strategy for each input tensor must be equal
  Strategies strategies = strategy->GetInputDim();
  if (strategies.size() != 3) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 3, but the strategy is " << strategies;
    return FAILED;
  }

  if (strategies[1] != strategies[2]) {
    MS_LOG(ERROR) << name_
                  << ": The the strategy of min must be equal to the strategy of max, but the strategy of min is  "
                  << strategies[1] << ", the strategy of max is " << strategies[2];
    return FAILED;
  }

  if (strategies[0][channel_axis_] != strategies[1][0]) {
    MS_LOG(ERROR) << name_ << ": The the strategy is " << strategies;
    return FAILED;
  }
  return SUCCESS;
}

Status FakeQuantPerChannelInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    return SUCCESS;
  }
  dev_matrix_shape_ = strategies[0];

  MS_LOG(INFO) << name_ << ": dev matrix is: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status FakeQuantPerChannelInfo::InferTensorMap() {
  Shape sub_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    sub_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(sub_tensor_map);
  inputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  inputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  (void)outputs_tensor_map_.emplace_back(std::move(sub_tensor_map));
  return SUCCESS;
}

std::vector<StrategyPtr> FakeQuantPerChannelInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(first_input_strategy);
    tmp_strategy.push_back({first_input_strategy[channel_axis_]});
    tmp_strategy.push_back({first_input_strategy[channel_axis_]});
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

void FakeQuantPerChannelInfo::ReComputeBatchSplitFlagList() {
  if (!inputs_shape_[0].empty()) {
    split_flag_list_[0] = True;
  }

  if (channel_axis_ == 0) {
    // Batch dim of each input can be split
    for (size_t i = 1; i < split_flag_list_.size(); ++i) {
      split_flag_list_[i] = True;
    }
  }
  return;
}

Status MinMaxUpdatePerLayerInfo::GetAttrs() {
  ema_ = GetBoolAttr(EMA);
  ema_decay_ = GetFloatAttr(EMA_DECAY);
  op_name_ = MIN_MAX_UPDATE_PER_LAYER;
  Attr ema_attr = std::make_pair(EMA, MakeValue(ema_));
  Attr ema_decay_attr = std::make_pair(EMA_DECAY, MakeValue(ema_decay_));
  op_attrs_ = {ema_attr, ema_decay_attr};
  return SUCCESS;
}

Status MinMaxUpdatePerLayerInfo::InferTensorMap() {
  Shape sub_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    sub_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(sub_tensor_map);
  inputs_tensor_map_.push_back({MAP_NONE});
  inputs_tensor_map_.push_back({MAP_NONE});
  outputs_tensor_map_.push_back({MAP_NONE});
  outputs_tensor_map_.push_back({MAP_NONE});
  return SUCCESS;
}

Status MinMaxUpdatePerLayerInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }
  as_loss_divisor_ = stage_device_size_;
  return SUCCESS;
}

Status MinMaxUpdatePerLayerInfo::InferForwardGroup() {
  auto strategies = strategy_->GetInputDim();
  Dimensions stra = strategies.at(0);

  size_t size = stra.size();
  Shape group_map(size, MAP_NONE);  // init group map

  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
      // it need to handle the group_map and insert the 0 to the last dimension of the group_map.
      group_map.push_back(0);
    } else {
      // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
      // it need to handle the first dimension of group_map.
      (void)group_map.insert(group_map.cbegin(), dev_matrix_shape_.size() - 1);
    }
  }
  MS_LOG(INFO) << name_ << ": the dev matrix is " << dev_matrix_shape_ << ", forward group map is " << group_map;
  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_map, &forward_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (!forward_group.empty()) {
    forward_group_ = forward_group[0].name();
    MS_LOG(INFO) << name_ << ": The forward group is " << forward_group_;
  }

  return SUCCESS;
}

ReplaceGraphPtr MinMaxUpdatePerLayerInfo::replace_graph(const CNodePtr &cnode) {
  if (InferForwardGroup() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": Create group failed";
  }

  if (forward_group_.empty()) {
    MS_LOG(INFO) << name_ << ": The forward group is empty, no need to replace graph";
    return nullptr;
  }

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << "GenerateGraph Init failed";
  }

  auto quant_op = gen_g.PushBack({gen_g.NewOpInst(op_name_, op_attrs_), gen_g.virtual_input_node(),
                                  gen_g.virtual_input_node(), gen_g.virtual_input_node()});
  auto get_item0 = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), quant_op, CreatInt64Imm(0)});
  auto get_item1 = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), quant_op, CreatInt64Imm(1)});

  Attr min_attr = std::make_pair(OP, MakeValue(REDUCE_OP_MIN));
  Attr max_attr = std::make_pair(OP, MakeValue(REDUCE_OP_MAX));
  Attr group_attr = std::make_pair(GROUP, MakeValue(forward_group_));
  OperatorAttrs allreduce_min_attrs = {min_attr, group_attr};
  OperatorAttrs allreduce_max_attrs = {max_attr, group_attr};
  auto allreduce_min = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, allreduce_min_attrs), get_item0});
  auto allreduce_max = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, allreduce_max_attrs), get_item1});

  auto make_list = gen_g.PushBack({gen_g.NewOpInst(MAKE_TUPLE_OP), allreduce_min, allreduce_max});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(quant_op, 1), std::make_pair(quant_op, 2),
                                                             std::make_pair(quant_op, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, make_list));
  return replace_graph_;
}

Status MinMaxUpdatePerChannelInfo::GetAttrs() {
  (void)MinMaxUpdatePerLayerInfo::GetAttrs();
  channel_axis_ = GetIntAttr(CHANNEL_AXIS);
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": the size of inputs shape must be 3, but got " << inputs_shape_.size();
    return FAILED;
  }
  int64_t dim = SizeToLong(inputs_shape_[0].size());
  if ((channel_axis_ > dim) || (channel_axis_ < -dim)) {
    MS_LOG(ERROR) << name_ << ": the dim is " << dim << ", but got the invalid channel axis: " << channel_axis_;
    return FAILED;
  }

  if (channel_axis_ < 0) {
    channel_axis_ += dim;
  }
  MS_LOG(INFO) << name_ << ": the channel axis is " << channel_axis_;

  if (inputs_shape_[1].size() != 1 || inputs_shape_[2].size() != 1) {
    MS_LOG(ERROR) << name_ << ": only support that both dimension of min and max are 1, but the shape of min is "
                  << inputs_shape_[1] << ", and the shape of max is " << inputs_shape_[2];
    return FAILED;
  }

  op_name_ = MIN_MAX_UPDATE_PER_CHANNEL;
  Attr ema_attr = std::make_pair(EMA, MakeValue(ema_));
  Attr ema_decay_attr = std::make_pair(EMA_DECAY, MakeValue(ema_decay_));
  Attr channel_attr = std::make_pair(CHANNEL_AXIS, MakeValue(channel_axis_));
  op_attrs_ = {ema_attr, ema_decay_attr, channel_attr};
  return SUCCESS;
}

Status MinMaxUpdatePerChannelInfo::InferTensorMap() {
  Shape sub_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    sub_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(sub_tensor_map);
  inputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  inputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  outputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  outputs_tensor_map_.push_back({sub_tensor_map[channel_axis_]});
  return SUCCESS;
}

Status MinMaxUpdatePerChannelInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }
  MS_EXCEPTION_IF_ZERO("dev_matrix_shape_[channel_axis_]", dev_matrix_shape_[channel_axis_]);
  as_loss_divisor_ = stage_device_size_ / dev_matrix_shape_[channel_axis_];
  return SUCCESS;
}

Status MinMaxUpdatePerChannelInfo::InferForwardGroup() {
  Shape group_map = {inputs_tensor_map_[0][channel_axis_]};  // init map

  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
      // it need to handle the group_map.
      group_map[0] += 1;
    }
  }
  MS_LOG(INFO) << name_ << ": the dev matrix is " << dev_matrix_shape_ << ", forward group map is " << group_map;
  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_map, &forward_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (!forward_group.empty()) {
    forward_group_ = forward_group[0].name();
    MS_LOG(INFO) << name_ << ": The forward group is " << forward_group_;
  }

  return SUCCESS;
}

std::vector<StrategyPtr> MinMaxUpdatePerChannelInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(first_input_strategy);
    tmp_strategy.push_back({first_input_strategy[channel_axis_]});
    tmp_strategy.push_back({first_input_strategy[channel_axis_]});
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

void MinMaxUpdatePerChannelInfo::ReComputeBatchSplitFlagList() {
  if (!inputs_shape_[0].empty()) {
    split_flag_list_[0] = True;
  }

  if (channel_axis_ == 0) {
    // Batch dim of each input can be split
    for (size_t i = 1; i < split_flag_list_.size(); ++i) {
      split_flag_list_[i] = True;
    }
  }
  return;
}

Shape AntiQuantExpandShape(const Shape &bigger_size_shape, Shape smaller_size_shape) {
  size_t insert_num = bigger_size_shape.size() - smaller_size_shape.size();
  for (size_t num = 0; num < insert_num; ++num) {
    (void)smaller_size_shape.insert(smaller_size_shape.cbegin(), 1);
  }
  return smaller_size_shape;
}

Strategies AntiQuantExpandStrategy(const StrategyPtr &strategy) {
  Strategies expand_strategy;
  Strategies stra = strategy->GetInputDim();
  Dimensions in0_strategy = stra.at(ANTIQUANT_INPUT_X);
  Dimensions in1_strategy = stra.at(ANTIQUANT_INPUT_SCALE);
  Dimensions in2_strategy = stra.at(ANTIQUANT_INPUT_OFFSET);
  size_t in0_size = in0_strategy.size();
  size_t in1_size = in1_strategy.size();
  size_t in2_size = in2_strategy.size();

  // in0_size >= in1_size, in0_size >= in2_size
  if (in0_size < in1_size || in0_size < in2_size) {
    return expand_strategy;
  }
  expand_strategy.push_back(in0_strategy);
  if (in0_size > in1_size) {
    expand_strategy.push_back(AntiQuantExpandShape(in0_strategy, in1_strategy));
  } else {
    expand_strategy.push_back(in1_strategy);
  }

  if (in0_size > in2_size) {
    expand_strategy.push_back(AntiQuantExpandShape(in0_strategy, in2_strategy));
  } else {
    expand_strategy.push_back(in2_strategy);
  }
  return expand_strategy;
}

TensorMap SetExpandQuantTensorMap(const Shape &strategy, const Shape &dev_matrix_shape) {
  TensorMap tensor_map_index;
  for (size_t i = 0; i < strategy.size(); ++i) {
    if (strategy[i] == dev_matrix_shape[i]) {
      tensor_map_index.push_back(static_cast<int64_t>(LAST_INDEX(strategy.size()) - i));
    } else {
      tensor_map_index.push_back(-1);
    }
  }
  return tensor_map_index;
}

TensorMap SetQuantTensorMap(const Shape &strategy_expand, const Shape &dev_matrix_shape, const Shape &strategy) {
  TensorMap expand_map = SetExpandQuantTensorMap(strategy_expand, dev_matrix_shape);
  size_t dev_matrix_size = dev_matrix_shape.size();
  size_t strategy_size = strategy.size();
  if (dev_matrix_size != strategy_size) {
    (void)expand_map.erase(expand_map.cbegin(),
                           expand_map.cbegin() + static_cast<different_type>(dev_matrix_size - strategy_size));
  }
  return expand_map;
}

Status AntiQuantInfo::InferTensorMap() {
  if (inputs_shape_.size() == ANTIQUANT_INPUT_SIZE - 1) {
    return ArithmeticBase::InferTensorMap();
  }

  Shape tensor_map_index;
  Strategies expand_strategy = AntiQuantExpandStrategy(strategy_);
  if (expand_strategy.size() == 0) {
    return FAILED;
  }
  Dimensions in0_expand_strategy = expand_strategy.at(ANTIQUANT_INPUT_X);
  Dimensions in1_expand_strategy = expand_strategy.at(ANTIQUANT_INPUT_SCALE);
  Dimensions in2_expand_strategy = expand_strategy.at(ANTIQUANT_INPUT_OFFSET);
  Strategies stra = strategy_->GetInputDim();
  Dimensions in0_strategy = stra.at(ANTIQUANT_INPUT_X);
  Dimensions in1_strategy = stra.at(ANTIQUANT_INPUT_SCALE);
  Dimensions in2_strategy = stra.at(ANTIQUANT_INPUT_OFFSET);
  for (size_t i = 0; i < in0_expand_strategy.size(); ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(LAST_INDEX(in0_expand_strategy.size()) - i));
  }

  // Get dev matrix without repeated calculation
  Shape dev_shape = dev_matrix_shape_;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      dev_shape.pop_back();
    } else {
      (void)dev_shape.erase(dev_shape.cbegin());
    }
  }

  (void)inputs_tensor_map_.emplace_back(SetQuantTensorMap(in0_expand_strategy, dev_shape, in0_strategy));
  (void)inputs_tensor_map_.emplace_back(SetQuantTensorMap(in1_expand_strategy, dev_shape, in1_strategy));
  (void)inputs_tensor_map_.emplace_back(SetQuantTensorMap(in2_expand_strategy, dev_shape, in2_strategy));
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_index));
  return SUCCESS;
}

REGISTER(AntiQuantInfo);
REGISTER(FakeQuantPerLayerInfo);
REGISTER(FakeQuantPerChannelInfo);
REGISTER(MinMaxUpdatePerLayerInfo);
REGISTER(MinMaxUpdatePerChannelInfo);
REGISTER(QuantInfo);
REGISTER(QuantV2Info);
REGISTER(DequantInfo);
}  // namespace parallel
}  // namespace mindspore
