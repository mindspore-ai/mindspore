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

#include "frontend/parallel/ops_info/arithmetic_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Shape ExpandShape(const Shape &bigger_size_shape, Shape smaller_size_shape) {
  size_t insert_num = bigger_size_shape.size() - smaller_size_shape.size();
  for (size_t num = 0; num < insert_num; ++num) {
    (void)smaller_size_shape.insert(smaller_size_shape.cbegin(), 1);
  }
  return smaller_size_shape;
}

Shapes ArithmeticBase::InferExpandShape() {
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  Shapes input_shapes;
  size_t input_a_size = input_a_shape.size();
  size_t input_b_size = input_b_shape.size();
  if (input_a_size > input_b_size) {
    input_shapes.push_back(input_a_shape);
    input_shapes.push_back(ExpandShape(input_a_shape, input_b_shape));
  } else if (input_a_size < input_b_size) {
    input_shapes.push_back(ExpandShape(input_b_shape, input_a_shape));
    input_shapes.push_back(input_b_shape);
  } else {
    input_shapes.push_back(input_a_shape);
    input_shapes.push_back(input_b_shape);
  }
  return input_shapes;
}

Strategies ExpandStrategy(const StrategyPtr &strategy) {
  Strategies expand_strategy;
  Strategies stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  size_t input_a_size = sub_a_strategy.size();
  size_t input_b_size = sub_b_strategy.size();
  if (input_a_size > input_b_size) {
    expand_strategy.push_back(sub_a_strategy);
    expand_strategy.push_back(ExpandShape(sub_a_strategy, sub_b_strategy));
  } else if (input_a_size < input_b_size) {
    expand_strategy.push_back(ExpandShape(sub_b_strategy, sub_a_strategy));
    expand_strategy.push_back(sub_b_strategy);
  } else {
    expand_strategy = stra;
  }
  return expand_strategy;
}

Status ArithmeticBase::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  return BaseCheckStrategy(strategy);
}

Status ArithmeticBase::BaseCheckStrategy(const StrategyPtr &strategy) {
  Shapes input_shapes = InferExpandShape();
  Strategies expand_strategy = ExpandStrategy(strategy);
  Dimensions sub_a_strategy = expand_strategy.at(0);
  Dimensions sub_b_strategy = expand_strategy.at(1);
  Shape input_a_shape = input_shapes.at(0);
  Shape input_b_shape = input_shapes.at(1);

  for (size_t i = 0; i < input_a_shape.size(); ++i) {
    if ((sub_a_strategy[i] != sub_b_strategy[i]) && (input_a_shape[i] != 1) && (input_b_shape[i] != 1)) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ArithmeticBase::InferDevMatrixShape() {
  Strategies expand_strategy = ExpandStrategy(strategy_);
  Dimensions sub_a_strategy = expand_strategy.at(0);
  Dimensions sub_b_strategy = expand_strategy.at(1);
  Shape dev_shape;
  for (size_t i = 0; i < sub_a_strategy.size(); ++i) {
    if (sub_a_strategy[i] != sub_b_strategy[i]) {
      dev_shape.push_back(sub_a_strategy[i] * sub_b_strategy[i]);
    } else {
      dev_shape.push_back(sub_a_strategy[i]);
    }
  }
  dev_matrix_shape_ = dev_shape;

  return SUCCESS;
}

TensorMap SetExpandTensorMap(const Shape &strategy, const Shape &dev_matrix_shape) {
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

TensorMap SetTensorMap(const Shape &strategy_expand, const Shape &dev_matrix_shape, const Shape &strategy) {
  TensorMap expand_map = SetExpandTensorMap(strategy_expand, dev_matrix_shape);
  size_t dev_matrix_size = dev_matrix_shape.size();
  size_t strategy_size = strategy.size();
  if (dev_matrix_size != strategy_size) {
    (void)expand_map.erase(expand_map.cbegin(),
                           expand_map.cbegin() + static_cast<different_type>(dev_matrix_size - strategy_size));
  }
  return expand_map;
}

void ArithmeticBase::ReComputeBatchSplitFlagList() {
  Shapes expand_shapes = InferExpandShape();
  Shape expand_a_shape = expand_shapes.at(0);
  Shape expand_b_shape = expand_shapes.at(1);
  if (expand_a_shape.size() != expand_b_shape.size()) {
    MS_LOG(EXCEPTION) << name_ << " : Recompute batch split flag list is wrong.";
  }
  if (expand_a_shape.empty()) {
    split_flag_list_[0] = false;
    split_flag_list_[1] = false;
    return;
  }
  (expand_a_shape.at(0) != 1) ? (split_flag_list_[0] = true) : (split_flag_list_[0] = false);
  (expand_b_shape.at(0) != 1) ? (split_flag_list_[1] = true) : (split_flag_list_[1] = false);
}

Status ArithmeticBase::CheckLayoutConfig() {
  // if the shard_num is 1, the tensor map has reset to -1
  if (inputs_shape_[0] != inputs_shape_[1] && inputs_tensor_map_[0] == inputs_tensor_map_[1]) {
    MS_LOG(ERROR) << name_
                  << ": the input_tensor_map[0] must be equal to input_tensor_map[1], but the inputs_tensor_map is "
                  << inputs_tensor_map_ << ", and the inputs shape is " << inputs_shape_;
    return FAILED;
  }

  // broadcast: such as [a, b, c, d] and [a, -1, c, d],  [a, b, c, d] and [-1, d]
  size_t len_diff = 0;
  if (inputs_shape_[0].size() >= inputs_shape_[1].size()) {
    len_diff = inputs_shape_[0].size() - inputs_shape_[1].size();
    for (size_t i = 0; i < inputs_tensor_map_[1].size(); ++i) {
      if (inputs_shape_[0][i + len_diff] == inputs_shape_[1][i] &&
          inputs_tensor_map_[0][i + len_diff] != inputs_tensor_map_[1][i]) {
        MS_LOG(ERROR) << name_ << ": invalid tensor map, the inputs_tensor_map is " << inputs_tensor_map_
                      << ", and the inputs shape is " << inputs_shape_;
        return FAILED;
      }
    }
  } else {
    len_diff = inputs_shape_[1].size() - inputs_shape_[0].size();
    for (size_t i = 0; i < inputs_tensor_map_[0].size(); ++i) {
      if (inputs_shape_[0][i] == inputs_shape_[1][i + len_diff] &&
          inputs_tensor_map_[0][i] != inputs_tensor_map_[1][i + len_diff]) {
        MS_LOG(ERROR) << name_ << ": invalid tensor map, the inputs_tensor_map is " << inputs_tensor_map_
                      << ", and the inputs shape is " << inputs_shape_;
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status ArithmeticBase::InferOutputTensorMap() {
  if (inputs_tensor_map_[0] == inputs_tensor_map_[1]) {
    outputs_tensor_map_.push_back(inputs_tensor_map_[0]);
    return SUCCESS;
  }

  // if the shard_num is 1, the tensor map has reset to -1
  // broadcast: such as input tensor map : [a, b, c, d] and [-1, d], and output tensor map is [a, b, c, d]
  size_t len_diff = 0;
  Shape output_tensor_map;
  if (inputs_shape_[0].size() >= inputs_shape_[1].size()) {
    output_tensor_map = inputs_tensor_map_[0];
    len_diff = inputs_shape_[0].size() - inputs_shape_[1].size();
    for (size_t i = 0; i < inputs_tensor_map_[1].size(); ++i) {
      output_tensor_map[i + len_diff] = inputs_tensor_map_[0][i + len_diff] == MAP_NONE
                                          ? inputs_tensor_map_[1][i]
                                          : inputs_tensor_map_[0][i + len_diff];
    }
  } else {
    output_tensor_map = inputs_tensor_map_[1];
    len_diff = inputs_shape_[1].size() - inputs_shape_[0].size();
    for (size_t i = 0; i < inputs_tensor_map_[0].size(); ++i) {
      output_tensor_map[i + len_diff] = inputs_tensor_map_[1][i + len_diff] == MAP_NONE
                                          ? inputs_tensor_map_[0][i]
                                          : inputs_tensor_map_[1][i + len_diff];
    }
  }

  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(INFO) << name_ << ": the input tensor map is " << inputs_tensor_map_ << ", the output tensor map is "
               << outputs_tensor_map_;
  return SUCCESS;
}

Status ArithmeticBase::InferTensorMap() {
  Shape tensor_map_index;
  Strategies expand_strategy = ExpandStrategy(strategy_);
  Dimensions sub_a_expand_strategy = expand_strategy.at(0);
  Dimensions sub_b_expand_strategy = expand_strategy.at(1);
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  for (size_t i = 0; i < sub_a_expand_strategy.size(); ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_expand_strategy.size()) - i));
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

  (void)inputs_tensor_map_.emplace_back(SetTensorMap(sub_a_expand_strategy, dev_shape, sub_a_strategy));
  (void)inputs_tensor_map_.emplace_back(SetTensorMap(sub_b_expand_strategy, dev_shape, sub_b_strategy));
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_index));

  return SUCCESS;
}

Status ArithmeticBase::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> ArithmeticBase::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};
  if (inputs_shape_.size() < 2) {
    MS_LOG(EXCEPTION) << name_ << ": Size of inputs must be greater than or equal to 2, but got size "
                      << inputs_shape_.size();
  }
  Shapes inputs_shape(inputs_shape_.cbegin(), inputs_shape_.cbegin() + 2);

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies with broadcast failed.";
  }
  MS_LOG(INFO) << name_ << " : Generate strategies with broadcast success.";

  return sp_vector;
}

Status LerpInfo::GetAttrs() {
  inputs_size_ = inputs_shape_.size();
  if (inputs_size_ != 2 && inputs_size_ != 3) {
    MS_LOG(ERROR) << name_ << ": Inputs size must be 2 or 3, but got size " << inputs_size_;
    return FAILED;
  }

  return SUCCESS;
}

Status LerpInfo::CheckStrategy(const StrategyPtr &strategy) {
  size_t input_nums = 2;
  if (inputs_size_ == input_nums) {
    return ArithmeticBase::CheckStrategy(strategy);
  }

  // validate strategy between 'start' and 'end'
  if (ArithmeticBase::CheckStrategy(strategy) != SUCCESS) {
    return FAILED;
  }

  // validate strategy of weight
  Strategies expand_strategy = ExpandStrategy(strategy);
  Dimensions expand_begin_strategy = expand_strategy.at(0);
  Dimensions expand_end_strategy = expand_strategy.at(1);
  Dimensions expand_cmp_strategy;
  for (size_t i = 0; i < expand_begin_strategy.size(); ++i) {
    expand_cmp_strategy.push_back(std::max(expand_begin_strategy[i], expand_end_strategy[i]));
  }
  auto strategies = strategy->GetInputDim();
  Dimensions expand_weight_strategy = ExpandShape(expand_cmp_strategy, strategies.at(2));

  Shapes input_shapes = InferExpandShape();
  Shape expand_begin_shape = input_shapes.at(0);
  Shape expand_end_shape = input_shapes.at(1);
  Shape expand_cmp_shape;
  for (size_t i = 0; i < expand_begin_shape.size(); ++i) {
    expand_cmp_shape.push_back(std::max(expand_begin_shape[i], expand_end_shape[i]));
  }
  Shape expand_weight_shape = ExpandShape(expand_cmp_shape, inputs_shape_[2]);

  for (size_t i = 0; i < expand_cmp_shape.size(); ++i) {
    if ((expand_cmp_strategy[i] != expand_weight_strategy[i]) && (expand_cmp_shape[i] != 1) &&
        (expand_weight_shape[i] != 1)) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status LerpInfo::InferDevMatrixShape() {
  if (inputs_size_ == 2) {
    return ArithmeticBase::InferDevMatrixShape();
  }

  dev_matrix_shape_.clear();
  Strategies expand_strategy = ExpandStrategy(strategy_);
  Dimensions expand_start_strategy = expand_strategy.at(0);
  Dimensions expand_end_strategy = expand_strategy.at(1);
  auto strategies = strategy_->GetInputDim();
  Dimensions expand_weight_strategy = ExpandShape(expand_start_strategy, strategies.at(2));
  for (size_t i = 0; i < expand_start_strategy.size(); ++i) {
    if (expand_start_strategy[i] == expand_end_strategy[i] && expand_start_strategy[i] == expand_weight_strategy[i]) {
      dev_matrix_shape_.push_back(expand_start_strategy[i]);
    } else {
      dev_matrix_shape_.push_back(
        std::max(std::max(expand_start_strategy[i], expand_end_strategy[i]), expand_weight_strategy[i]));
    }
  }

  MS_LOG(INFO) << name_ << ": The dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status LerpInfo::InferTensorMap() {
  if (inputs_size_ == 2) {
    return ArithmeticBase::InferTensorMap();
  }

  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();
  // Generate inputs tensor map for 'start' and end, outputs tensor map
  if (ArithmeticBase::InferTensorMap() != SUCCESS) {
    return FAILED;
  }
  // Generate tensor map for 'weight'
  Strategies stra = strategy_->GetInputDim();
  Dimensions weight_strategy = stra.at(2);
  Strategies expand_strategy = ExpandStrategy(strategy_);
  Dimensions expand_start_strategy = expand_strategy.at(0);
  Dimensions expand_weight_strategy = ExpandShape(expand_start_strategy, weight_strategy);
  Shape dev_shape = dev_matrix_shape_;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      dev_shape.pop_back();
    } else {
      (void)dev_shape.erase(dev_shape.cbegin());
    }
  }
  inputs_tensor_map_.push_back(SetTensorMap(expand_weight_strategy, dev_shape, weight_strategy));
  return SUCCESS;
}

Status LerpInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (mirror_ops_.size() == kSizeTwo) {
    // Push empty mirror op for value
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

std::vector<StrategyPtr> LerpInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_size_ == 2) {
    return ArithmeticBase::GenerateOpStrategies(stage_id);
  }

  // search strategy for 'start' and 'end'
  auto sub_sp_vector = ArithmeticBase::GenerateOpStrategies(stage_id);

  // infer strategy for 'weight' according to strategy of 'start' and 'end'
  std::vector<StrategyPtr> sp_vector;
  for (const auto &sub_sp : sub_sp_vector) {
    auto expand_sub_strategies = ExpandStrategy(sub_sp);
    auto expand_start_strategy = expand_sub_strategies.at(0);
    auto expand_end_strategy = expand_sub_strategies.at(1);
    Dimensions expand_cmp_strategy;
    for (size_t i = 0; i < expand_start_strategy.size(); ++i) {
      expand_cmp_strategy.push_back(std::max(expand_start_strategy[i], expand_end_strategy[i]));
    }
    auto weight_shape = inputs_shape_.at(2);
    size_t offset = expand_cmp_strategy.size() - weight_shape.size();
    Dimensions weight_strategy;
    for (size_t i = 0; i < weight_shape.size(); ++i) {
      if (weight_shape[i] == 1) {
        weight_strategy.push_back(1);
      } else {
        weight_strategy.push_back(expand_cmp_strategy[offset + i]);
      }
    }
    auto strategies = sub_sp->GetInputDim();
    (void)strategies.emplace_back(weight_strategy);
    (void)sp_vector.emplace_back(std::make_shared<Strategy>(stage_id, strategies));
  }

  return sp_vector;
}

void LerpInfo::ReComputeBatchSplitFlagList() {
  // Set split flag for 'start' and 'end'
  ArithmeticBase::ReComputeBatchSplitFlagList();

  // if 'weight' is float, return
  if (inputs_shape_.size() == 2) {
    return;
  }

  // set split flag for 'weight'
  Shapes expand_shapes = InferExpandShape();
  Shape expand_a_shape = expand_shapes.at(0);
  Shape expand_weight_shape = ExpandShape(expand_a_shape, inputs_shape_.at(2));
  (expand_weight_shape.at(0) != 1) ? (split_flag_list_[2] = true) : (split_flag_list_[2] = false);
}

Status MaskedFillInfo::CheckStrategy(const StrategyPtr &strategy) {
  auto stra = strategy->GetInputDim();
  if (stra.size() == kSizeThree) {
    // The input strategy may be 2 or 3 in work script, so pop one here marking sure there are 2 in latter procession.
    stra.pop_back();
  }
  if (CheckStrategyByVector(stra, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  return BaseCheckStrategy(strategy);
}

Status MaskedFillInfo::GetAttrs() {
  if (inputs_shape_.size() == kSizeThree) {
    // For ArithmeticBase, the target inputs size is 2, so pop one here...
    inputs_shape_.pop_back();
  }
  input_size_ = inputs_shape_.size();
  if (input_size_ != 2 && input_size_ != 3) {
    MS_LOG(ERROR) << name_ << ": inputs_shape_.size() must be 2 or 3, but got size " << input_size_;
    return FAILED;
  }
  return SUCCESS;
}

Status MaskedFillInfo::InferTensorMap() {
  if (ArithmeticBase::InferTensorMap() != SUCCESS) {
    return FAILED;
  }

  if (input_size_ == kSizeThree) {
    // append a void tensor map for 0-dimensional tensor input 'value'
    (void)inputs_tensor_map_.emplace_back(TensorMap());
  }
  return SUCCESS;
}

std::vector<StrategyPtr> MaskedFillInfo::GenerateOpStrategies(int64_t stage_id) {
  auto sp_vector = ArithmeticBase::GenerateOpStrategies(stage_id);
  if (input_size_ == 3) {
    // append void strategy for input `value`
    for (size_t i = 0; i < sp_vector.size(); ++i) {
      auto strategies = sp_vector[i]->GetInputDim();
      (void)strategies.emplace_back(Dimensions());
      sp_vector[i] = std::make_shared<Strategy>(stage_id, strategies);
    }
  }
  return sp_vector;
}

Status MaskedFillInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (mirror_ops_.size() == kSizeTwo) {
    // Push empty mirror op for value
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

void ExpandSmallerShapes(const Shapes *bigger_size_shapes, Shapes *smaller_size_shapes) {
  size_t insert_num = bigger_size_shapes->size() - smaller_size_shapes->size();
  Shape map_none_shape(1, MAP_NONE);
  for (size_t num = 0; num < insert_num; ++num) {
    (void)smaller_size_shapes->insert(smaller_size_shapes->cbegin(), map_none_shape);
  }
}

Status AddInfo::CheckInputLayout() {
  // Check all device matrix should be the same
  if (inputs_tensor_info_.size() != kSizeTwo) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for add is " << inputs_tensor_info_.size() << " rather than 2.";
    return FAILED;
  }
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  if (in_layout0.device_arrangement_origin().array() != in_layout1.device_arrangement_origin().array()) {
    MS_LOG(ERROR) << "The device_matrix of input0 " << in_layout0.device_arrangement_origin().array()
                  << " dose not equal to device_matrix of input1 " << in_layout1.device_arrangement_origin().array();
    return FAILED;
  }

  Shapes input_shapes = InferExpandShape();
  Shape input_shape_0 = input_shapes.at(0);
  Shape input_shape_1 = input_shapes.at(1);

  Shapes tensormap0 = in_layout0.tensor_map_before();
  Shapes tensormap1 = in_layout1.tensor_map_before();
  if (tensormap0.size() > tensormap1.size()) {
    (void)ExpandSmallerShapes(&tensormap0, &tensormap1);
  } else {
    (void)ExpandSmallerShapes(&tensormap1, &tensormap0);
  }

  for (size_t i = 0; i < input_shape_0.size(); ++i) {
    if (tensormap0[i] != tensormap1[i] && input_shape_0[i] != 1 && input_shape_1[i] != 1) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }
  return SUCCESS;
}

TensorLayout AddInfo::InferOutputLayout() {
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  Shapes tensormap0 = in_layout0.tensor_map_before();
  Shapes tensormap1 = in_layout1.tensor_map_before();

  Shapes input_shapes = InferExpandShape();
  Shape input_a_shape = input_shapes.at(0);
  Shape input_b_shape = input_shapes.at(1);

  for (size_t i = 0; i < input_a_shape.size(); ++i) {
    input_a_shape[i] = (input_a_shape[i] == 1) ? input_b_shape[i] : input_a_shape[i];
  }

  Shapes output_tensormap;
  Shape map_none_shape(1, MAP_NONE);
  size_t len_diff = 0;
  if (tensormap0.size() > tensormap1.size()) {
    output_tensormap = tensormap0;
    len_diff = tensormap0.size() - tensormap1.size();
    for (size_t i = 0; i < tensormap1.size(); ++i) {
      output_tensormap[i + len_diff] =
        tensormap0[i + len_diff] == map_none_shape ? tensormap1[i] : tensormap0[i + len_diff];
    }
  } else {
    output_tensormap = tensormap1;
    len_diff = tensormap1.size() - tensormap0.size();
    for (size_t i = 0; i < tensormap0.size(); ++i) {
      output_tensormap[i + len_diff] =
        tensormap1[i + len_diff] == map_none_shape ? tensormap0[i] : tensormap1[i + len_diff];
    }
  }

  TensorLayout output_tensor_layout;
  output_tensor_layout.InitFromExtendVector(in_layout0.device_arrangement_origin().array(), output_tensormap,
                                            input_a_shape);
  return output_tensor_layout;
}

Status AddInfo::InferOutputTensorInfo() {
  output_infer_tensor_layout_ = InferOutputLayout();
  if (output_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "The infer output shape " << output_infer_tensor_layout_.tensor_shape_before().array()
                  << " dose not match the output shape " << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status AddInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << "The size of output_tensor_layout for matmul is " << outputs_tensor_info_.size()
                  << " rather than 1.";
    return FAILED;
  }

  if (output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    MS_LOG(ERROR) << "Parameter of output tensor layout for add is not allowed to be set by users.";
    return FAILED;
  }
  MS_LOG(INFO) << "Using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

REGISTER(SubInfo);
REGISTER(AddInfo);
REGISTER(MulInfo);
REGISTER(DivInfo);
REGISTER(ModInfo);
REGISTER(RealDivInfo);
REGISTER(FloorDivInfo);
REGISTER(FloorModInfo);
REGISTER(PowInfo);
REGISTER(AssignSubInfo);
REGISTER(AssignInfo);
REGISTER(AssignAddInfo);
REGISTER(SigmoidCrossEntropyWithLogitsInfo);
REGISTER(Atan2Info);
REGISTER(DivNoNanInfo);
REGISTER(LogicalAndInfo);
REGISTER(LogicalOrInfo);
REGISTER(BitwiseAndInfo);
REGISTER(BitwiseOrInfo);
REGISTER(BitwiseXorInfo);
REGISTER(MulNoNanInfo);
REGISTER(TruncateDivInfo);
REGISTER(TruncateModInfo);
REGISTER(XdivyInfo);
REGISTER(XlogyInfo);
REGISTER(HypotInfo);
REGISTER(IgammaInfo);
REGISTER(IgammacInfo);
REGISTER(LeftShiftInfo);
REGISTER(RightShiftInfo);
REGISTER(NextAfterInfo);
REGISTER(ZetaInfo);
REGISTER(GcdInfo);
REGISTER(LerpInfo);
REGISTER(SquaredDifferenceInfo);
REGISTER(MaskedFillInfo);
}  // namespace parallel
}  // namespace mindspore
