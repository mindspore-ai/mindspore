/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include <unordered_set>

#include "include/utils/parallel_context.h"
#include "frontend/parallel/graph_util/generate_graph.h"
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
      if ((input_a_shape[i] == -1 || input_b_shape[i] == -1) && (sub_a_strategy[i] == 1 || sub_b_strategy[i] == 1)) {
        MS_LOG(WARNING) << name_ << ": the dim " << i << " is dynamic and broadcast, ignore the check";
        continue;
      }
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

Status ArithmeticBase::ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  Attr output_nums_attr = {"output_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_begin_attrs = {output_nums_attr};

  bool first_input_interleaved = inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel();
  bool second_input_interleaved = inputs_tensor_info_[kIndex1].tensor_layout().IsInterleavedParallel();

  std::vector<AnfNodePtr> virtual_converter_end_inputs_vector;
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes;

  if (first_input_interleaved && second_input_interleaved) {
    // take first as begin 1
    auto virtual_converter_begin_1 = gen_g.PushBack(
      {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
    input_nodes.push_back(std::make_pair(virtual_converter_begin_1, 1));
    // take second as begin 2
    auto virtual_converter_begin_2 = gen_g.PushBack(
      {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
    input_nodes.push_back(std::make_pair(virtual_converter_begin_2, kIndexTwo));
    for (int64_t i = 0; i < interleaved_num; ++i) {
      auto tuple_get_item_1 =
        gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin_1, CreatInt64Imm(i)});
      auto tuple_get_item_2 =
        gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin_2, CreatInt64Imm(i)});
      auto arithmetic = gen_g.PushBack({gen_g.NewOpInst(prim_name_), tuple_get_item_1, tuple_get_item_2});
      virtual_converter_end_inputs_vector.push_back(arithmetic);
    }
  } else {
    auto virtual_converter_begin = gen_g.PushBack(
      {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
    int64_t take_input = first_input_interleaved ? kIndexOne : kIndexTwo;
    input_nodes.push_back(std::make_pair(virtual_converter_begin, take_input));
    for (int64_t i = 0; i < interleaved_num; ++i) {
      auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin, CreatInt64Imm(i)});
      AnfNodePtr arithmetic;
      if (first_input_interleaved) {
        arithmetic = gen_g.PushBack({gen_g.NewOpInst(prim_name_), tuple_get_item, gen_g.virtual_input_node()});
        input_nodes.push_back(std::make_pair(arithmetic, kIndexTwo));
      } else {
        arithmetic = gen_g.PushBack({gen_g.NewOpInst(prim_name_), gen_g.virtual_input_node(), tuple_get_item});
        input_nodes.push_back(std::make_pair(arithmetic, 1));
      }
      virtual_converter_end_inputs_vector.push_back(arithmetic);
    }
  }
  Attr input_nums_attr = {"input_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_end_attrs = {input_nums_attr};
  std::vector<AnfNodePtr> virtual_converter_end_inputs = {
    gen_g.NewOpInst(VIRTUAL_CONVERTER_END, virtual_converter_end_attrs)};
  std::copy(virtual_converter_end_inputs_vector.begin(), virtual_converter_end_inputs_vector.end(),
            std::back_inserter(virtual_converter_end_inputs));
  auto virtual_converter_end = gen_g.PushBack(virtual_converter_end_inputs);
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, virtual_converter_end));
  return SUCCESS;
}

ReplaceGraphPtr ArithmeticBase::replace_graph(const CNodePtr &cnode) {
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel() ||
      inputs_tensor_info_[kIndex1].tensor_layout().IsInterleavedParallel()) {
    if (ComputeReplaceGraphForInterleaved(cnode) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << " splitting micro interleaved failed.";
    }
    return replace_graph_;
  }
  return replace_graph_;
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
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << " : Recompute batch split flag list is wrong.";
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

void ExpandSmallerShapes(const Shapes *bigger_size_shapes, Shapes *smaller_size_shapes) {
  size_t insert_num = bigger_size_shapes->size() - smaller_size_shapes->size();
  Shape map_none_shape(1, MAP_NONE);
  for (size_t num = 0; num < insert_num; ++num) {
    (void)smaller_size_shapes->insert(smaller_size_shapes->cbegin(), map_none_shape);
  }
}

Status ArithmeticBase::InferOutputTensorInfo() {
  output_infer_tensor_layout_ = InferOutputLayout();
  if (output_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "The infer output shape " << output_infer_tensor_layout_.tensor_shape_before().array()
                  << " does not match the output shape " << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status ArithmeticBase::CheckInputLayout() {
  // Check all device matrix should be the same
  if (inputs_tensor_info_.size() != kSizeTwo) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "The size of input_tensor_layout for " << name_ << " is " << inputs_tensor_info_.size()
                   << " rather than 2.";
    } else {
      MS_LOG(ERROR) << "The size of input_tensor_layout for " << name_ << " is " << inputs_tensor_info_.size()
                    << " rather than 2.";
    }
    return FAILED;
  }
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  if (in_layout0.device_arrangement_origin().array() != in_layout1.device_arrangement_origin().array()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "The device_matrix of input0 " << in_layout0.device_arrangement_origin().array()
                   << " does not equal to device_matrix of input1 " << in_layout1.device_arrangement_origin().array();
    } else {
      MS_LOG(ERROR) << "The device_matrix of input0 " << in_layout0.device_arrangement_origin().array()
                    << " does not equal to device_matrix of input1 " << in_layout1.device_arrangement_origin().array();
    }
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
      if (is_in_layout_propagation_) {
        MS_LOG(INFO) << name_ << " : Invalid strategy. The " << i << "th dim of input 0 tensor map is " << tensormap0[i]
                     << " is not equal to input 1 tensor map " << tensormap1[i] << ", also " << i
                     << "th input_shape0 and input_shape1 are not equal to 1 which is " << input_shape_0[i] << " and "
                     << input_shape_1[i];
      } else {
        MS_LOG(ERROR) << name_ << " : Invalid strategy. The " << i << "th dim of input 0 tensor map is "
                      << tensormap0[i] << " is not equal to input 1 tensor map " << tensormap1[i] << ", also " << i
                      << "th input_shape0 and input_shape1 are not equal to 1 which is " << input_shape_0[i] << " and "
                      << input_shape_1[i];
      }
      return FAILED;
    }
  }
  return SUCCESS;
}

TensorLayout ArithmeticBase::InferOutputLayout() {
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  Shapes tensormap0 = in_layout0.tensor_map_before();
  Shapes tensormap1 = in_layout1.tensor_map_before();

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
                                            outputs_shape_[0]);
  return output_tensor_layout;
}

Status ArithmeticBase::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "The size of output_tensor_layout for " << name_ << " is " << outputs_tensor_info_.size()
                   << " rather than 1.";
    } else {
      MS_LOG(ERROR) << "The size of output_tensor_layout for " << name_ << " is " << outputs_tensor_info_.size()
                    << " rather than 1.";
    }
    return FAILED;
  }

  if (output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "Parameter of output tensor layout for " << name_ << " is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "Parameter of output tensor layout for " << name_ << " is not allowed to be set by users.";
    }
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Using output tensor layout infer by input tensor layout.";
  UpdateOutputTensorInfoForInterleaved();
  return SUCCESS;
}

std::vector<StrategyPtr> ArithmeticBase::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};
  if (inputs_shape_.size() < 2) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Size of inputs must be greater than or equal to 2, but got size "
                                        << inputs_shape_.size();
  }
  Shapes inputs_shape(inputs_shape_.cbegin(), inputs_shape_.cbegin() + 2);

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << " : Generate strategies with broadcast failed.";
  }
  MS_LOG(INFO) << name_ << " : Generate strategies with broadcast success.";

  return sp_vector;
}

Status OuterInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status OuterInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the shard strategies is empty.";
    return FAILED;
  }
  if (strategies.size() != kSizeTwo) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", it has two 1D inputs so the shard strategies.size() "
                  << "should be 2, but the strategies is " << strategies << " and its size is " << strategies.size()
                  << ".";
    return FAILED;
  }
  Dimensions input_strategy = strategies.at(kIndex0);
  Dimensions vec2_strategy = strategies.at(kIndex1);
  dev_matrix_shape_.push_back(input_strategy[kIndex0]);
  dev_matrix_shape_.push_back(vec2_strategy[kIndex0]);
  return SUCCESS;
}

Status OuterInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();
  // Get dev matrix without repeated calculation
  Shape dev_matrix = dev_matrix_shape_;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      dev_matrix.pop_back();
    } else {
      (void)dev_matrix.erase(dev_matrix.cbegin());
    }
  }
  if (dev_matrix.size() != kSizeTwo) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", it has two 1D inputs so the dev_matrix.size() "
                  << "before extend should be 2, but the dev_matrix is " << dev_matrix << " and its size is "
                  << dev_matrix.size() << ".";
    return FAILED;
  }
  inputs_tensor_map_.push_back({kIndex1});
  inputs_tensor_map_.push_back({kIndex0});
  outputs_tensor_map_.push_back({kIndex1, kIndex0});  // output
  return SUCCESS;
}

std::shared_ptr<Strategies> OuterInfo::GenerateBatchStrategies() {
  Dimensions batch_input_strategy(inputs_shape_[kIndex0].size(), 1);
  Dimensions batch_vec2_strategy(inputs_shape_[kIndex1].size(), 1);
  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);
  Strategies strategy_v;

  batch_input_strategy[kIndex0] = stage_device_size_;

  strategy_v = {batch_input_strategy, batch_vec2_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

bool TensorMapHasRepeatElem(const std::vector<Shape> &input_tensor_map, const std::vector<Shape> &vec2_tensor_map) {
  std::unordered_set<int64_t> input_tensor_map_set;
  for (auto sub_map_i : input_tensor_map) {
    for (auto elem_i : sub_map_i) {
      input_tensor_map_set.insert(elem_i);
    }
  }
  for (auto sub_map_v : vec2_tensor_map) {
    for (auto elem_v : sub_map_v) {
      if (input_tensor_map_set.find(elem_v) != input_tensor_map_set.end()) {
        return true;
      }
    }
  }
  return false;
}

Status OuterInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeTwo) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of inputs_tensor_info should be 2, but got "
                   << inputs_tensor_info_.size() << ".";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of inputs_tensor_info should be 2, but got "
                    << inputs_tensor_info_.size() << ".";
    }
    return FAILED;
  }
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto vec2_tensor_layout = inputs_tensor_info_[kIndex1].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  auto vec2_tensor_map = vec2_tensor_layout.tensor_map_before();
  if (TensorMapHasRepeatElem(input_tensor_map, vec2_tensor_map)) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", there cannot be duplicate elements in "
                   << "input_tensor_map and vec2_tensor_map.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", there cannot be duplicate elements in "
                    << "input_tensor_map and vec2_tensor_map.";
    }
    return FAILED;
  }
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();
  return SUCCESS;
}

Status OuterInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto vec2_tensor_layout = inputs_tensor_info_[kIndex1].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  auto vec2_tensor_map = vec2_tensor_layout.tensor_map_before();
  Shapes outputs_tensor_map = {};
  outputs_tensor_map.push_back(input_tensor_map[kIndex0]);
  outputs_tensor_map.push_back(vec2_tensor_map[kIndex0]);

  if ((output_infer_tensor_layout_.InitFromExtendVector(dev_matrix_shape_, outputs_tensor_map,
                                                        outputs_shape_[kIndex0]) != SUCCESS)) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the output_tensor_layout init failed.";
    return FAILED;
  }
  if (output_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the infer output shape "
                  << output_infer_tensor_layout_.tensor_shape_before().array() << " does not match the output shape "
                  << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);  // output
  return SUCCESS;
}

ReplaceGraphPtr OuterInfo::replace_graph(const CNodePtr &cnode) {
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel() ||
      inputs_tensor_info_[kIndex1].tensor_layout().IsInterleavedParallel()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "For distributed operator " << name_ << " it does not support "
                                       << "interleaved parallel.";
  }
  return replace_graph_;
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

Status AddcmulExtInfo::GetAttrs() {
  inputs_size_ = inputs_shape_.size();
  if (inputs_size_ != kSizeThree) {
    MS_LOG(ERROR) << name_ << ": Inputs size should be 3, but get " << inputs_size_;
    return FAILED;
  }
  return SUCCESS;
}

// expand shapes to the same size, e.g. [a, b, c], [d, e], [f] -> [a, b, c], [1, d, e], [1, 1, f]
Shapes ExpandShapes(const Shapes &inputs_shape) {
  size_t larger_index = 0;
  for (size_t i = 1; i < inputs_shape.size(); ++i) {
    if (inputs_shape[i].size() > inputs_shape[larger_index].size()) {
      larger_index = i;
    }
  }
  Shapes expand_inputs_shape;
  for (const auto &shape : inputs_shape) {
    if (shape.size() < inputs_shape[larger_index].size()) {
      expand_inputs_shape.push_back(ExpandShape(inputs_shape[larger_index], shape));
    } else {
      expand_inputs_shape.push_back(shape);
    }
  }
  return expand_inputs_shape;
}

// infer strategies after broadcast, e.g. [a, 1, c], [1, b, 1] -> [a, b, c]
Dimensions InferBroadcastStrategy(const Strategies &expand_strategies) {
  size_t strategy_size = expand_strategies[0].size();
  Dimensions broadcast_strategy;
  for (size_t i = 0; i < strategy_size; ++i) {
    int64_t dim_max = -1;
    for (const auto &strategy : expand_strategies) {
      dim_max = std::max(dim_max, strategy[i]);
    }
    broadcast_strategy.push_back(dim_max);
  }
  return broadcast_strategy;
}

std::vector<StrategyPtr> AddcmulExtInfo::GenerateOpStrategies(int64_t stage_id) {
  // generate strategies for 2 inputs with larger size
  size_t smaller_index = 0;
  for (size_t i = 1; i < inputs_size_; ++i) {
    if (inputs_shape_[i].size() <= inputs_shape_[smaller_index].size()) {
      smaller_index = i;
    }
  }
  Shapes sub_inputs_shape;
  Shapes sub_splittable_inputs;
  for (size_t i = 0; i < inputs_size_; ++i) {
    if (i != smaller_index) {
      sub_inputs_shape.push_back(inputs_shape_[i]);
      sub_splittable_inputs.emplace_back(inputs_shape_[i].size(), 1);
    }
  }
  std::vector<StrategyPtr> sub_sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, sub_inputs_shape, sub_splittable_inputs, &sub_sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies with broadcast failed.";
  }

  // infer strategy for input with smaller size
  std::vector<StrategyPtr> sp_vector;
  for (const auto &sub_strategies : sub_sp_vector) {
    Strategies strategies = sub_strategies->GetInputDim();
    Strategies expand_sub_strategies = ExpandShapes(strategies);
    Dimensions broadcast_sub_strategy = InferBroadcastStrategy(expand_sub_strategies);
    Shape smaller_shape = inputs_shape_[smaller_index];
    size_t offset = broadcast_sub_strategy.size() - smaller_shape.size();
    Dimensions smaller_strategy;
    for (size_t i = 0; i < smaller_shape.size(); ++i) {
      if (smaller_shape[i] == 1) {
        smaller_strategy.push_back(1);
      } else {
        smaller_strategy.push_back(broadcast_sub_strategy[offset + i]);
      }
    }
    strategies.insert(strategies.begin() + smaller_index, smaller_strategy);
    sp_vector.emplace_back(std::make_shared<Strategy>(stage_id, strategies));
  }

  return sp_vector;
}

Status AddcmulExtInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // broadcast shape and strategy
  Shapes expand_inputs_shape = ExpandShapes(inputs_shape_);
  expand_strategies_ = ExpandShapes(strategy->GetInputDim());
  broadcast_strategy_ = InferBroadcastStrategy(expand_strategies_);
  MS_LOG(DEBUG) << name_ << ": inputs_shape: " << ShapesToString(inputs_shape_)
                << " expand_inputs_shape: " << ShapesToString(expand_inputs_shape)
                << " strategies: " << StrategyToString(strategy->GetInputDim())
                << " expand_strategies: " << StrategyToString(expand_strategies_)
                << " broadcast_strategy: " << ShapeToString(broadcast_strategy_);

  // check if input strategies are equal for non-broadcast dim
  size_t expand_shape_size = expand_inputs_shape[0].size();
  for (size_t i = 0; i < inputs_size_; ++i) {
    for (size_t j = 0; j < expand_shape_size; ++j) {
      if (expand_strategies_[i][j] != broadcast_strategy_[j] && expand_inputs_shape[i][j] != 1 &&
          (expand_inputs_shape[i][j] != -1 || expand_strategies_[i][j] != 1)) {
        MS_LOG(ERROR) << name_ << ": Invalid strategy. inputs_shape: " << ShapesToString(inputs_shape_)
                      << " strategy: " << StrategyToString(strategy->GetInputDim());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status AddcmulExtInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = broadcast_strategy_;
  MS_LOG(DEBUG) << name_ << ": dev_matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status AddcmulExtInfo::InferTensorMap() {
  // get dev matrix without repeated calculation
  Shape dev_shape = dev_matrix_shape_;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      dev_shape.pop_back();
    } else {
      (void)dev_shape.erase(dev_shape.cbegin());
    }
  }

  Strategies stra = strategy_->GetInputDim();
  for (size_t i = 0; i < stra.size(); ++i) {
    inputs_tensor_map_.push_back(SetTensorMap(expand_strategies_[i], dev_shape, stra[i]));
  }

  size_t dev_shape_size = dev_shape.size();
  Shape tensor_map_index;
  for (size_t i = 0; i < dev_shape_size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(dev_shape_size - i - 1));
  }
  outputs_tensor_map_.push_back(tensor_map_index);

  return SUCCESS;
}

// infer tensor map after broadcast, e.g. [a, b, -1, d], [c, d] -> [a, b, c, d]
Shape InferBroadcastTensorMap(const TensorMaps &tensor_maps) {
  size_t larger_index = 0;
  for (size_t i = 1; i < tensor_maps.size(); ++i) {
    if (tensor_maps[i].size() > tensor_maps[larger_index].size()) {
      larger_index = i;
    }
  }
  Shape broadcast_tensor_map = tensor_maps[larger_index];
  size_t broadcast_size = broadcast_tensor_map.size();
  for (size_t i = 0; i < broadcast_size; ++i) {
    if (broadcast_tensor_map[i] != MAP_NONE) {
      continue;
    }
    for (const auto &tensor_map : tensor_maps) {
      size_t offset = broadcast_size - tensor_map.size();
      if (i >= offset && tensor_map[i - offset] != MAP_NONE) {
        broadcast_tensor_map[i] = tensor_map[i - offset];
        break;
      }
    }
  }
  return broadcast_tensor_map;
}

Status AddcmulExtInfo::InferOutputTensorMap() {
  if (inputs_tensor_map_[kIndex0] == inputs_tensor_map_[kIndex1] &&
      inputs_tensor_map_[kIndex0] == inputs_tensor_map_[kIndex2]) {
    outputs_tensor_map_.push_back(inputs_tensor_map_[0]);
    return SUCCESS;
  }

  Shape output_tensor_map = InferBroadcastTensorMap(inputs_tensor_map_);
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(DEBUG) << name_ << ": outputs_tensor_map: " << ShapesToString(outputs_tensor_map_);

  return SUCCESS;
}

// expand tensor maps to the same size, e.g. [a, b, c], [d, e] -> [a, b, c], [-1, d, e]
void ExpandTensorMaps(std::vector<Shapes> *tensor_maps) {
  size_t larger_index = 0;
  for (size_t i = 1; i < tensor_maps->size(); ++i) {
    if ((*tensor_maps)[i].size() > (*tensor_maps)[larger_index].size()) {
      larger_index = i;
    }
  }
  const Shapes &larger_tensor_map = (*tensor_maps)[larger_index];
  for (auto &tensor_map : *tensor_maps) {
    if (tensor_map.size() < larger_tensor_map.size()) {
      ExpandSmallerShapes(&larger_tensor_map, &tensor_map);
    }
  }
}

TensorLayout AddcmulExtInfo::InferOutputLayout() {
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  auto in_layout2 = inputs_tensor_info_[kIndex2].tensor_layout();
  // broadcast inputs tensor map to get output tensor map, e.g [a, b, c, d], [-1, d] -> [a, b, c, d]
  std::vector<Shapes> inputs_tensor_map = {in_layout0.tensor_map_before(), in_layout1.tensor_map_before(),
                                           in_layout2.tensor_map_before()};
  ExpandTensorMaps(&inputs_tensor_map);
  Shapes output_tensor_map = inputs_tensor_map[0];
  Shape map_none_shape(1, MAP_NONE);
  for (size_t i = 0; i < output_tensor_map.size(); ++i) {
    if (output_tensor_map[i] != map_none_shape) {
      continue;
    }
    auto it = std::find_if(inputs_tensor_map.begin(), inputs_tensor_map.end(),
                           [i, map_none_shape](const auto &tensor_map) { return tensor_map[i] != map_none_shape; });
    if (it != inputs_tensor_map.end()) {
      output_tensor_map[i] = (*it)[i];
    }
  }

  TensorLayout output_tensor_layout;
  output_tensor_layout.InitFromExtendVector(in_layout0.device_arrangement_origin().array(), output_tensor_map,
                                            outputs_shape_[0]);
  return output_tensor_layout;
}

Status AddcmulExtInfo::CheckLayoutConfig() {
  // check size
  size_t inputs_size = inputs_shape_.size();
  if (inputs_size != inputs_tensor_map_.size()) {
    MS_LOG(ERROR) << name_ << ": inputs_size " << inputs_size << " is not equal to tensor_maps_size "
                  << inputs_tensor_map_.size();
    return FAILED;
  }
  size_t larger_index = 0;
  for (size_t i = 0; i < inputs_size; ++i) {
    size_t input_shape_size = inputs_shape_[i].size();
    if (input_shape_size != inputs_tensor_map_[i].size()) {
      MS_LOG(ERROR) << name_ << ": " << i << "-th input size " << input_shape_size
                    << " is not equal to tensor map size " << inputs_tensor_map_[i].size();
      return FAILED;
    }
    if (input_shape_size > inputs_shape_[larger_index].size()) {
      larger_index = i;
    }
  }

  // check with input with larger size
  Shape larger_input_shape = inputs_shape_[larger_index];
  Shape larger_tensor_map = inputs_tensor_map_[larger_index];
  for (size_t i = 0; i < inputs_size; ++i) {
    if (i == larger_index) {
      continue;
    }
    size_t offset = larger_input_shape.size() - inputs_shape_[i].size();
    for (size_t j = 0; i < inputs_shape_[i].size(); ++j) {
      if (inputs_shape_[i][j] == larger_input_shape[j + offset] &&
          inputs_tensor_map_[i][j] != larger_tensor_map[j + offset]) {
        MS_LOG(ERROR) << name_ << ": Invalid tensor map, inputs_tensor_map: " << ShapesToString(inputs_tensor_map_)
                      << " inputs_shape: " << ShapesToString(inputs_shape_);
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status AddcmulExtInfo::CheckInputLayout() {
  // check if all device matrix are same
  if (inputs_tensor_info_.size() != kSizeThree) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": input_tensor_layout size should be 3, but get " << inputs_tensor_info_.size();
    } else {
      MS_LOG(ERROR) << name_ << ": input_tensor_layout size should be 3, but get " << inputs_tensor_info_.size();
    }
    return FAILED;
  }
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  auto in_layout2 = inputs_tensor_info_[kIndex2].tensor_layout();
  auto dev_matrix0 = in_layout0.device_arrangement_origin().array();
  auto dev_matrix1 = in_layout1.device_arrangement_origin().array();
  auto dev_matrix2 = in_layout2.device_arrangement_origin().array();
  if (dev_matrix0 != dev_matrix1 || dev_matrix0 != dev_matrix2) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": Inputs device matrix are not equal. dev_matrix0: " << ShapeToString(dev_matrix0)
                   << " dev_matrix1: " << ShapeToString(dev_matrix1) << " dev_matrix2: " << ShapeToString(dev_matrix2);
    } else {
      MS_LOG(ERROR) << name_ << ": Inputs device matrix are not equal. dev_matrix0: " << ShapeToString(dev_matrix0)
                    << " dev_matrix1: " << ShapeToString(dev_matrix1) << " dev_matrix2: " << ShapeToString(dev_matrix2);
    }
    return FAILED;
  }

  // check if tensor map are equal for non-broadcast dim
  Shapes inputs_shape = ExpandShapes(inputs_shape_);
  std::vector<Shapes> tensor_maps = {in_layout0.tensor_map_before(), in_layout1.tensor_map_before(),
                                     in_layout2.tensor_map_before()};
  ExpandTensorMaps(&tensor_maps);
  size_t expand_size = inputs_shape[0].size();
  for (size_t i = 1; i < tensor_maps.size(); ++i) {
    for (size_t j = 0; j < expand_size; ++j) {
      if (tensor_maps[0][j] != tensor_maps[i][j] && inputs_shape[0][j] != 1 && inputs_shape[i][j] != 1) {
        MS_LOG(ERROR) << name_ << " : Invalid strategy. inputs_tensor_map0: " << ShapesToString(tensor_maps[0])
                      << " inputs_tensor_map" << i << ": " << ShapesToString(tensor_maps[i]) << " "
                      << " inputs_shape0 " << ShapeToString(inputs_shape_[0]) << " inputs_shape" << i << ": "
                      << ShapeToString(inputs_shape_[i]);
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

void AddcmulExtInfo::ReComputeBatchSplitFlagList() {
  Shapes expand_inputs_shape = ExpandShapes(inputs_shape_);
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    if (expand_inputs_shape[i].empty() || expand_inputs_shape[i].at(0) == 1) {
      split_flag_list_[i] = false;
    } else {
      split_flag_list_[i] = true;
    }
  }
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
REGISTER(AddExtInfo);
REGISTER(SubExtInfo);
REGISTER(DivModInfo);
REGISTER(OuterInfo);
REGISTER(AddcmulExtInfo);
REGISTER(PolarInfo);
REGISTER(IsCloseInfo);  // has not bprop
REGISTER(RemainderTensorTensorInfo);
REGISTER(FmodScalarInfo);
REGISTER(MulsInfo);
REGISTER(FmodTensorInfo);
REGISTER(InplaceCopyInfo);
REGISTER(InplaceAddExtInfo);
REGISTER(InplaceSubExtInfo);
REGISTER(InplaceMulInfo);
REGISTER(InplaceDivInfo);
REGISTER(InplaceFloorDivideInfo);
REGISTER(InplaceAddsExtInfo);
REGISTER(InplaceSubScalarInfo);
REGISTER(InplaceMulsInfo);
REGISTER(InplaceDivsInfo);
REGISTER(InplaceFloorDividesInfo);
}  // namespace parallel
}  // namespace mindspore
