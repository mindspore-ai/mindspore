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

#include "frontend/parallel/ops_info/transpose_info.h"

#include <memory>
#include <vector>
#include <numeric>
#include <utility>
#include <algorithm>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"
#include "include/common/utils/convert_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status TransposeInfo::CheckStrategy(const StrategyPtr &strategy) { return CheckStrategyValue(strategy, inputs_shape_); }

Status TransposeInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  input_strategy_ = stra.at(0);
  for (auto &iter : input_strategy_) {
    dev_matrix_shape_.push_back(iter);
  }
  return SUCCESS;
}

// there is no Parameter for Transpose Primitive, so no need to do all reduce
Status TransposeInfo::InferMirrorOps() { return SUCCESS; }

// there is no reduction dimension for forward computation of Transpose Primitive, so no need to do all reduce
Status TransposeInfo::InferForwardCommunication() { return SUCCESS; }

/*
 * get perm input of Transpose Primitive
 * perm is a permutation of the dimensions of input
 * the result is saved in axis_v_
 */
Status TransposeInfo::ComputeAxis() {
  if (input_value_[1] == nullptr) {
    MS_LOG(ERROR) << name_ << ": input_value_[1] is nullptr.";
    return FAILED;
  }
  std::vector<ValuePtr> elements;
  ValueTuplePtr dim_tuple = input_value_[1]->cast<ValueTuplePtr>();
  if (dim_tuple == nullptr) {
    MS_LOG(ERROR) << name_ << ": input_value_[1] must be ValueTuplePtr.";
    return FAILED;
  }
  elements = dim_tuple->value();
  if (elements.size() != inputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_ << ": elements size must be equal to inputs shape[0] size.";
    return FAILED;
  }
  axis_v_.clear();
  for (auto &element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t axis = element->cast<Int64ImmPtr>()->value();
      axis_v_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }

  for (int64_t i = 0; i < SizeToLong(axis_v_.size()); i++) {
    auto iter = std::find(axis_v_.begin(), axis_v_.end(), i);
    if (iter == axis_v_.end()) {
      MS_LOG(ERROR) << name_ << ": axis_v_ must be a permutation.";
    }
  }
  return SUCCESS;
}

// the output tensor map is the permutation of input tensor map, the permutation is axis_v
Status TransposeInfo::InferTensorMap() {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": inputs_shape_ and outputs_shape_ size must be 1, inputs shape and outputs shape is "
                  << inputs_shape_.size() << ", " << outputs_shape_.size();
    return FAILED;
  }

  Shape tensor_map_index_input;
  for (size_t j = 0; j < inputs_shape_[0].size(); ++j) {
    tensor_map_index_input.push_back(SizeToLong(inputs_shape_[0].size() - j - 1));
  }
  inputs_tensor_map_.push_back(tensor_map_index_input);

  Shape tensor_map_index_output = tensor_map_index_input;
  for (uint64_t i = 0; i < tensor_map_index_output.size(); i++) {
    tensor_map_index_output[i] = tensor_map_index_input[LongToUlong(axis_v_[i])];
  }
  outputs_tensor_map_.push_back(tensor_map_index_output);
  return SUCCESS;
}

Status TransposeInfo::InferOutputTensorMap() {
  Shape input_tensor_map = inputs_tensor_map_[0];
  Shape output_tensor_map = input_tensor_map;
  for (uint64_t i = 0; i < output_tensor_map.size(); i++) {
    output_tensor_map[i] = input_tensor_map[LongToUlong(axis_v_[i])];
  }
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(INFO) << name_ << ": the input tensor map is " << inputs_tensor_map_ << ", the output tensor map is "
               << outputs_tensor_map_;
  return SUCCESS;
}

// compute axis_v_ during this method
Status TransposeInfo::GetAttrs() { return ComputeAxis(); }

Status TransposeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status TransposeInfo::InferOutputTensorInfo() {
  output_infer_tensor_layout_ = InferOutputLayout();
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

std::vector<std::vector<int64_t>> CalculateExchangeAxes(const std::vector<int64_t> &target_arrangement) {
  std::vector<std::vector<int64_t>> exchange_axes;
  std::vector<int64_t> origin_arrangment(target_arrangement.size());
  std::iota(origin_arrangment.begin(), origin_arrangment.end(), 0);
  for (size_t i = 0; i < origin_arrangment.size(); ++i) {
    if (origin_arrangment[i] != target_arrangement[i]) {
      auto it = std::find(origin_arrangment.begin(), origin_arrangment.end(), target_arrangement[i]);
      auto target_pos = it - origin_arrangment.begin();
      std::vector<int64_t> exchange_axis = {SizeToLong(i), target_pos};
      exchange_axes.push_back(exchange_axis);
      std::swap(origin_arrangment[i], origin_arrangment[target_pos]);
    }
  }
  return exchange_axes;
}

std::vector<int64_t> SwapElement(const std::vector<int64_t> &input_device_arrangement, const int64_t og_start_pos,
                                 const int64_t target_start_pos, const int64_t target_end_pos,
                                 const Shape &temp_og_device_arrangement, const Shape &temp_target_device_arrangement) {
  std::vector<int64_t> expected_device_arrangement = input_device_arrangement;
  expected_device_arrangement.insert(expected_device_arrangement.begin() + target_start_pos,
                                     temp_og_device_arrangement.begin(), temp_og_device_arrangement.end());
  auto og_current_pos = og_start_pos + temp_og_device_arrangement.size();
  expected_device_arrangement.erase(
    expected_device_arrangement.begin() + og_current_pos,
    expected_device_arrangement.begin() + og_current_pos + temp_og_device_arrangement.size());
  if ((og_start_pos - target_end_pos) != 1) {
    expected_device_arrangement.insert(expected_device_arrangement.begin() + og_current_pos,
                                       temp_target_device_arrangement.begin(), temp_target_device_arrangement.end());
    auto target_current_pos = target_start_pos + temp_target_device_arrangement.size();
    expected_device_arrangement.erase(
      expected_device_arrangement.begin() + target_current_pos,
      expected_device_arrangement.begin() + target_current_pos + temp_target_device_arrangement.size());
  }
  return expected_device_arrangement;
}

bool IsValidTensorMap(const Shape &tensor_map) {
  if (tensor_map.empty()) {
    return false;
  }
  int64_t last_value = tensor_map[0];
  for (size_t i = 1; i < tensor_map.size(); ++i) {
    auto current_value = tensor_map[i];
    // If tensor map is not ordered in ascending order or descending order, and diff between adjacent elements is not 1
    // return false
    if (std::abs(last_value - current_value) != 1) {
      return false;
    }
    last_value = current_value;
  }

  return true;
}

Status TransposeInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << name_ << ": The size of output tensor info must be 1, but got " << outputs_tensor_info_.size();
    return FAILED;
  }
  if (!output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    MS_LOG(INFO) << name_ << ": Using output tensor layout infer by input tensor layout.";
    UpdateOutputTensorInfoForInterleaved();
    return SUCCESS;
  }

  auto in_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto out_layout = outputs_tensor_info_[kIndex0].tensor_layout();
  MS_LOG(INFO) << name_ << ": The output tensor layout is " << out_layout.ToString() << ", axis_v_ is " << axis_v_;

  auto out_tensor_layout = InferOutputLayout();
  // output layout is the same as inferred (transpose the tensor map)
  if (out_layout == out_tensor_layout) {
    UpdateOutputTensorInfoForInterleaved();
    return SUCCESS;
  }

  auto input_device_arrangement = in_layout.device_arrangement_origin().array();
  auto output_device_arrangement = out_layout.device_arrangement_origin().array();
  auto in_tensor_map = in_layout.tensor_map_before();
  auto out_tensor_map = out_layout.tensor_map_before();
  if (in_tensor_map != out_tensor_map) {
    MS_LOG(ERROR) << "To apply device matrix transposes, the input and output tensor map must be equal. But got "
                  << in_tensor_map << " and " << out_tensor_map;
    return FAILED;
  }
  if ((input_device_arrangement.size() != output_device_arrangement.size()) ||
      (in_tensor_map.size() != out_tensor_map.size())) {
    MS_LOG(ERROR) << name_ << ": The size of input and output device arrangement and tensor map must be equal.";
    return FAILED;
  }
  auto exchange_axes = CalculateExchangeAxes(axis_v_);
  std::vector<int64_t> expected_device_arrangement(std::begin(input_device_arrangement),
                                                   std::end(input_device_arrangement));
  for (auto exchange_axis : exchange_axes) {
    auto axis_0 = in_tensor_map[LongToUlong(exchange_axis[0])];
    auto axis_1 = in_tensor_map[LongToUlong(exchange_axis[1])];

    Shape correspond_og_tensor_map;
    Shape correspond_target_tensor_map;
    if (axis_0[0] < axis_1[0]) {
      correspond_og_tensor_map = axis_1;
      correspond_target_tensor_map = axis_0;
    } else {
      correspond_og_tensor_map = axis_0;
      correspond_target_tensor_map = axis_1;
    }
    if (!(IsValidTensorMap(correspond_target_tensor_map) && IsValidTensorMap(correspond_og_tensor_map))) {
      MS_LOG(ERROR) << name_ << ": the output tensor layout is not matched, and devicematrix can not be transpose.";
      return FAILED;
    }
    auto og_start_pos = correspond_og_tensor_map[0];
    auto target_start_pos = correspond_target_tensor_map[0];
    Shape temp_og_device_arrangement;
    Shape temp_target_device_arrangement;
    bool has_minus_one = false;
    for (auto idx : correspond_og_tensor_map) {
      if (idx == -1) {
        has_minus_one = true;
        break;
      }
      temp_og_device_arrangement.push_back(expected_device_arrangement[LongToUlong(idx)]);
    }
    for (auto idx : correspond_target_tensor_map) {
      if (idx == -1) {
        has_minus_one = true;
        break;
      }
      temp_target_device_arrangement.push_back(expected_device_arrangement[LongToUlong(idx)]);
    }
    if (!has_minus_one) {
      expected_device_arrangement =
        SwapElement(expected_device_arrangement, og_start_pos, target_start_pos, correspond_target_tensor_map.back(),
                    temp_og_device_arrangement, temp_target_device_arrangement);
    }
  }
  if (expected_device_arrangement != output_device_arrangement) {
    MS_LOG(ERROR) << name_ << ": The output device arrangement is not equal to the expected device arrangement.";
    return FAILED;
  }
  UpdateOutputTensorInfoForInterleaved();
  return SUCCESS;
}

Status TransposeInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor info must be 1, but got " << inputs_tensor_info_.size();
    return FAILED;
  }
  auto in_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  MS_LOG(INFO) << name_ << ": The input tensor layout is " << in_layout.ToString();
  return SUCCESS;
}

TensorLayout TransposeInfo::InferOutputLayout() {
  auto input_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  Shapes tensormap = input_layout.tensor_map_before();
  Shape input_shape = inputs_shape_.at(kIndex0);

  Shapes output_tensormap;
  Shape output_shape;
  for (uint64_t i = 0; i < input_shape.size(); i++) {
    output_shape.push_back(input_shape[LongToUlong(axis_v_[i])]);
    output_tensormap.push_back(tensormap[LongToUlong(axis_v_[i])]);
  }

  TensorLayout output_tensor_layout;
  output_tensor_layout.InitFromExtendVector(input_layout.device_arrangement_origin().array(), output_tensormap,
                                            output_shape);
  MS_LOG(INFO) << name_ << ": The output tensor layout inferred by input tensor layout is "
               << output_tensor_layout.ToString() << ", axis_v_ is " << axis_v_;
  return output_tensor_layout;
}

Status TransposeInfo::ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  Attr output_nums_attr = {"output_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_begin_attrs = {output_nums_attr};
  auto virtual_converter_begin = gen_g.PushBack(
    {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
  std::vector<AnfNodePtr> virtual_converter_end_inputs_vector;
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(virtual_converter_begin, 1)};
  for (int64_t i = 0; i < interleaved_num; ++i) {
    auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin, CreatInt64Imm(i)});
    auto trans_value = CreateTuple(axis_v_);
    auto transpose = gen_g.PushBack({gen_g.NewOpInst(TRANSPOSE), tuple_get_item, trans_value});
    virtual_converter_end_inputs_vector.push_back(transpose);
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

ReplaceGraphPtr TransposeInfo::replace_graph(const CNodePtr &cnode) {
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel()) {
    if (ComputeReplaceGraphForInterleaved(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << " splitting micro interleaved failed.";
    }
    return replace_graph_;
  }
  return nullptr;
}

std::vector<StrategyPtr> TransposeInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategiesForIndependentInputs failed";
  }

  return sp_vector;
}

REGISTER(TransposeInfo);
}  // namespace parallel
}  // namespace mindspore
