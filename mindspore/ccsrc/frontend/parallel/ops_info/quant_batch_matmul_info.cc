/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/quant_batch_matmul_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/core/ops/sequence_ops.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

namespace {
constexpr size_t kQbmmInputX1 = 0;
constexpr size_t kQbmmInputX2 = 1;
constexpr size_t kQbmmInputScale = 2;
constexpr size_t kQbmmInputTransposeX1 = 5;
constexpr size_t kQbmmInputTransposeX2 = 6;
constexpr size_t kQbmmOutput = 0;
constexpr size_t kQbmmInputMinNum = 3;
constexpr size_t kQbmmInputMinSize = 2;
constexpr size_t kQbmmInputNum = 5;
}  // namespace

Shape QuantBatchMatmulInfo::GetCommonShape(const Dimensions &x1_strategy, const Dimensions &x2_strategy) const {
  Shape common_shape;
  size_t x1_size = x1_strategy.size();
  size_t x2_size = x2_strategy.size();
  size_t diff_len = 0;
  if (x1_size >= x2_size) {
    // for example: x1_strategy:[2,1,8,16], x2_strategy:[4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
    // [2],[4] in the example above, support broadcast
    diff_len = x1_size - x2_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(x1_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(x1_size); ++i) {
      if (x1_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(x1_strategy.at(i));
      } else {
        common_shape.push_back(x2_strategy.at(i - diff_len));
      }
    }
  } else {
    // for example: x1_strategy:[4,8,16], x2_strategy:[2,1,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
    // [2],[4] in the example above
    diff_len = x2_size - x1_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(x2_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(x2_size); ++i) {
      if (x2_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(x2_strategy.at(i));
      } else {
        common_shape.push_back(x1_strategy.at(i - diff_len));
      }
    }
  }

  // [8],[16] in the example above
  if (transpose_a_) {
    common_shape.push_back(x1_strategy.back());
    common_shape.push_back(x1_strategy.at(SECOND_FROM_END(x1_size)));
  } else {
    common_shape.push_back(x1_strategy.at(SECOND_FROM_END(x1_size)));
    common_shape.push_back(x1_strategy.back());
  }

  // [32] in the example above
  if (!transpose_b_) {
    common_shape.push_back(x2_strategy.back());
  } else {
    common_shape.push_back(x2_strategy.at(SECOND_FROM_END(x2_size)));
  }
  return common_shape;
}

Status QuantBatchMatmulInfo::GetAttrs() {
  if (attrs_.size() < MATMUL_ATTRS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of attrs small than 2, got " << attrs_.size();
    return FAILED;
  }

  ValuePtr transpose_a_ptr = input_value_.at(kQbmmInputTransposeX1);
  if (transpose_a_ptr != nullptr) {
    transpose_a_ = GetValue<bool>(transpose_a_ptr);
  }

  ValuePtr transpose_b_ptr = input_value_.at(kQbmmInputTransposeX2);
  if (transpose_b_ptr != nullptr) {
    transpose_b_ = GetValue<bool>(transpose_b_ptr);
  }

  auto field_size_iter = attrs_.find(FIELD_SIZE);
  if (field_size_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(field_size_iter->second);
    if (field_size_iter->second->isa<Int64Imm>()) {
      field_size_ = field_size_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of field_size is not int64_t.";
      return FAILED;
    }
  }

  // infer inputs dimension size
  if ((inputs_shape_.size() < kQbmmInputMinNum) || (outputs_shape_.size() != MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, inputs_shape_.size() : "
                  << inputs_shape_.size() << ", outputs_shape_.size() : " << outputs_shape_.size();
    return FAILED;
  }
  x1_dimension_ = inputs_shape_.at(kQbmmInputX1).size();
  x2_dimension_ = inputs_shape_.at(kQbmmInputX2).size();
  if (x1_dimension_ < kQbmmInputMinSize || x2_dimension_ < kQbmmInputMinSize) {
    MS_LOG(ERROR) << name_ << ": The dim of mat_a or mat_b can not smaller than 2, but the dim of mat_a is "
                  << x1_dimension_ << ", the dim of mat_b is " << x2_dimension_;
  }

  MS_LOG(INFO) << name_ << ": the transpose_a is " << transpose_a_ << ", transpose_b is " << transpose_b_;
  return SUCCESS;
}

Status QuantBatchMatmulInfo::CheckBatchDimensions(const Dimensions &long_strategy, const Dimensions &short_strategy) {
  size_t long_size = long_strategy.size();
  size_t short_size = short_strategy.size();
  if (long_size < short_size) {
    MS_LOG(ERROR) << "Size error, the size of long strategy is " << long_size << ", the size of short strategy is "
                  << short_size;
    return FAILED;
  }

  Shape long_shape = inputs_shape_[kQbmmInputX1];
  Shape short_shape = inputs_shape_[kQbmmInputX2];
  if (inputs_shape_[kQbmmInputX1].size() < inputs_shape_[kQbmmInputX2].size()) {
    long_shape = inputs_shape_[kQbmmInputX2];
    short_shape = inputs_shape_[kQbmmInputX1];
  }

  size_t len_diff = long_size - short_size;
  for (size_t j = 0; j < SECOND_FROM_END(short_size); ++j) {
    if (long_strategy.at(len_diff + j) != short_strategy.at(j)) {
      if (long_shape.at(len_diff + j) == 1 || short_shape.at(j) == 1 || long_shape.at(len_diff + j) == -1 ||
          short_shape.at(j) == -1) {
        continue;  // support broadcast, such as: long shape:[A, 1, C, D], short shape:[B, C, D]
      }
      MS_LOG(ERROR) << "Strategies of batch dimensions are not equal, long strategy is " << ShapeToString(long_strategy)
                    << ", short strategy is " << ShapeToString(short_strategy);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status QuantBatchMatmulInfo::CheckInputStrategy(const Shape &x1_strategy, const Shape &x2_strategy) {
  size_t x1_size = x1_strategy.size();
  size_t x2_size = x2_strategy.size();
  if ((x1_size != x1_dimension_) || (x2_size != x2_dimension_)) {
    MS_LOG(ERROR) << name_ << ": The dimensions of mat_a or mat_b's strategy is wrong.";
    return FAILED;
  }

  if (transpose_a_) {
    if (!transpose_b_ && (x1_strategy.at(SECOND_FROM_END(x1_size)) != x2_strategy.at(SECOND_FROM_END(x2_size)))) {
      // for example: x1_strategy:[2,4,16,8], x2_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(x1_strategy) << " and mat_b "
                    << ShapeToString(x2_strategy) << ". The transpose_a is: " << transpose_a_ << ", and transpose_b is "
                    << transpose_b_ << ", the shard num of first input's row is "
                    << x1_strategy.at(SECOND_FROM_END(x1_size)) << ", but the shard num of second input's row is "
                    << x2_strategy.at(SECOND_FROM_END(x2_size));
      return FAILED;
    } else if (transpose_b_ && (x1_strategy.at(SECOND_FROM_END(x1_size)) != x2_strategy.back())) {
      // for example: x1_strategy:[2,4,16,8], x2_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(x1_strategy) << " and mat_b "
                    << ShapeToString(x2_strategy) << ". The transpose_a is: " << transpose_a_ << ", and transpose_b is "
                    << transpose_b_ << ", the shard num of first input's row is "
                    << x1_strategy.at(SECOND_FROM_END(x1_size)) << ", but the shard num of second input's column is "
                    << x2_strategy.back();
      return FAILED;
    }
  } else {
    if (!transpose_b_ && (x1_strategy.back() != x2_strategy.at(SECOND_FROM_END(x2_size)))) {
      // for example: x1_strategy:[2,4,8,16], x2_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(x1_strategy) << " and mat_b "
                    << ShapeToString(x2_strategy) << ". The transpose_a is: " << transpose_a_ << ", and transpose_b is "
                    << transpose_b_ << ", the shard num of first input's column is " << x1_strategy.back()
                    << ", but the shard num of second input's row is " << x2_strategy.at(SECOND_FROM_END(x2_size));
      return FAILED;
    } else if (transpose_b_ && (x1_strategy.back() != x2_strategy.back())) {
      // for example: x1_strategy:[2,4,8,16], x2_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(x1_strategy) << " and mat_b "
                    << ShapeToString(x2_strategy) << ". The transpose_a is: " << transpose_a_ << ", and transpose_b is "
                    << transpose_b_ << ", the shard num of first input's column is " << x1_strategy.back()
                    << ", but the shard num of second input's column is " << x2_strategy.back();
      return FAILED;
    }
  }

  if (x1_size >= x2_size) {
    if (CheckBatchDimensions(x1_strategy, x2_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
      return FAILED;
    }
  } else if (CheckBatchDimensions(x2_strategy, x1_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
    return FAILED;
  }
  return SUCCESS;
}

Status QuantBatchMatmulInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions x1_strategy = stra.at(kQbmmInputX1);
  Dimensions x2_strategy = stra.at(kQbmmInputX2);

  return CheckInputStrategy(x1_strategy, x2_strategy);
}

Status QuantBatchMatmulInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy == nullptr) {
    MS_LOG(INFO) << name_ << ": The output strategy is null";
    return SUCCESS;
  }

  if (x1_dimension_ != kQbmmInputMinSize || x2_dimension_ != kQbmmInputMinSize) {
    MS_LOG(ERROR) << name_ << ": The dimension of mat a and mat b must be 2 if set output strategy";
    return FAILED;
  }

  if (CheckStrategyValue(out_strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid output strategy.";
    return FAILED;
  }

  Strategies in_stra = strategy_->GetInputDim();
  Dimensions x_strategy = in_stra.at(kQbmmInputX1);
  Dimensions w_strategy = in_stra.at(kQbmmInputX2);

  int64_t in_shard_a = x_strategy[0];
  int64_t in_shard_b = x_strategy[1];
  int64_t in_shard_c = w_strategy[1];
  if (transpose_a_) {
    in_shard_a = x_strategy[1];
    in_shard_b = x_strategy[0];
  }

  if (transpose_b_) {
    in_shard_c = w_strategy[0];
  }

  Strategies out_stra = out_strategy->GetInputDim();
  Dimensions output_strategy = out_stra[0];

  int64_t out_shard_a_or_ab = output_strategy[0];
  int64_t out_shard_c = output_strategy[1];
  if (out_shard_c != in_shard_c) {
    MS_LOG(ERROR) << name_ << ": The input strategy is (" << x_strategy << ", " << w_strategy << ")"
                  << ", the second dimension of output strategy must be " << in_shard_c << ", but got " << out_shard_c;
    return FAILED;
  }

  if (out_shard_a_or_ab == in_shard_a) {
    forward_reduce_scatter_ = false;
  } else if (out_shard_a_or_ab == in_shard_a * in_shard_b) {
    forward_reduce_scatter_ = true;
  } else {
    MS_LOG(ERROR) << name_ << ": The input strategy is (" << x_strategy << ", " << w_strategy << ")"
                  << ", the first dimension of output strategy must be " << in_shard_a << " or "
                  << in_shard_a * in_shard_b << ", but got " << out_shard_a_or_ab;
    return FAILED;
  }

  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions x1_strategy = stra.at(kQbmmInputX1);
  Dimensions x2_strategy = stra.at(kQbmmInputX2);

  dev_matrix_shape_ = GetCommonShape(x1_strategy, x2_strategy);
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  MS_LOG(DEBUG) << name_ << ": The dev matrix shape is " << dev_matrix_shape_;
  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferForwardCommunication() {
  if (is_layout_config_) {
    return SUCCESS;
  }
  forward_op_.clear();
  size_t dimension = origin_dev_matrix_shape_.size();
  size_t relevant_dimension_index = SECOND_FROM_END(dimension);
  // Relevant dimension is not split and all reduce is not required,
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  if (origin_dev_matrix_shape_.at(relevant_dimension_index) == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
  }

  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    // if repeated calculation and repeated num in the left of dev matrix, the index of relevant dimension should add 1
    relevant_dimension_index += 1;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(relevant_dimension_index, &group_list) != SUCCESS) {
    ReportError(name_ + ": Infer forward communication, create group failed.");
    return FAILED;
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
  }

  Operator op;
  if (forward_reduce_scatter_) {
    op = CreateReduceScatterOp(REDUCE_OP_SUM, group_list[0].name());
  } else {
    op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  }

  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferOutputTensorMap() {
  Shape x1_map = inputs_tensor_map_[0];
  Shape x2_map = inputs_tensor_map_[1];

  if (transpose_a_ || transpose_b_) {
    MS_LOG(ERROR) << name_ << ": config layout can not support transpose_a or transpose_b";
    return FAILED;
  }

  if (x1_map.size() != x2_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  Shape x1_batch_map(x1_map.cbegin(), x1_map.cbegin() + x1_map.size() - kQbmmInputMinSize);
  Shape x2_batch_map(x2_map.cbegin(), x2_map.cbegin() + x2_map.size() - kQbmmInputMinSize);

  if (x1_batch_map.size() != x2_batch_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  int64_t relevant_dim_map = x1_map[x1_map.size() - 1];
  if (relevant_dim_map != MAP_NONE) {
    MS_LOG(ERROR) << name_ << ": config layout can not support shard relevant dimension";
    return FAILED;
  }

  Shape output_map = x1_batch_map;
  output_map.push_back(x1_map[x1_map.size() - 2]);  // row : x1_map.size() - 2
  output_map.push_back(x2_map[x2_map.size() - 1]);  // col : x2_map.size() - 1
  outputs_tensor_map_.push_back(output_map);
  MS_LOG(INFO) << name_ << ": the input tensor map is " << inputs_tensor_map_ << ", the output tensor map is "
               << output_map;
  return SUCCESS;
}

Status QuantBatchMatmulInfo::CheckLayoutConfig() {
  if (CheckInputStrategy(strategy_from_layout_[0], strategy_from_layout_[1]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": invalid layout config, the dev matrix is " << dev_matrix_shape_
                  << ", and input tensor map is " << inputs_tensor_map_;
    return FAILED;
  }
  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferTensorMap() {
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  size_t size = origin_dev_matrix_shape_.size();

  Shape tensor_map_index;
  // such as 5: tensor_map_index [4,3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(LAST_INDEX(size) - i));
  }

  // infer output tensor map: [4,3,2,0], delete the second-from-end element
  TensorMap output_tensor_map = tensor_map_index;
  (void)output_tensor_map.erase(output_tensor_map.cbegin() + static_cast<different_type>(SECOND_FROM_END(size)));

  // infer mat_a tensor map
  // for example: x1_dimension is 4, mat_a tensor map:[4,3,2,1]
  TensorMap x1_tensor_map = tensor_map_index;
  // delete last one element
  x1_tensor_map.pop_back();
  // delete the first (dev_matrix_size - 1 - x1_dimension) elements
  (void)x1_tensor_map.erase(x1_tensor_map.cbegin(),
                            x1_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - x1_dimension_));
  if (transpose_a_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&x1_tensor_map);
  }

  // infer mat_b tensor map
  TensorMap x2_tensor_map = tensor_map_index;
  // delete the third-to-last element
  (void)x2_tensor_map.erase(x2_tensor_map.cbegin() + static_cast<different_type>(THIRD_FROM_END(size)));
  // delete the first (dev_matrix_size - 1 - x2_dimension) elements
  (void)x2_tensor_map.erase(x2_tensor_map.cbegin(),
                            x2_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - x2_dimension_));
  if (transpose_b_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&x2_tensor_map);
  }

  if (forward_reduce_scatter_) {
    // the forward reduce scatter only support that the dimension of output is 2
    output_tensor_map = {1, 0};
  }

  // handle broadcast
  for (size_t i = 0; i < x1_tensor_map.size(); ++i) {
    if (inputs_shape_[kQbmmInputX1][i] == 1) {
      x1_tensor_map[i] = MAP_NONE;
    }
  }

  for (size_t j = 0; j < x2_tensor_map.size(); ++j) {
    if (inputs_shape_[kQbmmInputX2][j] == 1) {
      x2_tensor_map[j] = MAP_NONE;
    }
  }

  inputs_tensor_map_.push_back(x1_tensor_map);
  inputs_tensor_map_.push_back(x2_tensor_map);

  for (size_t i = kQbmmInputScale; i < inputs_shape_.size(); i++) {
    TensorMap tensor_map = {0};
    inputs_tensor_map_.push_back(tensor_map);
  }
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(DEBUG) << name_ << ": The mat_a's tensor map is " << x1_tensor_map << ", the mat_b's tensor map is "
                << x2_tensor_map << ", the output's tensor map is " << output_tensor_map;
  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  out_dev_matrix_shape_ = dev_matrix_shape_;
  if (forward_reduce_scatter_) {
    // the reduce scatter mode only use for QuantBatchMatmulInfo
    if (repeated_num_in_dev_matrix_right_ || repeated_calc_num_ == 1) {
      // dev_matrix_shape_ is: [a, b, c, repeat_num] or [a, b, c]
      // out_dev_matrix_shape_ is: [a*b, c, repeat_num] or [a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin(),
                                        out_dev_matrix_shape_.cbegin() + kQbmmInputMinSize);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin(), dev_matrix_shape_[0] * dev_matrix_shape_[1]);
    } else {
      // dev_matrix_shape_ is: [repeat_num, a, b, c]
      // out_dev_matrix_shape_ is: [repeat_num, a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin() + 1,
                                        out_dev_matrix_shape_.cbegin() + kQbmmInputMinNum);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin() + 1,
                                         dev_matrix_shape_[1] * dev_matrix_shape_[2]);  // 2 : b
    }
  }

  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    TensorLayout tensor_layout;
    if ((tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS)) {
      return FAILED;
    }

    if (i == kQbmmInputX2 && field_size_ != 0) {
      tensor_layout.set_field_size(field_size_);
    }
    inputs_layout->push_back(tensor_layout);
  }

  TensorLayout output_layout;
  if (output_layout.InitFromVector(out_dev_matrix_shape_, outputs_tensor_map_[kQbmmOutput],
                                   outputs_shape_[kQbmmOutput]) != SUCCESS) {
    return FAILED;
  }
  outputs_layout->push_back(output_layout);
  return SUCCESS;
}

Status QuantBatchMatmulInfo::InferTensorInfo() {
  // infer tensor layout
  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }

  for (size_t i = 0; i < inputs_layout.size(); i++) {
    TensorLayout tensor_layout = inputs_layout.at(i);
    TensorInfo tensor_info(tensor_layout);
    inputs_tensor_info_.push_back(tensor_info);
  }

  TensorLayout output_layout = outputs_layout.at(kQbmmOutput);
  TensorInfo output_tensor_info(output_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status QuantBatchMatmulInfo::SwapLastTwoElements(mindspore::parallel::Shape *const input) {
  if (input->size() < kQbmmInputMinSize) {
    MS_LOG(ERROR) << name_ << ": The size of inputs small than 2.";
    return FAILED;
  }
  auto last_1st_value = input->at(input->size() - 1);  // 1 : last 1nd
  auto last_2nd_value = input->at(input->size() - 2);  // 2 : last 2nd
  input->pop_back();
  input->pop_back();
  input->push_back(last_1st_value);
  input->push_back(last_2nd_value);
  return SUCCESS;
}

std::vector<StrategyPtr> QuantBatchMatmulInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape x1_shape = inputs_shape_[kQbmmInputX1];
  Shape x2_shape = inputs_shape_[kQbmmInputX2];

  // e.g. mat_a: [A, B, C, D], mat_b: [B, D, E], then to generate the strategy for [A, B, C, D, E]
  // e.g. mat_a: [B, C, D], mat_b: [A, 1, D, E], then to generate the strategy for [A, B, C, D, E]
  size_t long_shape_size = x1_shape.size();
  if (x1_shape.size() < x2_shape.size()) {
    long_shape_size = x2_shape.size();
  }
  std::vector<StrategyPtr> sp_vector;
  Shape splittable_flag(long_shape_size + 1, 1);
  Shapes splittable_input = {splittable_flag};
  Shape tmp_shape = GetCommonShape(x1_shape, x2_shape);
  Shapes tmp_inputs_shape = {tmp_shape};

  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // set the inputs' strategies
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }

    // handle mat_a's strategy
    Dimensions x1_strategy = sp->GetInputDim()[0];  // [A, B, C, D, E]
    x1_strategy.pop_back();                         // [A, B, C, D]
    if (x1_shape.size() < x2_shape.size()) {
      auto diff_len = x2_shape.size() - x1_shape.size();
      (void)x1_strategy.erase(x1_strategy.cbegin(), x1_strategy.cbegin() + static_cast<different_type>(diff_len));
    }

    // transpose_a
    if (transpose_a_) {
      (void)SwapLastTwoElements(&x1_strategy);
    }

    // broadcast
    for (size_t i = 0; i < x1_strategy.size(); ++i) {
      if (x1_shape[i] <= 1) {
        x1_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    // handle mat_b's strategy
    Dimensions x2_strategy = sp->GetInputDim()[0];  // [A, B, C, D, E]
    // x2_strategy: delete C, [A, B, D, E]
    (void)x2_strategy.erase(x2_strategy.cend() - kQbmmInputMinNum);
    // x2_strategy: delete A, [B, D, E]

    if (x2_shape.size() < x1_shape.size()) {
      auto diff_len = x1_shape.size() - x2_shape.size();
      (void)x2_strategy.erase(x2_strategy.cbegin(), x2_strategy.cbegin() + static_cast<different_type>(diff_len));
    }

    // handle transpose_b
    if (transpose_b_) {
      (void)SwapLastTwoElements(&x2_strategy);
    }

    // broadcast
    for (size_t i = 0; i < x2_strategy.size(); ++i) {
      if (x2_shape[i] <= 1) {
        x2_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    Strategies replace_strategy{x1_strategy, x2_strategy};
    sp->ResetInputs(replace_strategy);
  }
  return sp_vector;
}

std::shared_ptr<Strategies> QuantBatchMatmulInfo::GenerateBatchStrategies() {
  Dimensions batch_strategy_x1(inputs_shape_[kQbmmInputX1].size(), 1);
  Dimensions batch_strategy_x2(inputs_shape_[kQbmmInputX2].size(), 1);

  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);
  Strategies strategy_v;
  // input's shape equals to weight's shape
  if (inputs_shape_[kQbmmInputX1].size() == inputs_shape_[kQbmmInputX2].size()) {
    batch_strategy_x1[0] = stage_device_size_;
    if (inputs_shape_[kQbmmInputX1].size() > MATMUL_INPUTS_SIZE) {
      batch_strategy_x2[0] = stage_device_size_;
    }
  } else if (inputs_shape_[kQbmmInputX1].size() > inputs_shape_[kQbmmInputX2].size()) {
    batch_strategy_x1[0] = stage_device_size_;
  } else {
    batch_strategy_x2[0] = stage_device_size_;
  }
  strategy_v.emplace_back(batch_strategy_x1);
  strategy_v.emplace_back(batch_strategy_x2);

  for (size_t i = kQbmmInputScale; i < inputs_shape_.size(); i++) {
    Dimensions strategy_dimensions(inputs_shape_[i].size(), 1);
    strategy_v.emplace_back(strategy_dimensions);
  }
  return std::make_shared<Strategies>(strategy_v);
}

Status QuantBatchMatmulInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}
REGISTER(QuantBatchMatmulInfo);
}  // namespace parallel
}  // namespace mindspore
