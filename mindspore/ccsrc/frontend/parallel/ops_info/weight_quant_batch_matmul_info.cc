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

#include "frontend/parallel/ops_info/weight_quant_batch_matmul_info.h"

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
constexpr size_t kMatSize = 2;
constexpr size_t kInputX = 0;
constexpr size_t kInputWeight = 1;
constexpr size_t kInputAntiquantScale = 2;
constexpr size_t kInputTransposeX = 7;
constexpr size_t kInputTransposeWeight = 8;
constexpr size_t kInputGroupSize = 9;
constexpr size_t kOutput = 0;
constexpr size_t kInputLeastNum = 3;
}  // namespace

Shape WeightQuantBatchMatmulInfo::GetCommonShape(const Dimensions &mat_x_strategy,
                                                 const Dimensions &mat_weight_strategy) const {
  Shape common_shape;
  size_t mat_x_size = mat_x_strategy.size();
  size_t mat_weight_size = mat_weight_strategy.size();
  size_t diff_len = 0;
  if (mat_x_size >= mat_weight_size) {
    // for example: mat_x_strategy:[2,1,8,16], mat_weight_strategy:[4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_weight is false)
    // [2],[4] in the example above, support broadcast
    diff_len = mat_x_size - mat_weight_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(mat_x_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(mat_x_size); ++i) {
      if (mat_x_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(mat_x_strategy.at(i));
      } else {
        common_shape.push_back(mat_weight_strategy.at(i - diff_len));
      }
    }
  } else {
    // for example: mat_x_strategy:[4,8,16], mat_weight_strategy:[2,1,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_weight is false)
    // [2],[4] in the example above
    diff_len = mat_weight_size - mat_x_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(mat_weight_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(mat_weight_size); ++i) {
      if (mat_weight_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(mat_weight_strategy.at(i));
      } else {
        common_shape.push_back(mat_x_strategy.at(i - diff_len));
      }
    }
  }

  // [8],[16] in the example above
  if (transpose_x_) {
    common_shape.push_back(mat_x_strategy.back());
    common_shape.push_back(mat_x_strategy.at(SECOND_FROM_END(mat_x_size)));
  } else {
    common_shape.push_back(mat_x_strategy.at(SECOND_FROM_END(mat_x_size)));
    common_shape.push_back(mat_x_strategy.back());
  }

  // [32] in the example above
  if (!transpose_weight_) {
    common_shape.push_back(mat_weight_strategy.back());
  } else {
    common_shape.push_back(mat_weight_strategy.at(SECOND_FROM_END(mat_weight_size)));
  }
  return common_shape;
}

Status WeightQuantBatchMatmulInfo::GetAttrs() {
  ValuePtr transpose_x_ptr = input_value_.at(kInputTransposeX);
  if (transpose_x_ptr != nullptr) {
    transpose_x_ = GetValue<bool>(transpose_x_ptr);
  }

  ValuePtr transpose_weight_ptr = input_value_.at(kInputTransposeWeight);
  if (transpose_weight_ptr != nullptr) {
    transpose_weight_ = GetValue<bool>(transpose_weight_ptr);
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
  if ((inputs_shape_.size() < kInputLeastNum) || (outputs_shape_.size() != MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, inputs_shape_.size() : "
                  << inputs_shape_.size() << ", outputs_shape_.size() : " << outputs_shape_.size();
    return FAILED;
  }
  mat_x_dimension_ = inputs_shape_.at(kInputX).size();
  mat_weight_dimension_ = inputs_shape_.at(kInputWeight).size();
  if (mat_x_dimension_ < 2 || mat_weight_dimension_ < 2) {  // matmul min shapesize is 2
    MS_LOG(ERROR) << name_ << ": The dim of mat_x or mat_weight can not smaller than 2, but the dim of mat_x is "
                  << mat_x_dimension_ << ", the dim of mat_weight is " << mat_weight_dimension_;
  }

  MS_LOG(INFO) << name_ << ": the transpose_x is " << transpose_x_ << ", transpose_weight is " << transpose_weight_;
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::CheckBatchDimensions(const Dimensions &long_strategy,
                                                        const Dimensions &short_strategy) {
  size_t long_size = long_strategy.size();
  size_t short_size = short_strategy.size();
  if (long_size < short_size) {
    MS_LOG(ERROR) << "Size error, the size of long strategy is " << long_size << ", the size of short strategy is "
                  << short_size;
    return FAILED;
  }

  Shape long_shape = inputs_shape_[kInputX];
  Shape short_shape = inputs_shape_[kInputWeight];
  if (inputs_shape_[kInputX].size() < inputs_shape_[kInputWeight].size()) {
    long_shape = inputs_shape_[kInputWeight];
    short_shape = inputs_shape_[kInputX];
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

Status WeightQuantBatchMatmulInfo::CheckInputStrategy(const Shape &mat_x_strategy, const Shape &mat_weight_strategy) {
  size_t mat_x_size = mat_x_strategy.size();
  size_t mat_weight_size = mat_weight_strategy.size();
  if ((mat_x_size != mat_x_dimension_) || (mat_weight_size != mat_weight_dimension_)) {
    MS_LOG(ERROR) << name_ << ": The dimensions of mat_x or mat_weight's strategy is wrong.";
    return FAILED;
  }
  auto x_k_col = mat_x_strategy.at(SECOND_FROM_END(mat_x_size));
  auto weight_k_col = mat_weight_strategy.at(SECOND_FROM_END(mat_weight_size));
  std::stringstream error_common_info;
  error_common_info << name_ << ": Invalid strategy for mat_x " << ShapeToString(mat_x_strategy) << " and mat_weight "
                    << ShapeToString(mat_weight_strategy) << ". The trans_x is: " << transpose_x_;
  if (transpose_x_) {
    error_common_info << ", and trans_weight is " << transpose_weight_ << ", the x's k dim is " << x_k_col;
    if (!transpose_weight_ && (x_k_col != weight_k_col)) {
      // for example: mat_x_strategy:[2,4,16,8], mat_weight_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << error_common_info.str() << ", but the weight's k dim is " << weight_k_col;
      return FAILED;
    } else if (transpose_weight_ && (x_k_col != mat_weight_strategy.back())) {
      // for example: mat_x_strategy:[2,4,16,8], mat_weight_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << error_common_info.str() << ", but the weight's k dim is " << mat_weight_strategy.back();
      return FAILED;
    }
  } else {
    error_common_info << ", and trans_weight is " << transpose_weight_ << ", the x's k dim is "
                      << mat_x_strategy.back();
    if (!transpose_weight_ && (mat_x_strategy.back() != weight_k_col)) {
      // for example: mat_x_strategy:[2,4,8,16], mat_weight_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << error_common_info.str() << ", but the weight's k dim is " << weight_k_col;
      return FAILED;
    } else if (transpose_weight_ && (mat_x_strategy.back() != mat_weight_strategy.back())) {
      // for example: mat_x_strategy:[2,4,8,16], mat_weight_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << error_common_info.str() << ", but the weight's k dim is " << mat_weight_strategy.back();
      return FAILED;
    }
  }

  if (mat_x_size >= mat_weight_size) {
    if (CheckBatchDimensions(mat_x_strategy, mat_weight_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
      return FAILED;
    }
  } else {
    if (CheckBatchDimensions(mat_weight_strategy, mat_x_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions mat_x_strategy = stra.at(kInputX);
  Dimensions mat_weight_strategy = stra.at(kInputWeight);

  return CheckInputStrategy(mat_x_strategy, mat_weight_strategy);
}

Status WeightQuantBatchMatmulInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy == nullptr) {
    MS_LOG(INFO) << name_ << ": The output strategy is null";
    return SUCCESS;
  }

  if (mat_x_dimension_ != kMatSize || mat_weight_dimension_ != kMatSize) {
    MS_LOG(ERROR) << name_ << ": The dimension of mat a and mat b must be 2 if set output strategy";
    return FAILED;
  }

  if (CheckStrategyValue(out_strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid output strategy.";
    return FAILED;
  }

  Strategies in_stra = strategy_->GetInputDim();
  Dimensions x_strategy = in_stra.at(kInputX);
  Dimensions w_strategy = in_stra.at(kInputWeight);

  int64_t in_shard_a = x_strategy[0];
  int64_t in_shard_b = x_strategy[1];
  int64_t in_shard_c = w_strategy[1];
  if (transpose_x_) {
    in_shard_a = x_strategy[1];
    in_shard_b = x_strategy[0];
  }

  if (transpose_weight_) {
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

Status WeightQuantBatchMatmulInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions mat_x_strategy = stra.at(kInputX);
  Dimensions mat_weight_strategy = stra.at(kInputWeight);

  dev_matrix_shape_ = GetCommonShape(mat_x_strategy, mat_weight_strategy);
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  MS_LOG(DEBUG) << name_ << ": The dev matrix shape is " << dev_matrix_shape_;
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::InferForwardCommunication() {
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

Status WeightQuantBatchMatmulInfo::InferOutputTensorMap() {
  Shape mat_x_map = inputs_tensor_map_[0];
  Shape mat_weight_map = inputs_tensor_map_[1];

  if (transpose_x_ || transpose_weight_) {
    MS_LOG(ERROR) << name_ << ": config layout can not support transpose_x or transpose_weight";
    return FAILED;
  }

  if (mat_x_map.size() != mat_weight_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  Shape mat_x_batch_map(mat_x_map.cbegin(), mat_x_map.cbegin() + mat_x_map.size() - kMatSize);
  Shape mat_weight_batch_map(mat_weight_map.cbegin(), mat_weight_map.cbegin() + mat_weight_map.size() - kMatSize);

  if (mat_x_batch_map.size() != mat_weight_batch_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  int64_t relevant_dim_map = mat_x_map[mat_x_map.size() - 1];
  if (relevant_dim_map != MAP_NONE) {
    MS_LOG(ERROR) << name_ << ": config layout can not support shard relevant dimension";
    return FAILED;
  }

  Shape output_map = mat_x_batch_map;
  output_map.push_back(mat_x_map[mat_x_map.size() - kMatSize]);
  output_map.push_back(mat_weight_map[mat_weight_map.size() - 1]);
  outputs_tensor_map_.push_back(output_map);
  MS_LOG(INFO) << name_ << ": the input tensor map is " << inputs_tensor_map_ << ", the output tensor map is "
               << output_map;
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::CheckLayoutConfig() {
  if (CheckInputStrategy(strategy_from_layout_[0], strategy_from_layout_[1]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": invalid layout config, the dev matrix is " << dev_matrix_shape_
                  << ", and input tensor map is " << inputs_tensor_map_;
    return FAILED;
  }
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::InferTensorMap() {
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

  // infer mat_x tensor map
  // for example: mat_x_dimension is 4, mat_x tensor map:[4,3,2,1]
  TensorMap mat_x_tensor_map = tensor_map_index;
  // delete last one element
  mat_x_tensor_map.pop_back();
  // delete the first (dev_matrix_size - 1 - mat_x_dimension) elements
  (void)mat_x_tensor_map.erase(
    mat_x_tensor_map.cbegin(),
    mat_x_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - mat_x_dimension_));
  if (transpose_x_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&mat_x_tensor_map);
  }

  // infer mat_weight tensor map
  TensorMap mat_weight_tensor_map = tensor_map_index;
  // delete the third-to-last element
  (void)mat_weight_tensor_map.erase(mat_weight_tensor_map.cbegin() + static_cast<different_type>(THIRD_FROM_END(size)));
  // delete the first (dev_matrix_size - 1 - mat_weight_dimension) elements
  (void)mat_weight_tensor_map.erase(
    mat_weight_tensor_map.cbegin(),
    mat_weight_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - mat_weight_dimension_));
  if (transpose_weight_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&mat_weight_tensor_map);
  }

  if (forward_reduce_scatter_) {
    // the forward reduce scatter only support that the dimension of output is 2
    output_tensor_map = {1, 0};
  }

  // handle broadcast
  for (size_t i = 0; i < mat_x_tensor_map.size(); ++i) {
    if (inputs_shape_[kInputX][i] == 1) {
      mat_x_tensor_map[i] = MAP_NONE;
    }
  }

  for (size_t j = 0; j < mat_weight_tensor_map.size(); ++j) {
    if (inputs_shape_[kInputWeight][j] == 1) {
      mat_weight_tensor_map[j] = MAP_NONE;
    }
  }

  inputs_tensor_map_.push_back(mat_x_tensor_map);
  inputs_tensor_map_.push_back(mat_weight_tensor_map);

  for (size_t i = kInputAntiquantScale; i < inputs_shape_.size(); i++) {
    inputs_tensor_map_.push_back({0});
  }
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(DEBUG) << name_ << ": The mat_x's tensor map is " << mat_x_tensor_map << ", the mat_weight's tensor map is "
                << mat_weight_tensor_map << ", the output's tensor map is " << output_tensor_map;
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  out_dev_matrix_shape_ = dev_matrix_shape_;
  if (forward_reduce_scatter_) {
    // the reduce scatter mode only use for WeightQuantBatchMatmulInfo
    if (repeated_num_in_dev_matrix_right_ || repeated_calc_num_ == 1) {
      // dev_matrix_shape_ is: [a, b, c, repeat_num] or [a, b, c]
      // out_dev_matrix_shape_ is: [a*b, c, repeat_num] or [a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin(), out_dev_matrix_shape_.cbegin() + kMatSize);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin(), dev_matrix_shape_[0] * dev_matrix_shape_[1]);
    } else {
      // dev_matrix_shape_ is: [repeat_num, a, b, c]
      // out_dev_matrix_shape_ is: [repeat_num, a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin() + 1,
                                        out_dev_matrix_shape_.cbegin() + kInputLeastNum);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin() + 1,
                                         dev_matrix_shape_[1] * dev_matrix_shape_[2]);  // 1:(a) * 2:(b)
    }
  }

  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    TensorLayout tensor_layout;
    if (tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      return FAILED;
    }
    inputs_layout->push_back(tensor_layout);
  }

  TensorLayout output_layout;
  if ((output_layout.InitFromVector(out_dev_matrix_shape_, outputs_tensor_map_[kOutput], outputs_shape_[kOutput]) !=
       SUCCESS)) {
    return FAILED;
  }
  outputs_layout->push_back(output_layout);

  return SUCCESS;
}  // namespace parallel

Status WeightQuantBatchMatmulInfo::InferTensorInfo() {
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

  TensorLayout output_layout = outputs_layout.at(kOutput);
  TensorInfo output_tensor_info(output_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status WeightQuantBatchMatmulInfo::SwapLastTwoElements(mindspore::parallel::Shape *const input) {
  if (input->size() < kMatSize) {
    MS_LOG(ERROR) << name_ << ": The size of inputs small than 2.";
    return FAILED;
  }
  auto last_1st_value = input->at(input->size() - 1);  // last_1st_value
  auto last_2nd_value = input->at(input->size() - 2);  // last_2nd_value
  input->pop_back();
  input->pop_back();
  input->push_back(last_1st_value);
  input->push_back(last_2nd_value);
  return SUCCESS;
}

std::vector<StrategyPtr> WeightQuantBatchMatmulInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape mat_x_shape = inputs_shape_[kInputX];
  Shape mat_weight_shape = inputs_shape_[kInputWeight];

  // e.g. mat_x: [A, B, C, D], mat_weight: [B, D, E], then to generate the strategy for [A, B, C, D, E]
  // e.g. mat_x: [B, C, D], mat_weight: [A, 1, D, E], then to generate the strategy for [A, B, C, D, E]
  size_t long_shape_size = mat_x_shape.size();
  if (mat_x_shape.size() < mat_weight_shape.size()) {
    long_shape_size = mat_weight_shape.size();
  }
  std::vector<StrategyPtr> sp_vector;
  Shape splittable_flag(long_shape_size + 1, 1);
  Shapes splittable_input = {splittable_flag};
  Shape tmp_shape = GetCommonShape(mat_x_shape, mat_weight_shape);
  Shapes tmp_inputs_shape = {tmp_shape};

  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // set the inputs' strategies
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }

    // handle mat_x's strategy
    Dimensions x_strategy = sp->GetInputDim()[0];  // [A, B, C, D, E]
    x_strategy.pop_back();                         // [A, B, C, D]
    if (mat_x_shape.size() < mat_weight_shape.size()) {
      auto diff_len = mat_weight_shape.size() - mat_x_shape.size();
      (void)x_strategy.erase(x_strategy.cbegin(), x_strategy.cbegin() + static_cast<different_type>(diff_len));
    }

    // transpose_x
    if (transpose_x_) {
      (void)SwapLastTwoElements(&x_strategy);
    }

    // broadcast
    for (size_t i = 0; i < x_strategy.size(); ++i) {
      if (mat_x_shape[i] <= 1) {
        x_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    // handle mat_weight's strategy
    Dimensions w_strategy = sp->GetInputDim()[0];  // [A, B, C, D, E]
    // w_strategy: delete C, [A, B, D, E]
    (void)w_strategy.erase(w_strategy.cend() - 3);  // C 3
    // w_strategy: delete A, [B, D, E]

    if (mat_weight_shape.size() < mat_x_shape.size()) {
      auto diff_len = mat_x_shape.size() - mat_weight_shape.size();
      (void)w_strategy.erase(w_strategy.cbegin(), w_strategy.cbegin() + static_cast<different_type>(diff_len));
    }

    // handle transpose_weight
    if (transpose_weight_) {
      (void)SwapLastTwoElements(&w_strategy);
    }

    // broadcast
    for (size_t i = 0; i < w_strategy.size(); ++i) {
      if (mat_weight_shape[i] <= 1) {
        w_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    Strategies replace_strategy{x_strategy, w_strategy};
    sp->ResetInputs(replace_strategy);
  }
  return sp_vector;
}

std::shared_ptr<Strategies> WeightQuantBatchMatmulInfo::GenerateBatchStrategies() {
  Dimensions batch_strategy_x(inputs_shape_[kInputX].size(), 1);
  Dimensions batch_strategy_weight(inputs_shape_[kInputWeight].size(), 1);

  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);

  // input's shape equals to weight's shape
  if (inputs_shape_[kInputX].size() == inputs_shape_[kInputWeight].size()) {
    batch_strategy_x[0] = stage_device_size_;
    if (inputs_shape_[kInputX].size() > MATMUL_INPUTS_SIZE) {
      batch_strategy_weight[0] = stage_device_size_;
    }
  }
  if (inputs_shape_[kInputX].size() > inputs_shape_[kInputWeight].size()) {
    batch_strategy_x[0] = stage_device_size_;
  }
  if (inputs_shape_[kInputX].size() < inputs_shape_[kInputWeight].size()) {
    batch_strategy_weight[0] = stage_device_size_;
  }
  Strategies strategy_v;
  strategy_v.emplace_back(batch_strategy_x);
  strategy_v.emplace_back(batch_strategy_weight);
  for (size_t i = kInputAntiquantScale; i < inputs_shape_.size(); i++) {
    Dimensions dims(inputs_shape_[i].size(), 1);
    strategy_v.emplace_back(dims);
  }
  return std::make_shared<Strategies>(strategy_v);
}

Status WeightQuantBatchMatmulInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}
REGISTER(WeightQuantBatchMatmulInfo);
}  // namespace parallel
}  // namespace mindspore
