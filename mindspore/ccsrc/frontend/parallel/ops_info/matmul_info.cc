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

#include "frontend/parallel/ops_info/matmul_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
void SetDevMatrixShape(const Dimensions &mat_a_strategy, const Dimensions &mat_b_strategy, bool transpose_b,
                       Shape *dev_matrix_shape) {
  MS_EXCEPTION_IF_NULL(dev_matrix_shape);
  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  if (mat_a_size >= mat_b_size) {
    // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)

    // [2],[4] in the example above
    for (size_t i = 0; i < SECOND_FROM_END(mat_a_size); ++i) {
      dev_matrix_shape->push_back(mat_a_strategy.at(i));
    }
  } else {
    // for example: mat_a_strategy:[8,16], mat_b_strategy:[2,4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)

    // [2],[4] in the example above
    for (size_t i = 0; i < SECOND_FROM_END(mat_b_size); ++i) {
      dev_matrix_shape->push_back(mat_b_strategy.at(i));
    }
  }

  // [8],[16] in the example above
  dev_matrix_shape->push_back(mat_a_strategy.at(SECOND_FROM_END(mat_a_size)));
  dev_matrix_shape->push_back(mat_a_strategy.back());

  // [32] in the example above
  if (!transpose_b) {
    dev_matrix_shape->push_back(mat_b_strategy.back());
  } else {
    dev_matrix_shape->push_back(mat_b_strategy.at(SECOND_FROM_END(mat_b_size)));
  }
}

Status MatMulBase::GetAttrs() {
  if (attrs_.size() < MATMUL_ATTRS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of attrs small than 2, got " << attrs_.size();
    return FAILED;
  }

  auto transpose_a_iter = attrs_.find(TRANSPOSE_A);
  if (transpose_a_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(transpose_a_iter->second);
    if (transpose_a_iter->second->isa<BoolImm>()) {
      transpose_a_ = transpose_a_iter->second->cast<BoolImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of transpose_a is not bool.";
      return FAILED;
    }
  }

  if (transpose_a_) {
    MS_LOG(ERROR) << name_ << ": The transpose_a=true is not be supported";
    return FAILED;
  }

  auto transpose_b_iter = attrs_.find(TRANSPOSE_B);
  if (transpose_b_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(transpose_b_iter->second);
    if (transpose_b_iter->second->isa<BoolImm>()) {
      transpose_b_ = transpose_b_iter->second->cast<BoolImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of transpose_b is not bool.";
      return FAILED;
    }
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
  if ((inputs_shape_.size() != MATMUL_INPUTS_SIZE) || (outputs_shape_.size() != MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }
  mat_a_dimension_ = inputs_shape_.at(0).size();
  mat_b_dimension_ = inputs_shape_.at(1).size();
  if (mat_a_dimension_ < 2 || mat_b_dimension_ < 2) {
    MS_LOG(ERROR) << name_ << ": The dim of mat_a or mat_b can not smaller than 2, but the dim of mat_a is "
                  << mat_a_dimension_ << ", the dim of mat_b is " << mat_b_dimension_;
  }

  return SUCCESS;
}

Status CheckRelevantDimension(const Dimensions &long_strategy, const Dimensions &short_strategy) {
  size_t long_size = long_strategy.size();
  size_t short_size = short_strategy.size();
  if (long_size < short_size) {
    MS_LOG(ERROR) << "Size error, the size of long strategy is " << long_size << ", the size of short strategy is "
                  << short_size;
    return FAILED;
  }

  size_t len_diff = long_size - short_size;
  for (size_t j = 0; j < SECOND_FROM_END(short_size); ++j) {
    if (long_strategy.at(len_diff + j) != short_strategy.at(j)) {
      MS_LOG(ERROR) << "Strategies of relevant dimensions are not equal, long strategy is "
                    << ShapeToString(long_strategy) << ", short strategy is " << ShapeToString(short_strategy);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatMul::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions mat_a_strategy = stra.at(0);
  Dimensions mat_b_strategy = stra.at(1);

  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  if ((mat_a_size != mat_a_dimension_) || (mat_b_size != mat_b_dimension_)) {
    MS_LOG(ERROR) << name_ << ": The dimensions of mat_a or mat_b's strategy is wrong.";
    return FAILED;
  }

  // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,16,32]
  // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
  // [16] in the example above
  if (!transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.at(SECOND_FROM_END(mat_b_size)))) {
    MS_LOG(ERROR) << name_ << ": Can not do this operator in the strategy: " << StrategyToString(stra)
                  << ", the transpose_b is false, the shard num of first input's column is " << mat_a_strategy.back()
                  << ", but the shard num of second input's row is " << mat_b_strategy.at(SECOND_FROM_END(mat_b_size));
    return FAILED;
  } else if (transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.back())) {
    MS_LOG(ERROR) << name_ << ": Can not do this operator in the strategy: " << StrategyToString(stra)
                  << ", the transpose_b is true, the shard num of first input's column is " << mat_a_strategy.back()
                  << ", but the shard num of second input's column is " << mat_b_strategy.back();
    return FAILED;
  }

  if (mat_a_size >= mat_b_size) {
    if (CheckRelevantDimension(mat_a_strategy, mat_b_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of relevant dimensions are not equal.";
      return FAILED;
    }
  } else {
    if (CheckRelevantDimension(mat_b_strategy, mat_a_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of relevant dimensions are not equal.";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatMul::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy == nullptr) {
    MS_LOG(INFO) << name_ << ": The output strategy is null";
    return SUCCESS;
  }

  if (mat_a_dimension_ != 2 || mat_b_dimension_ != 2) {
    MS_LOG(ERROR) << name_ << ": The dimension of mat a and mat b must be 2 if set output strategy";
    return FAILED;
  }

  if (CheckStrategyValue(out_strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid output strategy.";
    return FAILED;
  }

  Strategies in_stra = strategy_->GetInputDim();
  Dimensions x_strategy = in_stra.at(0);
  Dimensions w_strategy = in_stra.at(1);

  int64_t in_shard_a = x_strategy[0];
  int64_t in_shard_b = x_strategy[1];
  int64_t in_shard_c = 1;
  if (transpose_b_) {
    in_shard_c = w_strategy[0];
  } else {
    in_shard_c = w_strategy[1];
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

Status MatMulBase::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions mat_a_strategy = stra.at(0);
  Dimensions mat_b_strategy = stra.at(1);

  SetDevMatrixShape(mat_a_strategy, mat_b_strategy, transpose_b_, &dev_matrix_shape_);
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  return SUCCESS;
}

Status MatMulBase::InferForwardCommunication() {
  forward_op_.clear();
  size_t dimension = origin_dev_matrix_shape_.size();
  size_t relevant_dimension_index = SECOND_FROM_END(dimension);
  // Relevant dimension is not split and all reduce is not required,
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  if (origin_dev_matrix_shape_.at(relevant_dimension_index) == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
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

Status MatMulBase::InferTensorMap() {
  size_t size = dev_matrix_shape_.size();
  if (repeated_calc_num_ > 1) {
    // move the first dimension(repeated_calc_num_), just for the convenience of tensor-map's calculation
    size = dev_matrix_shape_.size() - 1;
  }

  Shape tensor_map_index;
  // such as 5: tensor_map_index [4,3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(LAST_INDEX(size) - i));
  }

  // infer output tensor map: [4,3,2,0], delete the second-from-end element
  TensorMap output_tensor_map = tensor_map_index;
  (void)output_tensor_map.erase(output_tensor_map.begin() + static_cast<different_type>(SECOND_FROM_END(size)));

  // infer mat_a tensor map
  // for example: mat_a_dimension is 4, mat_a tensor map:[4,3,2,1]
  TensorMap mat_a_tensor_map = tensor_map_index;
  // delete last one element
  mat_a_tensor_map.pop_back();
  // delete the first (dev_matrix_size - 1 - mat_a_dimension) elements
  (void)mat_a_tensor_map.erase(
    mat_a_tensor_map.begin(),
    mat_a_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(size) - mat_a_dimension_));

  // infer mat_b tensor map
  TensorMap mat_b_tensor_map = tensor_map_index;
  // delete the third-to-last element
  (void)mat_b_tensor_map.erase(mat_b_tensor_map.begin() + static_cast<different_type>(THIRD_FROM_END(size)));
  // delete the first (dev_matrix_size - 1 - mat_b_dimension) elements
  (void)mat_b_tensor_map.erase(
    mat_b_tensor_map.begin(),
    mat_b_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(size) - mat_b_dimension_));
  if (transpose_b_) {
    // swap the last two elements
    int64_t last_value = mat_b_tensor_map.back();
    mat_b_tensor_map.pop_back();
    (void)mat_b_tensor_map.insert(
      mat_b_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(mat_b_tensor_map.size())), last_value);
  }

  if (forward_reduce_scatter_) {
    // the forward reduce scatter only support that the dimension of output is 2
    output_tensor_map = {1, 0};
  }

  inputs_tensor_map_.push_back(mat_a_tensor_map);
  inputs_tensor_map_.push_back(mat_b_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status MatMulBase::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  out_dev_matrix_shape_ = dev_matrix_shape_;
  if (forward_reduce_scatter_) {
    // the reduce scatter mode only use for MatMul
    out_dev_matrix_shape_ = dev_matrix_shape_;
    if (repeated_num_in_dev_matrix_right_ || repeated_calc_num_ == 1) {
      // dev_matrix_shape_ is: [a, b, c, repeat_num] or [a, b, c]
      // out_dev_matrix_shape_ is: [a*b, c, repeat_num] or [a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.begin(), out_dev_matrix_shape_.begin() + 2);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.begin(), dev_matrix_shape_[0] * dev_matrix_shape_[1]);
    } else {
      // dev_matrix_shape_ is: [repeat_num, a, b, c]
      // out_dev_matrix_shape_ is: [repeat_num, a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.begin() + 1, out_dev_matrix_shape_.begin() + 3);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.begin() + 1,
                                         dev_matrix_shape_[1] * dev_matrix_shape_[2]);
    }
  }

  TensorLayout mat_a_layout, mat_b_layout, output_layout;
  if ((mat_a_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) ||
      (mat_b_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[1], inputs_shape_[1]) != SUCCESS) ||
      (output_layout.InitFromVector(out_dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS)) {
    return FAILED;
  }

  if (field_size_ != 0) {
    mat_b_layout.set_field_size(field_size_);
  }

  inputs_layout->push_back(mat_a_layout);
  inputs_layout->push_back(mat_b_layout);
  outputs_layout->push_back(output_layout);
  return SUCCESS;
}

Status MatMulBase::InferTensorInfo() {
  // infer tensor layout
  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }

  TensorLayout mat_a_layout = inputs_layout.at(0);
  TensorLayout mat_b_layout = inputs_layout.at(1);
  TensorLayout output_layout = outputs_layout.at(0);
  TensorInfo mat_a_tensor_info(mat_a_layout);
  TensorInfo mat_b_tensor_info(mat_b_layout);
  TensorInfo output_tensor_info(output_layout);

  inputs_tensor_info_.push_back(mat_a_tensor_info);
  inputs_tensor_info_.push_back(mat_b_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status MatMulBase::SwapLastTwoElements(mindspore::parallel::Shape *const input) {
  if (input->size() < 2) {
    MS_LOG(ERROR) << name_ << ": The size of inputs small than 2.";
    return FAILED;
  }
  auto last_1st_value = input->at(input->size() - 1);
  auto last_2nd_value = input->at(input->size() - 2);
  input->pop_back();
  input->pop_back();
  input->push_back(last_1st_value);
  input->push_back(last_2nd_value);
  return SUCCESS;
}

std::vector<StrategyPtr> MatMulBase::GenerateOpStrategies(int64_t stage_id) {
  Shape mat_a_shape = inputs_shape_[0];
  Shape mat_b_shape = inputs_shape_[1];
  // it is not support transpose_a
  if (transpose_a_) {
    MS_LOG(EXCEPTION) << name_ << ": It's not yet supported transpose_a";
  }
  // it is not support [B, C, D] * [A, B, D, E]
  if (mat_b_shape.size() > mat_a_shape.size()) {
    MS_LOG(EXCEPTION) << name_
                      << ": It's not yet supported that the dim of mat_b larger than the dim of mat_a, but the dim of"
                         " mat_a is "
                      << mat_a_shape.size() << ", the dim of mat_b is " << mat_b_shape.size();
  }
  // it is not support that broadcasts containing 1, such as [A, B, C, D] * [A, 1, D, E]
  size_t diff_len = mat_a_shape.size() - mat_b_shape.size();
  for (size_t i = 0; i < mat_b_shape.size() - 2; ++i) {
    if (mat_b_shape[i] != mat_a_shape[i + diff_len]) {
      MS_LOG(EXCEPTION) << name_ << ": It's not yet supported that broadcasts containing 1, but the shape of mat a is "
                        << mat_a_shape << ", the shape of mat_b is " << mat_b_shape;
    }
  }

  // e.g. mat_a: [A, B, C, D], mat_b: [B, D, E], then to generate the strategy for [A, B, C, D, E]
  std::vector<StrategyPtr> sp_vector;
  Shape splittable_flag(mat_a_shape.size() + 1, 1);
  Shapes splittable_input = {splittable_flag};
  Shape tmp_shape = inputs_shape_[0];
  size_t index = 0;
  if (transpose_b_) {
    index = inputs_shape_[1].size() - 2;
    tmp_shape.push_back(inputs_shape_[1][index]);  // mat_a: [A, B, C, D], mat_b: [B, E, D], tmp_shape: [A, B, C, D, E]
  } else {
    index = inputs_shape_[1].size() - 1;
    tmp_shape.push_back(inputs_shape_[1][index]);  // mat_a: [A, B, C, D], mat_b: [B, D, E], tmp_shape: [A, B, C, D, E]
  }
  Shapes tmp_inputs_shape = {tmp_shape};

  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // set the inputs' strategies
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies replace_strategy;
    Dimensions tmp_strategy = sp->GetInputDim()[0];
    Dimensions mat_a_strategy = tmp_strategy;
    mat_a_strategy.pop_back();

    // mat_b_shape: [B, D, E], tmp_strategy: [A, B, C, D, E]
    // mat_b_strategy: init [A, B, C, D, E]
    Dimensions mat_b_strategy = tmp_strategy;
    // mat_b_strategy: delete C, [A, B, D, E]
    (void)mat_b_strategy.erase(mat_b_strategy.end() - 3);
    // mat_b_strategy: delete A, [B, D, E]
    (void)mat_b_strategy.erase(mat_b_strategy.begin(), mat_b_strategy.begin() + static_cast<different_type>(diff_len));
    // handle transpose_b
    if (transpose_b_) {
      (void)SwapLastTwoElements(&mat_b_strategy);
    }
    replace_strategy.push_back(mat_a_strategy);
    replace_strategy.push_back(mat_b_strategy);
    sp->ResetInputs(replace_strategy);
  }
  return sp_vector;
}

std::shared_ptr<Strategies> BatchMatMulInfo::GenerateBatchStrategies() {
  Dimensions batch_strategy(inputs_shape_[1].size() - 1, 1);
  (void)batch_strategy.insert(batch_strategy.begin(), stage_device_size_);
  Strategies strategy_v = {batch_strategy, batch_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

Status MatMulBase::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Shapes MatMulBase::InferParamStrategy(const Shapes &default_strategy) {
  // handle strategy for matmul to deal with corresponding dimension
  Shape left_matrix_strategy = default_strategy[0];
  Shape right_matrix_strategy = default_strategy[1];
  auto index_a = left_matrix_strategy.size() - 1;
  auto index_b = index_a - 1;

  if (transpose_a_) {
    index_a -= 1;
  }
  if (transpose_b_) {
    index_b += 1;
  }
  if (left_matrix_strategy[index_a] != right_matrix_strategy[index_b]) {
    if (left_matrix_strategy[index_a] == 1) {
      left_matrix_strategy[index_a] = right_matrix_strategy[index_b];
    } else {
      right_matrix_strategy[index_b] = left_matrix_strategy[index_a];
    }
  }

  Shapes ret;
  ret.push_back(left_matrix_strategy);
  ret.push_back(right_matrix_strategy);
  return ret;
}

REGISTER(MatMulInfo);
REGISTER(BatchMatMulInfo);
}  // namespace parallel
}  // namespace mindspore
