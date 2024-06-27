/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
Shape MatMulBase::GetCommonShape(const Dimensions &mat_a_strategy, const Dimensions &mat_b_strategy) const {
  Shape common_shape;
  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  size_t diff_len = 0;
  if (mat_a_size >= mat_b_size) {
    // for example: mat_a_strategy:[2,1,8,16], mat_b_strategy:[4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
    // [2],[4] in the example above, support broadcast
    diff_len = mat_a_size - mat_b_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(mat_a_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(mat_a_size); ++i) {
      if (mat_a_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(mat_a_strategy.at(i));
      } else {
        common_shape.push_back(mat_b_strategy.at(i - diff_len));
      }
    }
  } else {
    // for example: mat_a_strategy:[4,8,16], mat_b_strategy:[2,1,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
    // [2],[4] in the example above
    diff_len = mat_b_size - mat_a_size;
    for (size_t i = 0; i < diff_len; ++i) {
      common_shape.push_back(mat_b_strategy.at(i));  // [2] in the example
    }

    for (size_t i = diff_len; i < SECOND_FROM_END(mat_b_size); ++i) {
      if (mat_b_strategy.at(i) != NO_SPLIT_STRATEGY) {
        common_shape.push_back(mat_b_strategy.at(i));
      } else {
        common_shape.push_back(mat_a_strategy.at(i - diff_len));
      }
    }
  }

  // [8],[16] in the example above
  if (transpose_a_) {
    common_shape.push_back(mat_a_strategy.back());
    common_shape.push_back(mat_a_strategy.at(SECOND_FROM_END(mat_a_size)));
  } else {
    common_shape.push_back(mat_a_strategy.at(SECOND_FROM_END(mat_a_size)));
    common_shape.push_back(mat_a_strategy.back());
  }

  // [32] in the example above
  if (!transpose_b_) {
    common_shape.push_back(mat_b_strategy.back());
  } else {
    common_shape.push_back(mat_b_strategy.at(SECOND_FROM_END(mat_b_size)));
  }
  return common_shape;
}

Status MatMulBase::GetAttrs() {
  auto enable_nd_tp_iter = attrs_.find(ENABLE_ND_TP);
  if (enable_nd_tp_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_nd_tp_iter->second);
    if (enable_nd_tp_iter->second->isa<BoolImm>()) {
      enable_nd_tp_ = enable_nd_tp_iter->second->cast<BoolImmPtr>()->value();
      MS_LOG(INFO) << "enable_nd_tp_: " << enable_nd_tp_;
    } else {
      MS_LOG(ERROR) << name_ << ": The value of enable_nd_tp is not bool.";
      return FAILED;
    }
  }

  auto transpose_a_value = GetScalarValueFromInputsWithCheck<bool>(input_value_, name_, TRANSPOSE_A);
  if (!transpose_a_value.has_value()) {
    return FAILED;
  }
  transpose_a_ = transpose_a_value.value();
  auto transpose_b_value = GetScalarValueFromInputsWithCheck<bool>(input_value_, name_, TRANSPOSE_B);
  if (!transpose_b_value.has_value()) {
    return FAILED;
  }
  transpose_b_ = transpose_b_value.value();

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
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": the transpose_a is " << transpose_a_ << ", transpose_b is " << transpose_b_;
  return SUCCESS;
}

Status MatMulExtInfo::GetAttrs() {
  transpose_a_ = false;
  transpose_b_ = false;

  if ((inputs_shape_.size() != MATMUL_INPUTS_SIZE) || (outputs_shape_.size() != MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }
  mat_a_dimension_ = inputs_shape_.at(0).size();
  mat_b_dimension_ = inputs_shape_.at(1).size();
  constexpr size_t kMinDim = 2;
  if (mat_a_dimension_ < kMinDim || mat_b_dimension_ < kMinDim) {
    MS_LOG(ERROR) << name_ << ": The dim of mat_a or mat_b can not smaller than 2, but the dim of mat_a is "
                  << mat_a_dimension_ << ", the dim of mat_b is " << mat_b_dimension_;
    return FAILED;
  }

  return SUCCESS;
}

Status MatMulBase::CheckBatchDimensions(const Dimensions &long_strategy, const Dimensions &short_strategy) {
  size_t long_size = long_strategy.size();
  size_t short_size = short_strategy.size();
  if (long_size < short_size) {
    MS_LOG(ERROR) << "Size error, the size of long strategy is " << long_size << ", the size of short strategy is "
                  << short_size;
    return FAILED;
  }

  Shape long_shape = inputs_shape_[0];
  Shape short_shape = inputs_shape_[1];
  if (inputs_shape_[0].size() < inputs_shape_[1].size()) {
    long_shape = inputs_shape_[1];
    short_shape = inputs_shape_[0];
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

void MatMul::CheckPCLMatMul(const Shape &mat_a_strategy, const Shape &mat_b_strategy) {
  candidate_flag_ = false;
  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  int64_t mat_a_device = std::accumulate(mat_a_strategy.begin(), mat_a_strategy.end(), 1, std::multiplies<int64_t>());
  if (mat_a_size == mat_b_size && !transpose_b_ && mat_a_size == MATMUL_DIM && mat_a_strategy == mat_b_strategy &&
      mat_a_device == stage_device_size_ && !transpose_a_) {
    candidate_flag_ = True;
  }
}

Status MatMul::CheckInputStrategy(const Shape &mat_a_strategy, const Shape &mat_b_strategy) {
  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  if ((mat_a_size != mat_a_dimension_) || (mat_b_size != mat_b_dimension_)) {
    MS_LOG(ERROR) << name_ << ": The dimensions of mat_a or mat_b's strategy is wrong.";
    return FAILED;
  }

  // PCL MatMul: mat_a_strategy and mat_b_strategy are [a, b], and use replace graph to handle it
  CheckPCLMatMul(mat_a_strategy, mat_b_strategy);
  if (candidate_flag_) {
    MS_LOG(INFO) << name_ << ": Using PCL MatMul.";
    return SUCCESS;
  }

  if (transpose_a_) {
    if (!transpose_b_ &&
        (mat_a_strategy.at(SECOND_FROM_END(mat_a_size)) != mat_b_strategy.at(SECOND_FROM_END(mat_b_size)))) {
      // for example: mat_a_strategy:[2,4,16,8], mat_b_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(mat_a_strategy) << " and mat_b "
                    << ShapeToString(mat_b_strategy) << ". The transpose_a is: " << transpose_a_
                    << ", and transpose_b is " << transpose_b_ << ", the shard num of first input's row is "
                    << mat_a_strategy.at(SECOND_FROM_END(mat_a_size)) << ", but the shard num of second input's row is "
                    << mat_b_strategy.at(SECOND_FROM_END(mat_b_size));
      return FAILED;
    } else if (transpose_b_ && (mat_a_strategy.at(SECOND_FROM_END(mat_a_size)) != mat_b_strategy.back())) {
      // for example: mat_a_strategy:[2,4,16,8], mat_b_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(mat_a_strategy) << " and mat_b "
                    << ShapeToString(mat_b_strategy) << ". The transpose_a is: " << transpose_a_
                    << ", and transpose_b is " << transpose_b_ << ", the shard num of first input's row is "
                    << mat_a_strategy.at(SECOND_FROM_END(mat_a_size))
                    << ", but the shard num of second input's column is " << mat_b_strategy.back();
      return FAILED;
    }
  } else {
    if (!transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.at(SECOND_FROM_END(mat_b_size)))) {
      // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,16,32], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(mat_a_strategy) << " and mat_b "
                    << ShapeToString(mat_b_strategy) << ". The transpose_a is: " << transpose_a_
                    << ", and transpose_b is " << transpose_b_ << ", the shard num of first input's column is "
                    << mat_a_strategy.back() << ", but the shard num of second input's row is "
                    << mat_b_strategy.at(SECOND_FROM_END(mat_b_size));
      return FAILED;
    } else if (transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.back())) {
      // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,32,16], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_a " << ShapeToString(mat_a_strategy) << " and mat_b "
                    << ShapeToString(mat_b_strategy) << ". The transpose_a is: " << transpose_a_
                    << ", and transpose_b is " << transpose_b_ << ", the shard num of first input's column is "
                    << mat_a_strategy.back() << ", but the shard num of second input's column is "
                    << mat_b_strategy.back();
      return FAILED;
    }
  }

  if (mat_a_size >= mat_b_size) {
    if (CheckBatchDimensions(mat_a_strategy, mat_b_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
      return FAILED;
    }
  } else {
    if (CheckBatchDimensions(mat_b_strategy, mat_a_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Strategies of batch dimensions are not equal.";
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

  return CheckInputStrategy(mat_a_strategy, mat_b_strategy);
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

Status MatMulBase::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions mat_a_strategy = stra.at(0);
  Dimensions mat_b_strategy = stra.at(1);
  if (candidate_flag_) {
    dev_matrix_shape_ = mat_a_strategy;
    return SUCCESS;
  }
  dev_matrix_shape_ = GetCommonShape(mat_a_strategy, mat_b_strategy);
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  MS_LOG(DEBUG) << name_ << ": The dev matrix shape is " << dev_matrix_shape_;
  return SUCCESS;
}

Status MatMulBase::InferForwardCommunication() {
  if (candidate_flag_) {
    return SUCCESS;
  }

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
  } else if (group_list.empty() || group_list.front().GetDevNum() <= kSizeOne) {
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

Status MatMul::InferOutputTensorMap() {
  Shape mat_a_map = inputs_tensor_map_[0];
  Shape mat_b_map = inputs_tensor_map_[1];

  if (transpose_a_ || transpose_b_) {
    MS_LOG(ERROR) << name_ << ": config layout can not support transpose_a or transpose_b";
    return FAILED;
  }

  if (mat_a_map.size() != mat_b_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  Shape mat_a_batch_map(mat_a_map.cbegin(), mat_a_map.cbegin() + mat_a_map.size() - 2);
  Shape mat_b_batch_map(mat_b_map.cbegin(), mat_b_map.cbegin() + mat_b_map.size() - 2);

  if (mat_a_batch_map.size() != mat_b_batch_map.size()) {
    MS_LOG(ERROR) << name_ << ": config layout can not support broadcast";
    return FAILED;
  }

  int64_t relevant_dim_map = mat_a_map[mat_a_map.size() - 1];
  if (relevant_dim_map != MAP_NONE) {
    MS_LOG(ERROR) << name_ << ": config layout can not support shard relevant dimension";
    return FAILED;
  }

  Shape output_map = mat_a_batch_map;
  output_map.push_back(mat_a_map[mat_a_map.size() - 2]);
  output_map.push_back(mat_b_map[mat_b_map.size() - 1]);
  outputs_tensor_map_.push_back(output_map);
  MS_LOG(INFO) << name_ << ": the input tensor map is " << inputs_tensor_map_ << ", the output tensor map is "
               << output_map;
  return SUCCESS;
}

Status MatMul::CheckNDTPInputLayout(const TensorLayout &a_in_layout, const TensorLayout &b_in_layout) {
  MS_LOG(INFO) << "CheckNDTPInputLayout";
  if (!enable_nd_tp_) {
    return FAILED;
  }

  auto a_tensor_map = a_in_layout.tensor_map_before();
  auto b_tensor_map = b_in_layout.tensor_map_before();
  if (a_tensor_map.size() < kSizeTwo || b_tensor_map.size() < kSizeTwo) {
    MS_LOG(ERROR) << "The size of input_tensor_map for matmul are " << a_tensor_map.size() << " and "
                  << b_tensor_map.size() << ", which should not be less than 2";
    return FAILED;
  }

  size_t axis0_0 = a_in_layout.tensor_shape_before().array().size() - kIndex2;
  size_t axis0_1 = axis0_0 + kIndex1;
  if (transpose_a_) {
    size_t temp = axis0_0;
    axis0_0 = axis0_1;
    axis0_1 = temp;
  }

  size_t axis1_0 = b_in_layout.tensor_shape_before().array().size() - kIndex2;
  size_t axis1_1 = axis1_0 + kIndex1;
  if (transpose_b_) {
    size_t temp = axis1_0;
    axis1_0 = axis1_1;
    axis1_1 = temp;
  }

  // for 2D-WS: x,y       y,x
  // for 3D: (z,x),y     (z,y),x
  // In the above case, the last dim are both not divided
  if (a_tensor_map[axis0_0].size() != kSizeOne || a_tensor_map[axis0_1].size() != kSizeOne ||
      b_tensor_map[axis1_0].size() != kSizeOne || b_tensor_map[axis1_1].size() != kSizeOne) {
    // Judge whether  meet these conditions for the 3D.
    return Check3DTPInputLayout(a_in_layout, b_in_layout, axis0_0, axis0_1, axis1_0, axis1_1);
  } else {
    // Judge whether  meet these conditions for 2D.
    if (a_tensor_map[axis0_0] != b_tensor_map[axis1_1] || a_tensor_map[axis0_1] != b_tensor_map[axis1_0]) {
      MS_LOG(ERROR) << "For 2D MatMul/Batch MatMul, the input layout for the last two dimensions should be like: \n"
                    << " (x, y), (x, y) when one of transpose_a and transpose_b is 'true'; or (x, y), (y, x) in the "
                       "other situation. But now they are: ("
                    << a_tensor_map[a_in_layout.tensor_shape_before().array().size() - kSizeTwo] << ", "
                    << a_tensor_map[a_in_layout.tensor_shape_before().array().size() - kSizeOne] << "), ("
                    << b_tensor_map[b_in_layout.tensor_shape_before().array().size() - kSizeTwo] << ", "
                    << b_tensor_map[b_in_layout.tensor_shape_before().array().size() - kSizeOne] << ").";
      return FAILED;
    }
    MS_LOG(INFO) << "2D TP inputLayout check pass, it is activated.";
    return SUCCESS;
  }
}

Status MatMul::Check3DTPInputLayout(const TensorLayout &a_in_layout, const TensorLayout &b_in_layout, size_t axis0_0,
                                    size_t axis0_1, size_t axis1_0, size_t axis1_1) {
  auto a_tensor_map = a_in_layout.tensor_map_before();
  auto b_tensor_map = b_in_layout.tensor_map_before();
  if (a_tensor_map[axis0_1].size() != kSizeOne ||
      b_tensor_map[axis1_1].size() !=
        kSizeOne  // any one of the last dimension should not be divided more than once in the input layout
      || a_tensor_map[axis0_0].size() != kSizeTwo ||
      b_tensor_map[axis1_0].size() != kSizeTwo  // the second last dimension should be divide twice.
      || a_tensor_map[axis0_0][kIndex0] != b_tensor_map[axis1_0][kIndex0]  // the z in two inputs should be equal
      || a_tensor_map[axis0_0][kIndex1] != b_tensor_map[axis1_1][kIndex0] ||
      a_tensor_map[axis0_1][kIndex0] != b_tensor_map[axis1_0][kIndex1]  // the pattern of x and y should be satisfied.
  ) {
    MS_LOG(ERROR) << "For 3D MatMul/Batch MatMul, the input layout for the last two dimensions should be like: \n"
                  << " ((z,x),y), (x,(z,y))  when transpose_b is 'true'; or "
                  << " (y,(z,x)), ((z,y),x)  when transpose_a is 'true'; or "
                  << " ((z,x),y), ((z,y),x)  in the other situation. But now they are: ("
                  << a_tensor_map[a_in_layout.tensor_shape_before().array().size() - kSizeTwo] << ", "
                  << a_tensor_map[a_in_layout.tensor_shape_before().array().size() - kSizeOne] << "), ("
                  << b_tensor_map[b_in_layout.tensor_shape_before().array().size() - kSizeTwo] << ", "
                  << b_tensor_map[b_in_layout.tensor_shape_before().array().size() - kSizeOne] << ").";
    return FAILED;
  }
  three_d_tp_ = true;
  MS_LOG(INFO) << "3D TP inputLayout check pass, it is activated.";
  return SUCCESS;
}

Status MatMul::CheckInputLayout() {
  // Check all device matrix should be the same
  if (inputs_tensor_info_.size() != kSizeTwo) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for matmul is " << inputs_tensor_info_.size()
                  << " rather than 2.";
    return FAILED;
  }
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  if (in_layout0.device_arrangement_origin().array() != in_layout1.device_arrangement_origin().array()) {
    MS_LOG(ERROR) << "The device_matrix of input0 " << in_layout0.device_arrangement_origin().array()
                  << " dose not equal to device_matrix of input1 " << in_layout1.device_arrangement_origin().array();
    return FAILED;
  }

  //  2D MatMul: mat_a_strategy is [x, y] and mat_b_strategy is [x, y], and use replace graph to handle it
  if (enable_nd_tp_) {
    MS_LOG(INFO) << name_ << ": Using  2D/3D MatMul.";
    return CheckNDTPInputLayout(in_layout0, in_layout1);
  }

  std::vector<int64_t> map_verify;
  size_t axis0 = in_layout0.tensor_shape_before().array().size() - 1;
  if (transpose_a_) {
    axis0--;
  }
  auto m_v =
    in_layout0.tensor_map_before()[in_layout0.tensor_shape_before().array().size() * kSizeTwo - kIndex3 - axis0];
  (void)std::copy(m_v.begin(), m_v.end(), std::back_inserter(map_verify));
  size_t axis1 = in_layout1.tensor_shape_before().array().size() - 2;
  if (transpose_b_) {
    axis1++;
  }
  auto n_v =
    in_layout1.tensor_map_before()[in_layout1.tensor_shape_before().array().size() * kSizeTwo - kIndex3 - axis1];
  (void)std::copy(n_v.begin(), n_v.end(), std::back_inserter(map_verify));

  if (in_layout0.tensor_map_before()[axis0] != in_layout1.tensor_map_before()[axis1]) {
    MS_LOG(ERROR) << "The shard size of reduce_dim is not equal for input0 and input1";
    return FAILED;
  }

  std::sort(map_verify.begin(), map_verify.end());
  for (size_t i = 0; i + 1 < map_verify.size(); ++i) {
    if (map_verify[i] == map_verify[i + 1] && map_verify[i] > 0) {
      MS_LOG(ERROR) << "The device_matrix " << in_layout0.device_arrangement_origin().array() << " axis "
                    << in_layout0.device_arrangement_origin().array().size() - 1 - LongToSize(map_verify[i])
                    << " has been shard for more than once and not sharding the reduce_dim for matmul.";
      return FAILED;
    }
  }
  if (in_layout0.IsInterleavedParallel()) {
    auto tensor_map_interleaved0 = in_layout0.tensor_map_before();
    auto reduce_axis_map = tensor_map_interleaved0[axis0];
    if (std::find(reduce_axis_map.begin(), reduce_axis_map.end(), 0) != reduce_axis_map.end()) {
      MS_LOG(ERROR) << "Only support splitting micro interleaved in batch axis for matmul.";
      return FAILED;
    }
  }
  if (in_layout1.IsInterleavedParallel()) {
    auto tensor_map_intereaved1 = in_layout1.tensor_map_before();
    if (std::any_of(tensor_map_intereaved1.begin(), tensor_map_intereaved1.end(),
                    [](const auto &map1) { return std::find(map1.begin(), map1.end(), 0) != map1.end(); })) {
      MS_LOG(ERROR) << "Only support splitting micro interleaved in batch axis for matmul.";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatMul::CheckOutputLayout() {
  // Check all device matrix should be the same
  if (outputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << "The size of output_tensor_layout for matmul is " << outputs_tensor_info_.size()
                  << " rather than 1.";
    return FAILED;
  }
  auto out_layout = outputs_tensor_info_[kIndex0].tensor_layout();
  if (!output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    MS_LOG(INFO) << "Using output tensor layout infer by input tensor layout.";
    UpdateOutputTensorInfoForInterleaved();
    return SUCCESS;
  }
  output_infer_tensor_layout_ = InferOutputLayout();
  if (output_infer_tensor_layout_ == out_layout) {
    MS_LOG(INFO)
      << "output tensor layout infer by input tensor layout is same with user configured output tensor layout.";
    UpdateOutputTensorInfoForInterleaved();
    return SUCCESS;
  } else if (enable_nd_tp_) {
    return FAILED;
  }

  auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  int64_t axis0 = SizeToLong(input_layout0.tensor_shape_before().array().size()) - 1;
  if (transpose_a_) {
    axis0 -= 1;
  }
  auto output_extended_tensor_map = output_infer_tensor_layout_.tensor_map_before();
  auto axis_map = input_layout0.tensor_map_before()[axis0];
  (void)output_extended_tensor_map[0].insert(output_extended_tensor_map[0].end(), axis_map.begin(), axis_map.end());
  TensorLayout reduce_scatter_out_layout;
  reduce_scatter_out_layout.InitFromExtendVector(output_infer_tensor_layout_.device_arrangement_origin().array(),
                                                 output_extended_tensor_map,
                                                 output_infer_tensor_layout_.tensor_shape_before().array());
  if (reduce_scatter_out_layout != out_layout) {
    MS_LOG(ERROR) << "The user configured output layout { device_matrix:"
                  << out_layout.device_arrangement_origin().array() << ", tensor_map:" << out_layout.tensor_map_before()
                  << ", tensor_shape:" << out_layout.tensor_shape_before().array()
                  << " } dose not match the inferred output layout { device_matrix:"
                  << output_infer_tensor_layout_.device_arrangement_origin().array()
                  << ", tensor_map:" << output_infer_tensor_layout_.tensor_map_before()
                  << ", tensor_shape:" << output_infer_tensor_layout_.tensor_shape_before().array()
                  << " } (using all_reduce) or { device_matrix:"
                  << reduce_scatter_out_layout.device_arrangement_origin().array()
                  << ", tensor_map:" << reduce_scatter_out_layout.tensor_map_before()
                  << ", tensor_shape:" << reduce_scatter_out_layout.tensor_shape_before().array()
                  << " } (using reduce_scatter)";
    return FAILED;
  }
  forward_reduce_scatter_ = true;
  UpdateOutputTensorInfoForInterleaved();
  return SUCCESS;
}

TensorLayout MatMul::InferNDTPOutputLayout() {
  TensorLayout output_tensor_layout;

  if (!enable_nd_tp_) {
    return output_tensor_layout;
  }

  auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  size_t axis0 = input_layout0.tensor_shape_before().array().size() - kIndex1;
  if (transpose_a_) {
    axis0 -= kIndex1;
  }

  std::vector<Shape> output_extended_tensor_map;
  Shape output_tensor_shape;

  for (size_t i = 0; i < input_layout0.tensor_shape_before().array().size(); ++i) {
    auto shp_dim = input_layout0.tensor_shape_before().array()[i];
    if (i != axis0) {
      output_tensor_shape.push_back(shp_dim);
    }
  }

  if (!transpose_b_) {
    output_tensor_shape.push_back(
      input_layout1.tensor_shape_before().GetDimByIdx(inputs_shape_[kIndex1].size() - kIndex1));
    output_extended_tensor_map = input_layout1.tensor_map_before();
  } else {
    output_tensor_shape.push_back(
      input_layout1.tensor_shape_before().GetDimByIdx(inputs_shape_[kIndex1].size() - kIndex2));
    // the last two dims output_extended_tensor_map swap position and others unchange
    for (size_t i = 0; i < input_layout1.tensor_shape_before().array().size() - kIndex2; ++i) {
      output_extended_tensor_map.push_back(input_layout1.tensor_map_before()[i]);
    }
    output_extended_tensor_map.push_back(
      input_layout1.tensor_map_before()[input_layout1.tensor_shape_before().array().size() - kIndex1]);
    output_extended_tensor_map.push_back(
      input_layout1.tensor_map_before()[input_layout1.tensor_shape_before().array().size() - kIndex2]);
  }

  output_tensor_layout.InitFromExtendVector(input_layout0.device_arrangement_origin().array(),
                                            output_extended_tensor_map, output_tensor_shape);
  return output_tensor_layout;
}

TensorLayout MatMul::InferOutputLayout() {
  if (enable_nd_tp_) {
    MS_LOG(INFO) << name_ << ": Infer  2D/3D MatMul output layout.";
    return InferNDTPOutputLayout();
  }

  auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  size_t axis0 = input_layout0.tensor_shape_before().array().size() - 1;
  if (transpose_a_) {
    axis0 -= 1;
  }
  std::vector<Shape> output_extended_tensor_map;
  Shape output_tensor_shape;

  for (size_t i = 0; i < input_layout0.tensor_shape_before().array().size(); ++i) {
    auto map_dim = input_layout0.tensor_map_before()[i];
    auto shp_dim = input_layout0.tensor_shape_before().array()[i];
    if (i != axis0) {
      output_extended_tensor_map.push_back(map_dim);
      output_tensor_shape.push_back(shp_dim);
    }
  }

  if (!transpose_b_) {
    output_extended_tensor_map.push_back(input_layout1.tensor_map_before()[inputs_shape_[kIndex1].size() - kDim1]);
    output_tensor_shape.push_back(
      input_layout1.tensor_shape_before().GetDimByIdx(inputs_shape_[kIndex1].size() - kDim1));
  } else {
    output_extended_tensor_map.push_back(input_layout1.tensor_map_before()[inputs_shape_[kIndex1].size() - kDim2]);
    output_tensor_shape.push_back(
      input_layout1.tensor_shape_before().GetDimByIdx(inputs_shape_[kIndex1].size() - kDim2));
  }

  TensorLayout output_tensor_layout;
  output_tensor_layout.InitFromExtendVector(input_layout0.device_arrangement_origin().array(),
                                            output_extended_tensor_map, output_tensor_shape);
  return output_tensor_layout;
}

Status MatMul::InferOutputTensorInfo() {
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

Status MatMul::InferForwardCommunicationByLayout() {
  if (enable_nd_tp_) {
    return SUCCESS;
  }

  forward_op_.clear();
  forward_op_interleaved_.clear();
  auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  size_t axis0 = input_layout0.tensor_shape_before().array().size() - 1;
  if (transpose_a_) {
    axis0 -= 1;
  }
  auto axis_tensor_map = input_layout0.tensor_map_before()[axis0];
  int64_t axis_shard = 1;
  std::vector<int64_t> r_dim_vector;
  for (const auto &dim : axis_tensor_map) {
    if (dim == -1) {
      continue;
    }
    int64_t divisor = input_layout0.device_arrangement_origin().GetDimByReverseIdx(LongToUlong(dim));
    axis_shard *= divisor;
    auto r_dim = SizeToLong(input_layout0.device_arrangement_origin().array().size() - 1) - dim;
    r_dim_vector.push_back(r_dim);
  }
  // Relevant dimension is not split and all reduce is not required,
  if (axis_shard == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward communication is not required.";
    return SUCCESS;
  }
  RankList repeated_rank_list;
  auto device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                    input_layout0.device_arrangement_origin().array());
  if (device_matrix.GetDevicesAlongMultiDim(r_dim_vector, &repeated_rank_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Forward communication by multi axis failed.";
    return FAILED;
  }
  if (repeated_rank_list.size() == 1) {
    MS_LOG(INFO) << name_ << ": Forward communication is not required.";
    return SUCCESS;
  }

  Group forward_group;
  if (g_device_manager->CreateGroup(repeated_rank_list, &forward_group) != SUCCESS) {
    MS_LOG(ERROR) << name_
                  << ": Create communication group by tensor_map failed, the rank_list is: " << repeated_rank_list
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }

  Operator op;
  if (forward_reduce_scatter_) {
    op = CreateReduceScatterOp(REDUCE_OP_SUM, forward_group.name());
  } else {
    op = CreateAllReduceOp(REDUCE_OP_SUM, forward_group.name());
  }
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel()) {
    forward_op_interleaved_.push_back(op);
  } else {
    forward_op_.push_back(op);
  }
  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << forward_group.name();
  return SUCCESS;
}

Status MatMul::CheckLayoutConfig() {
  if (CheckInputStrategy(strategy_from_layout_[0], strategy_from_layout_[1]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": invalid layout config, the dev matrix is " << dev_matrix_shape_
                  << ", and input tensor map is " << inputs_tensor_map_;
    return FAILED;
  }
  return SUCCESS;
}

Status MatMulBase::InferTensorMap() {
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  size_t size = origin_dev_matrix_shape_.size();

  Shape tensor_map_index;
  // such as 5: tensor_map_index [4,3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(LAST_INDEX(size) - i));
  }

  if (candidate_flag_) {
    inputs_tensor_map_.push_back({1, 0});
    inputs_tensor_map_.push_back({1, 0});
    outputs_tensor_map_.push_back({1, 0});
    return SUCCESS;
  }
  // infer output tensor map: [4,3,2,0], delete the second-from-end element
  TensorMap output_tensor_map = tensor_map_index;
  (void)output_tensor_map.erase(output_tensor_map.cbegin() + static_cast<different_type>(SECOND_FROM_END(size)));

  // infer mat_a tensor map
  // for example: mat_a_dimension is 4, mat_a tensor map:[4,3,2,1]
  TensorMap mat_a_tensor_map = tensor_map_index;
  // delete last one element
  mat_a_tensor_map.pop_back();
  // delete the first (dev_matrix_size - 1 - mat_a_dimension) elements
  (void)mat_a_tensor_map.erase(
    mat_a_tensor_map.cbegin(),
    mat_a_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - mat_a_dimension_));
  if (transpose_a_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&mat_a_tensor_map);
  }

  // infer mat_b tensor map
  TensorMap mat_b_tensor_map = tensor_map_index;
  // delete the third-to-last element
  (void)mat_b_tensor_map.erase(mat_b_tensor_map.cbegin() + static_cast<different_type>(THIRD_FROM_END(size)));
  // delete the first (dev_matrix_size - 1 - mat_b_dimension) elements
  (void)mat_b_tensor_map.erase(
    mat_b_tensor_map.cbegin(),
    mat_b_tensor_map.cbegin() + static_cast<different_type>(LAST_INDEX(size) - mat_b_dimension_));
  if (transpose_b_) {
    // swap the last two elements
    (void)SwapLastTwoElements(&mat_b_tensor_map);
  }

  if (forward_reduce_scatter_) {
    // the forward reduce scatter only support that the dimension of output is 2
    output_tensor_map = {1, 0};
  }

  // handle broadcast
  for (size_t i = 0; i < mat_a_tensor_map.size(); ++i) {
    if (inputs_shape_[0][i] == 1) {
      mat_a_tensor_map[i] = MAP_NONE;
    }
  }

  for (size_t j = 0; j < mat_b_tensor_map.size(); ++j) {
    if (inputs_shape_[1][j] == 1) {
      mat_b_tensor_map[j] = MAP_NONE;
    }
  }

  inputs_tensor_map_.push_back(mat_a_tensor_map);
  inputs_tensor_map_.push_back(mat_b_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(DEBUG) << name_ << ": The mat_a's tensor map is " << mat_a_tensor_map << ", the mat_b's tensor map is "
                << mat_b_tensor_map << ", the output's tensor map is " << output_tensor_map;
  return SUCCESS;
}

Status MatMulBase::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  out_dev_matrix_shape_ = dev_matrix_shape_;
  if (forward_reduce_scatter_) {
    // the reduce scatter mode only use for MatMul
    if (repeated_num_in_dev_matrix_right_ || repeated_calc_num_ == 1) {
      // dev_matrix_shape_ is: [a, b, c, repeat_num] or [a, b, c]
      // out_dev_matrix_shape_ is: [a*b, c, repeat_num] or [a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin(), out_dev_matrix_shape_.cbegin() + 2);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin(), dev_matrix_shape_[0] * dev_matrix_shape_[1]);
    } else {
      // dev_matrix_shape_ is: [repeat_num, a, b, c]
      // out_dev_matrix_shape_ is: [repeat_num, a*b, c]
      (void)out_dev_matrix_shape_.erase(out_dev_matrix_shape_.cbegin() + 1, out_dev_matrix_shape_.cbegin() + 3);
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.cbegin() + 1,
                                         dev_matrix_shape_[1] * dev_matrix_shape_[2]);
    }
  }

  TensorLayout mat_a_layout;
  TensorLayout mat_b_layout;
  TensorLayout output_layout;
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

  // e.g. mat_a: [A, B, C, D], mat_b: [B, D, E], then to generate the strategy for [A, B, C, D, E]
  // e.g. mat_a: [B, C, D], mat_b: [A, 1, D, E], then to generate the strategy for [A, B, C, D, E]
  size_t long_shape_size = mat_a_shape.size();
  if (mat_a_shape.size() < mat_b_shape.size()) {
    long_shape_size = mat_b_shape.size();
  }
  std::vector<StrategyPtr> sp_vector;
  Shape splittable_flag(long_shape_size + 1, 1);
  Shapes splittable_input = {splittable_flag};
  Shape tmp_shape = GetCommonShape(mat_a_shape, mat_b_shape);
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
    Dimensions tmp_strategy = sp->GetInputDim()[0];  // [A, B, C, D, E]

    size_t diff_len = 0;

    // handle mat_a's strategy
    Dimensions mat_a_strategy = tmp_strategy;
    mat_a_strategy.pop_back();  // [A, B, C, D]
    if (mat_a_shape.size() < mat_b_shape.size()) {
      diff_len = mat_b_shape.size() - mat_a_shape.size();
      (void)mat_a_strategy.erase(mat_a_strategy.cbegin(),
                                 mat_a_strategy.cbegin() + static_cast<different_type>(diff_len));  // [B, C, D]
    }

    // transpose_a
    if (transpose_a_) {
      (void)SwapLastTwoElements(&mat_a_strategy);
    }

    if (mat_a_strategy.size() != mat_a_shape.size()) {
      MS_LOG(EXCEPTION) << name_ << ": The size of mat_a_shape and mat_a_strategy must be equal, the mat_a_shape is "
                        << mat_a_shape << ", but the mat_a_strategy is " << mat_a_strategy;
    }

    // broadcast
    for (size_t i = 0; i < mat_a_strategy.size(); ++i) {
      if (mat_a_shape[i] <= 1) {
        mat_a_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    // handle mat_b's strategy
    Dimensions mat_b_strategy = tmp_strategy;  // [A, B, C, D, E]
    // mat_b_strategy: delete C, [A, B, D, E]
    (void)mat_b_strategy.erase(mat_b_strategy.cend() - 3);
    // mat_b_strategy: delete A, [B, D, E]

    if (mat_b_shape.size() < mat_a_shape.size()) {
      diff_len = mat_a_shape.size() - mat_b_shape.size();
      (void)mat_b_strategy.erase(mat_b_strategy.cbegin(),
                                 mat_b_strategy.cbegin() + static_cast<different_type>(diff_len));
    }

    // handle transpose_b
    if (transpose_b_) {
      (void)SwapLastTwoElements(&mat_b_strategy);
    }

    // broadcast
    for (size_t i = 0; i < mat_b_strategy.size(); ++i) {
      if (mat_b_shape[i] <= 1) {
        mat_b_strategy[i] = NO_SPLIT_STRATEGY;
      }
    }

    replace_strategy.push_back(mat_a_strategy);
    replace_strategy.push_back(mat_b_strategy);
    sp->ResetInputs(replace_strategy);
  }
  return sp_vector;
}

std::shared_ptr<Strategies> MatMulInfo::GenerateBatchStrategies() {
  Dimensions batch_strategy_a(inputs_shape_[0].size(), 1);
  Dimensions batch_strategy_b(inputs_shape_[1].size(), 1);
  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);
  Strategies strategy_v;

  if (transpose_a_) {
    batch_strategy_a[1] = stage_device_size_;
  } else {
    batch_strategy_a[0] = stage_device_size_;
  }

  strategy_v = {batch_strategy_a, batch_strategy_b};
  return std::make_shared<Strategies>(strategy_v);
}

std::shared_ptr<Strategies> BatchMatMulInfo::GenerateBatchStrategies() {
  Dimensions batch_strategy_a(inputs_shape_[0].size(), 1);
  Dimensions batch_strategy_b(inputs_shape_[1].size(), 1);
  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);
  Strategies strategy_v;
  // input's shape equals to weight's shape
  if (inputs_shape_[0].size() == inputs_shape_[1].size()) {
    batch_strategy_a[0] = stage_device_size_;
    if (inputs_shape_[0].size() > MATMUL_INPUTS_SIZE) {
      batch_strategy_b[0] = stage_device_size_;
    }
  }
  if (inputs_shape_[0].size() > inputs_shape_[1].size()) {
    batch_strategy_a[0] = stage_device_size_;
  }
  if (inputs_shape_[0].size() < inputs_shape_[1].size()) {
    batch_strategy_b[0] = stage_device_size_;
  }
  strategy_v = {batch_strategy_a, batch_strategy_b};
  return std::make_shared<Strategies>(strategy_v);
}

Status MatMulBase::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status MatMul::ComputeNDTPReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "GenerateGraph Init failed";
  }
  auto input_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  auto device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                    input_layout0.device_arrangement_origin().array());
  size_t all_gather_tensor_axis = SECOND_FROM_END(input_layout0.tensor_shape_before().array().size());
  size_t scatter_tensor_axis = all_gather_tensor_axis;
  size_t all_gather_tensor_axis1 = SECOND_FROM_END(input_layout1.tensor_shape_before().array().size());
  if (transpose_a_) {
    all_gather_tensor_axis += kIndex1;
    // scatter_tensor_axis should remain unchanged
  }
  if (transpose_b_) {
    all_gather_tensor_axis1 += kIndex1;
  }
  int64_t all_gather_dim0 =
    GetAllGatherDim(all_gather_tensor_axis, input_layout0, kIndex0);  // left input all gather device dim.
  int64_t all_gather_dim1 =
    GetAllGatherDim(all_gather_tensor_axis1, input_layout1, kIndex1);  // right input (weight) all gather device dim.

  std::vector<Group> x_group_list;
  if (all_gather_dim0 != -1) {
    if (CreateGroupByDimWithDevMatrix(&device_matrix, all_gather_dim0, &x_group_list) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << "Create group failed";
    }
  }
  bool x_flag = !x_group_list.empty();
  std::vector<Group> z_group_list;  // 3D TP z dimension communication group
  if (all_gather_dim1 != -1) {
    if (CreateGroupByDimWithDevMatrix(&device_matrix, all_gather_dim1, &z_group_list) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << "Create group failed";
    }
  }
  bool z_flag = !z_group_list.empty();
  AnfNodePtr matmul_left_input;
  AnfNodePtr matmul_right_input;
  AnfNodePtr x_all_gather;
  AnfNodePtr z_all_gather;
  if (x_flag) {
    OperatorAttrs all_gather_attrs;
    Attr attr_group = std::make_pair(GROUP, MakeValue(x_group_list[kIndex0].name()));
    all_gather_attrs.push_back(attr_group);
    x_all_gather = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER, all_gather_attrs), gen_g.virtual_input_node()});
    matmul_left_input = this->ComputePreAllGatherGraph(cnode, &gen_g, x_group_list, SizeToLong(all_gather_tensor_axis),
                                                       x_all_gather, transpose_a_);
  } else {
    matmul_left_input = gen_g.virtual_input_node();
  }
  if (z_flag) {
    OperatorAttrs all_gather_attrs;
    Attr attr_group = std::make_pair(GROUP, MakeValue(z_group_list[kIndex0].name()));
    all_gather_attrs.push_back(attr_group);
    z_all_gather = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER, all_gather_attrs), gen_g.virtual_input_node()});
    matmul_right_input = this->ComputePreAllGatherGraph(
      cnode, &gen_g, z_group_list, SizeToLong(all_gather_tensor_axis1), z_all_gather, transpose_b_);
  } else {
    matmul_right_input = gen_g.virtual_input_node();
  }
  // matmul
  OperatorAttrs matmul_attrs = {std::make_pair(TRANSPOSE_A, MakeValue(transpose_a_)),
                                std::make_pair(TRANSPOSE_B, MakeValue(transpose_b_))};
  auto matmul = gen_g.PushBack(
    {gen_g.NewOpInst(IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul) ? BATCH_MATMUL : MATMUL, matmul_attrs),
     matmul_left_input, matmul_right_input});
  std::pair<AnfNodePtr, int64_t> left_input_node;
  std::pair<AnfNodePtr, int64_t> right_input_node;

  left_input_node = x_flag ? std::make_pair(x_all_gather, kIndex1) : std::make_pair(matmul, kIndex1);
  right_input_node = z_flag ? std::make_pair(z_all_gather, kIndex2) : std::make_pair(matmul, kIndex2);
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {left_input_node, right_input_node};
  // post matmul
  AnfNodePtr post_matmul_op =
    this->ComputePostMatMulGraph(cnode, &gen_g, matmul, input_layout0, SizeToLong(scatter_tensor_axis));
  if (post_matmul_op != nullptr) {
    replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
      std::make_pair(input_nodes, post_matmul_op));
  } else {
    replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
      std::make_pair(input_nodes, matmul));
  }
  return SUCCESS;
}

int64_t MatMul::GetAllGatherDim(size_t all_gather_tensor_axis, const TensorLayout &input_layout, size_t tensor_index) {
  auto all_gather_tensor_axis_map = input_layout.tensor_map_before()[all_gather_tensor_axis];

  if (tensor_index == kIndex0) {
    if (three_d_tp_) {
      return all_gather_tensor_axis_map[kIndex1] == -1
               ? -1
               : SizeToLong(input_layout.device_arrangement_origin().array().size() - kIndex1) -
                   all_gather_tensor_axis_map[kIndex1];
    } else {
      return all_gather_tensor_axis_map[kIndex0] == -1
               ? -1
               : SizeToLong(input_layout.device_arrangement_origin().array().size() - 1) -
                   all_gather_tensor_axis_map[kIndex0];  // 2D-WS doesn't allow a axis to be divided more than once.
    }
  } else {
    if (three_d_tp_) {
      return all_gather_tensor_axis_map[kIndex0] == -1
               ? -1
               : SizeToLong(input_layout.device_arrangement_origin().array().size() - kIndex1) -
                   all_gather_tensor_axis_map[kIndex0];
    }
  }
  return -1;
}

AnfNodePtr MatMul::ComputePreAllGatherGraph(const CNodePtr &cnode, GenerateGraph *gen_g,
                                            const std::vector<Group> &device_group_list, int64_t all_gather_tensor_axis,
                                            const AnfNodePtr &all_gather_node, bool transpose) {
  // BatchMatMul
  // The gather dim of the All Gather ops is 0 by default, but we need it to gather in the `all_gather_tensor_axis`,
  // so the tensor should be split in the 0 dim and concat to the  `all_gather_tensor_axis` dim
  if (IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul) || transpose) {
    size_t split_count = device_group_list[0].GetDevicesList().size();
    int64_t split_axis = kIndex0;
    Attr split_axis_attr = std::make_pair(AXIS, MakeValue(split_axis));
    Attr split_count_attr = std::make_pair(OUTPUT_NUM, MakeValue(SizeToLong(split_count)));
    OperatorAttrs split_attrs = {split_axis_attr, split_count_attr};
    auto split = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), all_gather_node});
    // tuple get item
    std::vector<AnfNodePtr> make_tuple_inputs;
    make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (int64_t i = 0; i < SizeToLong(split_count); ++i) {
      auto tuple_get_item = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), split, CreatInt64Imm(i)});
      make_tuple_inputs.push_back(tuple_get_item);
    }
    // make tuple
    auto make_tuple = gen_g->PushBack(make_tuple_inputs);
    // concat
    int64_t concat_axis = all_gather_tensor_axis;  // concat_axis is the index of the tensor, not the dev_matrix.
    auto concat = gen_g->PushBack({gen_g->NewOpInst(CONCAT), make_tuple, CreatInt64Imm(concat_axis)});
    return concat;
  } else {
    // MatMul
    // special case: the left tensor with tanspose_a_=true, and the right tensor with transpose_b_=true
    // their scatter axis are both the last dim, needing the split.
    return all_gather_node;
  }
}

AnfNodePtr MatMul::ComputePostMatMulGraph(const CNodePtr &cnode, GenerateGraph *gen_g, const AnfNodePtr &matmul,
                                          const TensorLayout &input_layout, int64_t scatter_tensor_axis) {
  size_t reduce_scatter_tensor_axis = LAST_INDEX(input_layout.tensor_shape_before().array().size());
  if (transpose_a_) {
    reduce_scatter_tensor_axis -= 1;
  }
  std::vector<Group> reduce_scatter_group_list;
  int64_t reduce_scatter_dim = -1;
  auto reduce_scatter_tensor_axis_map = input_layout.tensor_map_before()[reduce_scatter_tensor_axis];
  for (const auto &dim : reduce_scatter_tensor_axis_map) {
    if (dim == -1) {
      continue;
    }
    reduce_scatter_dim = SizeToLong(input_layout.device_arrangement_origin().array().size() - kIndex1) - dim;
    break;  // 2D-WS doesn't allow a axis to be divided more than once.
  }
  auto device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                    input_layout.device_arrangement_origin().array());
  if (reduce_scatter_dim != -1) {
    if (CreateGroupByDimWithDevMatrix(&device_matrix, reduce_scatter_dim, &reduce_scatter_group_list) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << "Create group failed";
    }
  }
  bool reduce_scatter_flag = !reduce_scatter_group_list.empty();
  if (reduce_scatter_flag) {
    AnfNodePtr reduce_scatter_op;
    Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
    Attr attr_group = std::make_pair(GROUP, MakeValue(reduce_scatter_group_list[kIndex0].name()));
    OperatorAttrs attrs = {attr_op, attr_group};
    if (IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul)) {
      // transpose the 0th dim and the wanted scatter dim of the tensor.
      size_t tensor_array_size_before = input_layout.tensor_shape_before().array().size();
      std::vector<int64_t> t1;
      t1.resize(SizeToLong(tensor_array_size_before));
      for (int64_t i = 0; i < SizeToLong(tensor_array_size_before); ++i) {
        t1[i] = i;
      }
      std::swap(t1[kIndex0], t1[scatter_tensor_axis]);
      AnfNodePtr transpose_1 = gen_g->PushBack({gen_g->NewOpInst(TRANSPOSE), matmul, CreateTuple(t1)});
      reduce_scatter_op = gen_g->PushBack({gen_g->NewOpInst(REDUCE_SCATTER, attrs), transpose_1});
      AnfNodePtr transpose_2 = gen_g->PushBack({gen_g->NewOpInst(TRANSPOSE), reduce_scatter_op, CreateTuple(t1)});
      return transpose_2;
    } else {
      reduce_scatter_op = gen_g->PushBack({gen_g->NewOpInst(REDUCE_SCATTER, attrs), matmul});
      return reduce_scatter_op;
    }
  }
  return nullptr;
}

Status MatMul::ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) {
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
  auto trans_a = CreateBoolImm(transpose_a_);
  auto trans_b = CreateBoolImm(transpose_b_);
  std::vector<AnfNodePtr> virtual_converter_end_inputs_vector;
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(virtual_converter_begin, 1)};
  for (int64_t i = 0; i < interleaved_num; ++i) {
    auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin, CreatInt64Imm(i)});
    // matmul
    auto matmul =
      gen_g.PushBack({gen_g.NewOpInst(MATMUL), tuple_get_item, gen_g.virtual_input_node(), trans_a, trans_b});
    input_nodes.push_back(std::make_pair(matmul, 2));
    if (forward_op_interleaved_.empty()) {
      virtual_converter_end_inputs_vector.push_back(matmul);
      continue;
    }
    // create allreduce/reduce_scatter
    auto comm_op = gen_g.PushBack(
      {gen_g.NewOpInst(forward_op_interleaved_.front().first, forward_op_interleaved_.front().second.first), matmul});
    auto comm_cnode = comm_op->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(comm_cnode);
    auto comm_prim = GetCNodePrimitive(comm_cnode);
    auto instance_name = comm_prim->instance_name();
    comm_prim->set_instance_name(FORWARD_OP + instance_name);
    virtual_converter_end_inputs_vector.push_back(comm_op);
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

// PCL matmul
ReplaceGraphPtr MatMul::replace_graph(const CNodePtr &cnode) {
  if (enable_nd_tp_) {
    if (ComputeNDTPReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": ComputeNDTPReplaceGraph failed.";
    }
    return replace_graph_;
  }

  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel()) {
    if (ComputeReplaceGraphForInterleaved(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << " splitting micro interleaved failed.";
    }
    return replace_graph_;
  }

  if (!candidate_flag_) {
    return nullptr;
  }

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "GenerateGraph Init failed";
  }

  std::vector<Group> x_group_list;
  std::vector<Group> w_group_list;
  if (CreateGroupByDim(1, &x_group_list) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "Create group failed";
  }
  if (CreateGroupByDim(0, &w_group_list) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "Create group failed";
  }
  bool x_flag = !x_group_list.empty();
  bool w_flag = !w_group_list.empty();
  AnfNodePtr matmul_left_input, matmul_right_input;
  AnfNodePtr x_all_gather, w_all_gather;
  if (x_flag) {
    OperatorAttrs x_all_gather_attrs;
    Attr x_attr_group = std::make_pair(GROUP, MakeValue(x_group_list[0].name()));
    x_all_gather_attrs.push_back(x_attr_group);
    x_all_gather = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER, x_all_gather_attrs), gen_g.virtual_input_node()});
    // split
    int64_t split_count = dev_matrix_shape_[1];
    int64_t split_axis = 0;

    Attr split_axis_attr = std::make_pair(AXIS, MakeValue(split_axis));
    Attr split_count_attr = std::make_pair(OUTPUT_NUM, MakeValue(split_count));
    OperatorAttrs split_attrs = {split_axis_attr, split_count_attr};
    auto split = gen_g.PushBack({gen_g.NewOpInst(SPLIT, split_attrs), x_all_gather});

    // tuple get item
    std::vector<AnfNodePtr> make_tuple_inputs;
    make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

    for (int64_t i = 0; i < split_count; ++i) {
      auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), split, CreatInt64Imm(i)});
      make_tuple_inputs.push_back(tuple_get_item);
    }

    // make tuple
    auto make_tuple = gen_g.PushBack(make_tuple_inputs);
    // concat
    int64_t concat_axis = 1;
    auto concat = gen_g.PushBack({gen_g.NewOpInst(CONCAT), make_tuple, CreatInt64Imm(concat_axis)});
    matmul_left_input = concat;
  } else {
    matmul_left_input = gen_g.virtual_input_node();
  }

  if (w_flag) {
    OperatorAttrs w_all_gather_attrs;
    Attr w_attr_group = std::make_pair(GROUP, MakeValue(w_group_list[0].name()));
    w_all_gather_attrs.push_back(w_attr_group);
    w_all_gather = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER, w_all_gather_attrs), gen_g.virtual_input_node()});
    matmul_right_input = w_all_gather;
  } else {
    matmul_right_input = gen_g.virtual_input_node();
  }

  // matmul
  auto trans_a = CreateBoolImm(transpose_a_);
  auto trans_b = CreateBoolImm(transpose_b_);
  auto matmul = gen_g.PushBack({gen_g.NewOpInst(MATMUL), matmul_left_input, matmul_right_input, trans_a, trans_b});

  std::pair<AnfNodePtr, int64_t> left_input_node, right_input_node;
  if (x_flag) {
    left_input_node = std::make_pair(x_all_gather, 1);
  } else {
    left_input_node = std::make_pair(matmul, 1);
  }

  if (w_flag) {
    right_input_node = std::make_pair(w_all_gather, 2);
  } else {
    right_input_node = std::make_pair(matmul, 2);
  }

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {left_input_node, right_input_node};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, matmul));

  return replace_graph_;
}
REGISTER(MatMulInfo);
REGISTER(BatchMatMulInfo);
REGISTER(MatMulExtInfo);
REGISTER(BatchMatMulExtInfo);
}  // namespace parallel
}  // namespace mindspore
