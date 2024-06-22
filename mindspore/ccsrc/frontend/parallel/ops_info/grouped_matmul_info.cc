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

#include "frontend/parallel/ops_info/grouped_matmul_info.h"

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
/*
separated means the size of tensorlist not equal 1.
integrated means the size of tensorlist is 1.
split_item        inputs     weight      outputs
      0:      separated     separated    separated
      1:     integrated     b, k, n      separated
      2:      separated     separated    integrated
      3:     integrated     b, k, n      integrated
*/
constexpr size_t kInputX = 0;
constexpr size_t kInputWeight = 1;
constexpr size_t kInputBias = 2;
constexpr size_t kInputScale = 3;
constexpr size_t kInputOffset = 4;
constexpr size_t kInputAntiquantScale = 5;
constexpr size_t kInputAntiquantOffset = 6;
// optional
constexpr size_t kInputGroupList = 7;
// attr
constexpr size_t kInputSplitItem = 8;
// output
constexpr size_t kOutputY = 0;
// TensorShape
constexpr size_t gmmTensor2D = 2;
constexpr size_t gmmTensor3D = 3;
constexpr size_t gmmTensor4D = 4;

// 1.getattr
Status GroupedMatmulInfo::GetAttrs() {
  auto tensorlist_x_shape = inputs_shape_new_.at(kInputX);
  auto tensorlist_w_shape = inputs_shape_new_.at(kInputWeight);
  MS_EXCEPTION_IF_NULL(tensorlist_x_shape);
  MS_EXCEPTION_IF_NULL(tensorlist_w_shape);

  if (!tensorlist_x_shape->is_list() || !tensorlist_w_shape->is_list()) {
    MS_EXCEPTION(ValueError) << "For '" << name_ << "', the input x or input weight is not TensorList";
  }

  mat_x_dimension_ = tensorlist_x_shape->GetElement(0)->GetValue().size();  // get inputx[0] shape size
  mat_w_dimension_ = tensorlist_w_shape->GetElement(0)->GetValue().size();  // get weight[0] shape size
  if (!(mat_x_dimension_ >= gmmTensor2D && mat_x_dimension_ <= gmmTensor4D)) {
    MS_LOG(ERROR) << name_ << ": The dim of mat_x should be 2D ~ 3D Tensor , but the dim of mat_x is "
                  << mat_x_dimension_ << ", the dim of mat_w is " << mat_w_dimension_;
    return FAILED;
  }

  if (mat_x_dimension_ > gmmTensor3D || mat_w_dimension_ > gmmTensor3D) {
    MS_LOG(ERROR) << name_ << ": The dim of mat_x or mat_w can not smaller than 2, but the dim of mat_x is "
                  << mat_x_dimension_ << ", the dim of mat_w is " << mat_w_dimension_;
    return FAILED;
  }

  return SUCCESS;
}

// 2.checkstra
Status GroupedMatmulInfo::CheckStrategy(const StrategyPtr &strategy) {
  NewStrategies stra = strategy->GetInputNewDim();
  auto mat_x_strategy = stra[0]->GetAllElements();
  auto mat_w_strategy = stra[1]->GetAllElements();

  for (size_t i = 0; i < mat_x_strategy.size(); i++) {
    size_t mat_x_size = mat_x_strategy[i].size();
    size_t mat_w_size = mat_w_strategy[i].size();
    if ((mat_x_size != mat_x_dimension_) || (mat_w_size != mat_w_dimension_)) {
      MS_LOG(ERROR) << name_ << ": The dimensions of mat_x or mat_w's strategy is wrong. "
                    << "The length of strategy should equal the length of input. Current input x dimension is "
                    << mat_x_dimension_ << " input x strategy is " << ShapeToString(mat_x_strategy[i])
                    << " Current input w dimension is " << mat_w_dimension_ << " input w strategy is "
                    << ShapeToString(mat_w_strategy[i]);
      return FAILED;
    }

    if (mat_x_strategy[i].back() != mat_w_strategy[i].at(SECOND_FROM_END(mat_w_size))) {
      // for example: mat_x_strategy[i]:[2,4,16], mat_w_strategy[i]:[8,16,32], [16] in the example
      MS_LOG(ERROR) << name_ << ": Invalid strategy for mat_x " << ShapeToString(mat_x_strategy[i]) << " and mat_w "
                    << ShapeToString(mat_w_strategy[i]) << ". The shard num of first input's column is "
                    << mat_x_strategy[i].back() << ", but the shard num of second input's row is "
                    << mat_w_strategy[i].at(SECOND_FROM_END(mat_w_size));
      return FAILED;
    }
  }
  return SUCCESS;
}

// 3.devmatrix
Status GroupedMatmulInfo::InferDevMatrixShape() {
  NewStrategies stra = strategy_->GetInputNewDim();
  auto mat_x_strategy = (stra.at(0)->GetAllElements())[0];
  auto mat_w_strategy = (stra.at(1)->GetAllElements())[0];

  Shape common_shape;
  size_t mat_x_size = mat_x_strategy.size();
  size_t mat_w_size = mat_w_strategy.size();

  // mat_x_strategy: [bs, N,  h] or [N, h]
  // mat_b_strategy: [E,  h, 4h] or [h, 4h]
  // cur dev_matrix_shape: [bs, N] or [N],  only use mat_x_stra
  for (size_t i = 0; i < mat_x_size - 1; i++) {
    common_shape.push_back(mat_x_strategy.at(i));  // [2] in the example
  }

  // mat_x_strategy: [bs, N, h] or [N, h]   x   mat_b_strategy: [E, h, 4h] or [h, 4h]
  // cur dev_matrix_shape: [bs, N, h, 4h] or [N, h, 4h]
  common_shape.push_back(mat_w_strategy.at(SECOND_FROM_END(mat_w_size)));
  common_shape.push_back(mat_w_strategy.back());

  dev_matrix_shape_ = common_shape;
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  MS_LOG(INFO) << name_ << ": The dev matrix shape is " << dev_matrix_shape_;
  return SUCCESS;
}

void GroupedMatmulInfo::SetOptionalInputTensorMap(const size_t &index, size_t *valid_input_index) {
  MS_EXCEPTION_IF_NULL(valid_input_index);
  if (input_value_[index] != nullptr && !input_value_[index]->isa<None>()) {
    MS_EXCEPTION_IF_NULL(inputs_shape_new_[*valid_input_index]);
    auto input_shape = inputs_shape_new_[*valid_input_index]->GetElement(0);
    Shape nosplit_tensor_map_idx;
    if (input_shape->size() == 1) {
      nosplit_tensor_map_idx.emplace_back(0);  // {0}
    } else if (input_shape->size() == gmmTensor2D) {
      nosplit_tensor_map_idx.emplace_back(-1);
      nosplit_tensor_map_idx.emplace_back(0);  // {-1, 0}
    } else {
      MS_EXCEPTION(ShapeError) << "op [" << name_
                               << " ] set infer tensor map error. Current input_value_ size: " << input_value_.size()
                               << ", inputs_shape_new_ size: " << inputs_shape_new_.size()
                               << ". Current input_value_ idx is: " << index
                               << ", inputs_shape_new_ index is: " << *valid_input_index << ". inputs_shape_new_ ["
                               << *valid_input_index << "] size is: " << input_shape->size();
    }
    std::vector<ShapeBasePtr> nosplit_tensorlist_map_idx;
    for (size_t i = 0; i < inputs_shape_new_[*valid_input_index]->size(); i++) {
      nosplit_tensorlist_map_idx.emplace_back(std::make_shared<ShapeValue>(nosplit_tensor_map_idx));
    }
    inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(nosplit_tensorlist_map_idx));
    (*valid_input_index)++;
  }
}

// 4.TensorMap
Status GroupedMatmulInfo::InferTensorMap() {
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  // origin_dev_matrix_shape_: [bs, N, h, 4h] or [N, h, 4h]
  //                            3   2  1   0      2  1  0
  auto size = origin_dev_matrix_shape_.size();

  // x: [bs, N, h] or [N, h] --> {3, 2, 1} or {2, 1}
  Shape x_tensor_map_idx;
  for (size_t i = size - 1; i >= 1; i--) {
    x_tensor_map_idx.emplace_back(i);
  }
  std::vector<ShapeBasePtr> x_tensorist_map_idx;
  for (size_t i = 0; i < inputs_shape_new_[kInputX]->size(); i++) {
    x_tensorist_map_idx.emplace_back(std::make_shared<ShapeValue>(x_tensor_map_idx));
  }
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(x_tensorist_map_idx));

  // weight: [h, 4h] --> {1, 0} [E, h, 4h] --> {-1, 1, 0}
  Shape weight_tensor_map_idx;
  if (inputs_shape_new_[kInputWeight]->GetElement(0)->size() == gmmTensor3D) {
    weight_tensor_map_idx.emplace_back(-1);
  }
  weight_tensor_map_idx.emplace_back(1);
  weight_tensor_map_idx.emplace_back(0);
  std::vector<ShapeBasePtr> weight_tensorlist_map_idx;
  for (size_t i = 0; i < inputs_shape_new_[kInputWeight]->size(); i++) {
    weight_tensorlist_map_idx.emplace_back(std::make_shared<ShapeValue>(weight_tensor_map_idx));
  }
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(weight_tensorlist_map_idx));

  // bias: [4h] --> {0} / [E, 4h] --> {-1, 0} / not emplace_back shape when b is None
  if (!input_value_[kInputBias]->isa<None>()) {
    Shape bias_tensor_map_idx;
    if (inputs_shape_new_[kInputBias]->GetElement(0)->size() == gmmTensor2D) {
      bias_tensor_map_idx.emplace_back(-1);
    }
    bias_tensor_map_idx.emplace_back(0);
    std::vector<ShapeBasePtr> bias_tensorlist_map_idx;
    for (size_t i = 0; i < inputs_shape_new_[kInputBias]->size(); i++) {
      bias_tensorlist_map_idx.emplace_back(std::make_shared<ShapeValue>(bias_tensor_map_idx));
    }
    inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(bias_tensorlist_map_idx));
  }

  size_t valid_input_index = kInputScale;

  SetOptionalInputTensorMap(kInputScale, &valid_input_index);
  SetOptionalInputTensorMap(kInputOffset, &valid_input_index);
  SetOptionalInputTensorMap(kInputAntiquantScale, &valid_input_index);
  SetOptionalInputTensorMap(kInputAntiquantOffset, &valid_input_index);

  // optional grouplist
  // input_value_[kInputGroupList] != nullptr && input_value_[index]->isa<None>()
  if (input_value_[kInputGroupList] == nullptr) {
    Shape grouplist_tensor_map_idx{-1};
    inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(grouplist_tensor_map_idx));
  }

  // origin_dev_matrix_shape_: [bs, N, h, 4h] or [N, h, 4h]
  // out: [bs, N, 4h] or [N, 4h] --> {3, 2} or {2}
  Shape out_tensor_map_idx;
  for (size_t i = size - 1; i > 1; i--) {
    out_tensor_map_idx.emplace_back(i);
  }
  // out: [bs, N, 4h] or [N, 4h] --> {3, 2, 0} or {2, 0}
  out_tensor_map_idx.emplace_back(0);

  // split all output
  for (size_t i = 0; i < outputs_shape_new_.size(); i++) {
    outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(out_tensor_map_idx));
  }

  return SUCCESS;
}

TensorInfoBasePtr GroupedMatmulInfo::CreateTensorInfo(const Shape &device_matrix, const ShapeBasePtr &inputs_shape,
                                                      const ShapeBasePtr &inputs_tensor_map) {
  TensorInfoBasePtr out_tensor_info;
  if (inputs_shape->is_list()) {
    std::vector<TensorInfoBasePtr> tensor_info_list;
    for (int64_t i = 0; i < SizeToLong(inputs_shape->size()); ++i) {
      auto tensor_map = inputs_tensor_map->GetElement(i);
      auto shape = inputs_shape->GetElement(i);
      auto input_tensor_info = CreateTensorInfo(device_matrix, shape, tensor_map);
      tensor_info_list.emplace_back(input_tensor_info);
    }
    out_tensor_info = std::make_shared<TensorInfoList>(tensor_info_list);
  } else {
    TensorLayout input_layout;
    input_layout.InitFromVector(device_matrix, inputs_tensor_map->GetValue(), inputs_shape->GetValue());
    TensorInfo input_tensor_info(input_layout);
    out_tensor_info = std::make_shared<TensorInfoValue>(input_tensor_info);
  }
  return out_tensor_info;
}

Status GroupedMatmulInfo::InferTensorInfo() {
  if (inputs_shape_new_.empty()) {
    return FAILED;
  } else {
    size_t real_input_index = 0;
    for (size_t i = 0; i < inputs_shape_new_.size(); ++i) {
      // Insert placeholder TensorInfo for optional input
      while (real_input_index < input_value_.size() && input_value_[real_input_index] != nullptr &&
             input_value_[real_input_index]->isa<None>()) {
        (void)inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(TensorInfo()));
        ++real_input_index;
      }
      auto input_tensor_info =
        CreateTensorInfo(origin_dev_matrix_shape_, inputs_shape_new_[i], inputs_tensor_map_new_[i]);
      inputs_tensor_info_new_.emplace_back(input_tensor_info);
      ++real_input_index;
    }

    for (size_t i = 0; i < outputs_tensor_map_new_.size(); ++i) {
      auto output_tensor_info =
        CreateTensorInfo(origin_dev_matrix_shape_, outputs_shape_new_[i], outputs_tensor_map_new_[i]);
      outputs_tensor_info_new_.emplace_back(output_tensor_info);
    }
  }
  return SUCCESS;
}

Status GroupedMatmulInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_new_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_new_[0]->empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  if (out_dev_matrix_shape_.empty()) {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ =
    ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_new_[0]->GetAllElements()[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_new_[0]->GetAllElements()[0])
               << ", loss divisor is " << as_loss_divisor_;
  return SUCCESS;
}

Status GroupedMatmulInfo::InferForwardCommunication() {
  if (is_layout_config_) {
    return SUCCESS;
  }

  forward_op_.clear();

  size_t dimension = origin_dev_matrix_shape_.size();
  size_t relevant_dimension_index = SECOND_FROM_END(dimension);
  // Get N axis
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

  MS_LOG(INFO) << name_ << ": Need push_back op num: " << outputs_shape_new_.size();
  for (size_t i = 0; i < outputs_shape_new_.size(); i++) {
    Operator op;
    op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
    forward_op_.push_back(op);
  }

  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

REGISTER(GroupedMatmulInfo);
}  // namespace parallel
}  // namespace mindspore
