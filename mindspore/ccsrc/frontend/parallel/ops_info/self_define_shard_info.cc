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

#include "frontend/parallel/ops_info/self_define_shard_info.h"

#include <memory>
#include <utility>
#include <algorithm>

#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
Status SelfDefineShardInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_LOG(ERROR) << "self define shard op " << name_ << " only support config layout rather than strategy";
  return FAILED;
}

Status SelfDefineShardInfo::UnreachableError() {
  MS_LOG(ERROR) << "For self define shard op " << name_ << ", it should not reach this function";
  return FAILED;
}

Status SelfDefineShardInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return UnreachableError(); }

std::vector<StrategyPtr> SelfDefineShardInfo::GenerateOpStrategies(int64_t stage_id) {
  MS_LOG(EXCEPTION) << "For self define shard op " << name_ << ", it should not reach this function";
}

Status SelfDefineShardInfo::CheckLayout(const Shapes &in_shapes, const std::vector<TensorInfo> &tensor_info,
                                        const string &name) {
  // Check the number of input shape and input layout
  if (tensor_info.size() != in_shapes.size()) {
    MS_LOG(ERROR) << "The " << name << " shape of " << name_ << " is " << in_shapes.size()
                  << ", which is not equal to the input tensor layout size " << tensor_info.size();
  }
  // Check the device matrix
  auto prev_dev_arrangment = tensor_info.at(kIndex0).tensor_layout().device_arrangement_origin().array();
  for (size_t i = 1; i < tensor_info.size(); ++i) {
    auto current_tensor_layout = tensor_info.at(i).tensor_layout();
    if (prev_dev_arrangment != current_tensor_layout.device_arrangement_origin().array()) {
      MS_LOG(ERROR) << "The device_matrix of input " << i << " is "
                    << current_tensor_layout.device_arrangement_origin().array() << ", which is not equal to previous "
                    << name << " device_matrix " << prev_dev_arrangment;
      return FAILED;
    }
    prev_dev_arrangment = current_tensor_layout.device_arrangement_origin().array();
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::CheckInputLayout() {
  // Check self_define_shard attribute
  MS_LOG(WARNING) << "Use self define shard for " << name_
                  << ". User needs to ensure the accuracy and correctness of input/output layout, and framework "
                     "will only do basic check.";
  if (CheckLayout(inputs_shape_, inputs_tensor_info_, "input") != SUCCESS) {
    MS_LOG(ERROR) << name_ << " check input layout failed";
    return FAILED;
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::CheckOutputLayout() {
  if (CheckLayout(outputs_shape_, outputs_tensor_info_, "output") != SUCCESS) {
    MS_LOG(ERROR) << name_ << " check input layout failed";
    return FAILED;
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::InferOutputTensorInfo() {
  MS_LOG(ERROR) << "Please pass output layout to " << name_
                << ", self define shard ops does not support infer output tensor layout";
  return FAILED;
}

Status SelfDefineShardInfo::InferAsLossDivisorByLayout() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_info_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor info is empty.";
    return FAILED;
  }

  TensorMaps outputs_tensor_map = outputs_tensor_info_[0].tensor_layout().tensor_map_before();
  if (outputs_tensor_map.empty()) {
    MS_LOG(INFO) << name_ << ": out_dev_matrix_shape is empty";
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = outputs_tensor_info_[0].tensor_layout().device_arrangement_origin().array();
  if (out_dev_matrix_shape.empty()) {
    out_dev_matrix_shape = dev_matrix_shape_;
  }
  Shape squashed_tensor_map;
  for (const auto &tensor_map : outputs_tensor_map) {
    std::copy(tensor_map.begin(), tensor_map.end(), std::back_inserter(squashed_tensor_map));
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape, squashed_tensor_map);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape)
               << ", the output tensor map is " << ShapeToString(squashed_tensor_map) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

REGISTER(SelfDefineShardInfo);
}  // namespace parallel
}  // namespace mindspore
