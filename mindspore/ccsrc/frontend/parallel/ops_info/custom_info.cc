/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/custom_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
constexpr char KEmptyMirrorOpNum[] = "empty_mirror_ops";

Status CustomInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status CustomInfo::InferAsLossDivisor() {
  auto as_loss_divisor_iter = attrs_.find(KAttrAsLossDivisor);
  if (as_loss_divisor_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(as_loss_divisor_iter->second);
    as_loss_divisor_ = GetValue<int64_t>(as_loss_divisor_iter->second);

    if (as_loss_divisor_ > 0) {
      return SUCCESS;
    } else {
      MS_LOG(WARNING) << name_ << ": the input of as_loss_divisor from attrs is non-negative. "
                      << "This input will be ignored and "
                      << "as_loss_divisor will be computed from dev_matrix_shape and outputs_tensor_map";
    }
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status CustomInfo::InferDevMatrixShape() {
  auto dev_matrix_shape_iter = attrs_.find(KAttrDevMatrixShape);
  if (dev_matrix_shape_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(dev_matrix_shape_iter->second);
    dev_matrix_shape_ = GetValue<std::vector<int64_t>>(dev_matrix_shape_iter->second);
  }
  if (dev_matrix_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": can't find dev matrix shape from attrs in Custom Op";
    return FAILED;
  }
  return SUCCESS;
}

Status CustomInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  auto empty_mirror_ops_iter = attrs_.find(KEmptyMirrorOpNum);
  if (empty_mirror_ops_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(empty_mirror_ops_iter->second);
    auto empty_mirror_ops_ = GetValue<int64_t>(empty_mirror_ops_iter->second);
    for (int64_t idx = 0; idx < empty_mirror_ops_; idx++) {
      (void)mirror_ops_.emplace_back(OperatorVector());
    }
  }
  return SUCCESS;
}

Status CustomInfo::InferTensorMap() {
  auto inputs_tensor_map_iter = attrs_.find(KAttrInputsTensorMap);
  if (inputs_tensor_map_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(inputs_tensor_map_iter->second);
    inputs_tensor_map_ = GetValue<std::vector<std::vector<int64_t>>>(inputs_tensor_map_iter->second);
  }

  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": can't find inputs tensor map from attrs in Custom Op";
    return FAILED;
  }

  auto outputs_tensor_map_iter = attrs_.find(KAttrOutputsTensorMap);
  if (outputs_tensor_map_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(outputs_tensor_map_iter->second);
    outputs_tensor_map_ = GetValue<std::vector<std::vector<int64_t>>>(outputs_tensor_map_iter->second);
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": can't find outputs tensor map from attrs in Custom Op";
    return FAILED;
  }

  return SUCCESS;
}

std::vector<StrategyPtr> CustomInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  return sp_vector;
}

REGISTER(CustomInfo);
}  // namespace parallel
}  // namespace mindspore
