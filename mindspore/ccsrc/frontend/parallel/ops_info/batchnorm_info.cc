/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/batchnorm_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "utils/ms_context.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status BatchNormInfo::GetAttrs() {
  is_training_ = GetBoolAttr(IS_TRAINING);

  epsilon_ = GetFloatAttr(EPSILON);

  momentum_ = GetFloatAttr(MOMENTUM);

  format_ = GetStringAttr(FORMAT);
  if (format_ != NCHW) {
    MS_LOG(ERROR) << name_ << ": The data format must be 'NCHW', but got " << format_;
    return FAILED;
  }

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  if (inputs_shape_[0].size() == 2) {
    input_is_4d_ = false;
  } else if (inputs_shape_[0].size() == 4) {
    input_is_4d_ = true;
  } else {
    MS_LOG(ERROR) << name_ << ": The size of input[0]'shape must be 2 or 4, but got " << inputs_shape_[0].size();
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The is_traing is " << is_training_ << ", epsilon is " << epsilon_ << ", momentum is "
               << momentum_ << ", data format is " << format_;

  return SUCCESS;
}

Status BatchNormInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 5, but got " << stra.size();
    return FAILED;
  }

  if ((stra[0].size() != 4) && (stra[0].size() != 2)) {
    MS_LOG(ERROR) << name_ << ": The size of strategy[0] must be 4 or 2, but got " << stra[0].size();
    return FAILED;
  }

  for (size_t i = 1; i < 5; ++i) {
    if (stra[i].empty()) {
      MS_LOG(ERROR) << name_ << ": The strategy can not be empty, the index is " << i;
      return FAILED;
    }
    if (stra[0][1] != stra[i][0]) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy, the index is " << i << ", it must be equal to " << stra[0][1]
                    << ", but got " << stra[i][0];
      return FAILED;
    }
  }

  return SUCCESS;
}

Status BatchNormInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy can not be empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status BatchNormInfo::InferTensorMap() {
  TensorMap input_tensor_map;
  TensorMap in_other_tensor_map;

  if (input_is_4d_) {
    // if input is 4d:
    // input_strategy: ((n, c, h, w), (c), (c), (c), (c))
    // output_strategy: ((n, c, h, w), (c), (c), (c), (c))
    // dev_matrix: (n, c, h, w)
    input_tensor_map = {3, 2, 1, 0};
    in_other_tensor_map = {2};
  } else {
    // if input is 2d:
    // input_strategy: ((n, c), (c), (c), (c), (c))
    // output_strategy: ((n, c), (c), (c), (c), (c))
    // dev_matrix: (n, c)
    input_tensor_map = {1, 0};
    in_other_tensor_map = {0};
  }

  inputs_tensor_map_.push_back(input_tensor_map);     // input
  inputs_tensor_map_.push_back(in_other_tensor_map);  // scale
  inputs_tensor_map_.push_back(in_other_tensor_map);  // bias
  inputs_tensor_map_.push_back(in_other_tensor_map);  // mean
  inputs_tensor_map_.push_back(in_other_tensor_map);  // variance

  outputs_tensor_map_ = inputs_tensor_map_;
  return SUCCESS;
}

Status BatchNormInfo::InferForwardCommunication() {
  // if it is not training, no need forward allreduce
  if (!is_training_) {
    MS_LOG(INFO) << name_ << ": It is not training, no need forward allreduce";
    return SUCCESS;
  }

  TensorMap tmp_map;
  if (input_is_4d_) {
    // input is 4d:
    // if has not repeated calculation, the dev matirx is [n, c, h, w]
    // if repeated calculation and repeated num in the left of dev matrix, the dev matrix is [repeated_num, n, c, h, w]
    // if repeated calculation and repeated num in the right of dev matrix, the dev matrix is [n, c, h, w, repeated_num]
    // and the forward allreduce need to use the dimensions of n/h/w
    if (repeated_calc_num_ == 1) {
      // has not repeated calculation
      tmp_map = {-1, 2, -1, -1};
    } else if (!repeated_num_in_dev_matrix_right_) {
      // repeated calculation and repeated num in the left of dev matrix
      tmp_map = {4, -1, 2, -1, -1};
    } else {
      // repeated calculation and repeated num in the right of dev matrix
      tmp_map = {-1, 3, -1, -1, 0};
    }
  } else {
    // input is 2d:
    // if has not repeated calculation, the dev matirx is [n, c]
    // if repeated calculation and repeated num in the left of dev matrix, the dev matrix is [repeated_num, n, c]
    // if repeated calculation and repeated num in the right of dev matrix, the dev matrix is [n, c, repeated_num]
    // and the forward allreduce need to use the dimensions of n
    if (repeated_calc_num_ == 1) {
      // has not repeated calculation
      tmp_map = {-1, 0};
    } else if (!repeated_num_in_dev_matrix_right_) {
      // repeated calculation and repeated num in the left of dev matrix
      tmp_map = {2, -1, 0};
    } else {
      // repeated calculation and repeated num in the right of dev matrix
      tmp_map = {-1, 1, 0};
    }
  }

  std::vector<Group> group_list;
  if (CreateGroupByTensorMap(tmp_map, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed";
    return FAILED;
  }

  if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  } else {
    MS_LOG(INFO) << name_ << ": The group name of forward all reduce is " << group_list[0].name();
  }

  forward_allreduce_group_ = group_list;
  return SUCCESS;
}

Status BatchNormInfo::InferReplaceOps() {
  replace_op_.clear();

  if (!is_training_) {
    MS_LOG(INFO) << name_ << ": It is not training, no need to replace op";
    return SUCCESS;
  }

  if (forward_allreduce_group_.empty()) {
    MS_LOG(INFO) << name_ << ": The forward allreduce group is empty, no need to replace op";
    return SUCCESS;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  if (backend != kAscendDevice && backend != kDavinciDevice) {
    MS_LOG(INFO) << name_ << ": The backend is " << backend << ", it does not support SyncBatchNorm operator";
    return SUCCESS;
  }

  ValuePtr epsilon = MakeValue(epsilon_);
  ValuePtr momentum = MakeValue(momentum_);
  ValuePtr group = MakeValue(forward_allreduce_group_[0].name());
  ValuePtr device_num = MakeValue(forward_allreduce_group_[0].GetDevNum());

  Attr attr_epsilon = std::make_pair(EPSILON, epsilon);
  Attr attr_momentum = std::make_pair(MOMENTUM, momentum);
  Attr attr_group = std::make_pair(GROUP, group);
  Attr attr_device_num = std::make_pair(DEVICE_NUM, device_num);

  OperatorAttrs attrs = {attr_epsilon, attr_momentum, attr_group, attr_device_num};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(SYNC_BATCH_NORM, args)};
  return SUCCESS;
}

Status BatchNormInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 5, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status BatchNormInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> BatchNormInfo::GenerateOpStrategies(int64_t stage_id) {
  Strategys strategy;
  if (input_is_4d_) {
    strategy = {{stage_device_size_, 1, 1, 1}, {1}, {1}, {1}, {1}};
  } else {
    strategy = {{stage_device_size_, 1}, {1}, {1}, {1}, {1}};
  }
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status BatchNormInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }

  (void)InferReplaceOps();
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status BatchNormInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
