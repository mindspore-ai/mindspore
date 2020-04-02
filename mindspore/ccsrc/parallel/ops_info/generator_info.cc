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

#include "parallel/ops_info/generator_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status GeneratorBase::InferTensorMap() {
  TensorMap output_tensor_map = {MAP_NONE};
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status GeneratorBase::InferTensorInfo() {
  Shape output_shape = outputs_shape_.at(0);
  Shape output_slice_shape = outputs_shape_.at(0);

  TensorLayout output_tensor_layout;
  if (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Creat output tensor layout failed.";
    return FAILED;
  }
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);
  outputs_tensor_info_.push_back(output_tensor_info);

  return SUCCESS;
}

Status GeneratorBase::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

Status GeneratorBase::SetCostUnderStrategy(const StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status DropoutGenMaskInfo::GenerateStrategies(int32_t stage_id) {
  if (input_value_.empty()) {
    MS_LOG(ERROR) << name_ << " : Input value is empty.";
    return FAILED;
  }
  Shape param = GetValue<const std::vector<int>>(input_value_[0]);
  if (param.empty()) {
    MS_LOG(ERROR) << name_ << " : Input value [0] is empty.";
    return FAILED;
  }
  // Now,only support batch parallel.
  CheckGlobalDeviceManager();
  is_auto_parallel_ = true;
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  Dimensions strategy(param.size() - 1, 1);
  (void)strategy.insert(strategy.begin(), SizeToInt(dev_num));
  std::vector<Dimensions> stra = {strategy};
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, stra);
  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << " : Successfully generated batch-parallel-strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << " : Generating batch-parallel-strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status DropoutGenMaskInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (strategy->GetInputNumber() != 1) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : The strategy is wrong.";
    } else {
      MS_LOG(ERROR) << name_ << " : The strategy is wrong.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status DropoutGenMaskInfo::InferReplaceOps(const StrategyPtr &strategy) {
  Shape shape = GetValue<const std::vector<int>>(input_value_[0]);
  Strategys stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  int32_t dev_num = *(input_strategy.begin());
  if (dev_num <= 0) {
    MS_LOG(ERROR) << name_ << " : The number of devices should not be less than 0.";
    return FAILED;
  }
  // Batch parallel
  if (shape[0] % dev_num != 0) {
    MS_LOG(ERROR) << name_ << " : The shape " << shape[0] << " can't be exact divided by device number " << dev_num;
    return FAILED;
  }
  shape[0] = shape[0] / dev_num;
  ValuePtr shape_ptr = MakeValue(shape);
  Attr attr_0 = std::make_pair(SEED0, attrs_[SEED0]);
  Attr attr_1 = std::make_pair(SEED1, attrs_[SEED1]);
  OperatorAttrs attrs = {attr_0, attr_1};
  Attr param_0 = std::make_pair(SHAPE, shape_ptr);
  Attr param_1 = std::make_pair(KEEP_PROB, input_value_[1]);
  OperatorParams params = {std::make_pair(param_0, 1), std::make_pair(param_1, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(DROPOUT_GEN_MASK, args)};
  return SUCCESS;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> DropoutGenMaskInfo::GenerateBatchStrategies() {
  if (input_value_.empty()) {
    MS_LOG(EXCEPTION) << name_ << " : Input value is empty.";
  }
  Shape param = GetValue<const std::vector<int>>(input_value_[0]);
  if (param.empty()) {
    MS_LOG(EXCEPTION) << name_ << " : Input value [0] is empty.";
  }
  // Now,only support batch parallel.
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  Dimensions strategy(param.size() - 1, 1);
  (void)strategy.insert(strategy.begin(), SizeToInt(dev_num));
  std::vector<Dimensions> strategy_v = {strategy};
  return std::make_shared<std::vector<std::vector<int32_t>>>(strategy_v);
}

Status GeneratorBase::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  if (InferReplaceOps(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer replace ops failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status GeneratorBase::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
