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

#include "parallel/ops_info/virtual_dataset_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "parallel/device_manager.h"
#include "parallel/device_matrix.h"
#include "parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status VirtualDatasetInfo::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() < 1) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Strategy size must be larger than 1.";
    } else {
      MS_LOG(ERROR) << name_ << ": Strategy size must be larger than 1.";
    }
    return FAILED;
  }
  if (stra.size() == 1) {
    MS_LOG(WARNING) << name_ << ": Strategy size is 1.";
    return SUCCESS;
  }
  Dimensions strategy_first = stra.at(1);
  for (auto iter_strategy = stra.begin() + 1; iter_strategy != stra.end(); ++iter_strategy) {
    if (iter_strategy->empty()) {
      MS_LOG(ERROR) << name_ << ": iter_strategy size is zero.";
    }
    if (strategy_first.at(0) != *(iter_strategy->begin())) {
      if (is_auto_parallel_) {
        MS_LOG(DEBUG) << name_ << ": The first dimension of each strategy must be the same.";
      } else {
        MS_LOG(ERROR) << name_ << ": The first dimension of each strategy must be the same.";
      }
      return FAILED;
    }

    for (auto iter_element = iter_strategy->begin() + 1; iter_element != iter_strategy->end(); ++iter_element) {
      if (*iter_element != 1) {
        if (is_auto_parallel_) {
          MS_LOG(DEBUG) << name_ << ": All dimension except the first dimension of each strategy must be 1.";
        } else {
          MS_LOG(ERROR) << name_ << ": All dimension except the first dimension of each strategy must be 1.";
        }
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions strategy_first = stra.at(0);
  int32_t stage = strategy_->GetInputStage();
  CheckGlobalDeviceManager();
  int32_t dev_num = SizeToInt(g_device_manager->GetDeviceListByStageId(stage).size());
  int32_t batch_split_num = strategy_first.at(0);
  dev_matrix_shape_.push_back(batch_split_num);
  if (dev_num > batch_split_num) {
    dev_matrix_shape_.push_back(dev_num / batch_split_num);
  }

  return SUCCESS;
}

Status VirtualDatasetInfo::InferMirrorOps() {
  mirror_ops_.clear();

  int32_t stage = strategy_->GetInputStage();
  CheckGlobalDeviceManager();
  RankList dev_list = g_device_manager->GetDeviceListByStageId(stage);
  if (dev_list.empty()) {
    MS_LOG(ERROR) << name_ << ": The current stage is empty!";
    return Status::FAILED;
  }
  if (dev_list.size() == 1) {
    MS_LOG(INFO) << name_ << ": No need mirror ops.";
    return Status::SUCCESS;
  }

  OperatorName operator_name = BROADCAST;
  ValuePtr attr0_value = MakeValue(dev_list.front());
  std::vector<Group> group_list;
  if (CreateGroupByDim(dev_matrix_shape_.size() - 1, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer mirror ops, create group failed.";
    return FAILED;
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": No need mirror ops.";
    return SUCCESS;
  }
  std::string group = group_list[0].name();
  ValuePtr attr1_value = MakeValue(group);

  Attr attr0 = std::make_pair(SRC, attr0_value);
  Attr attr1 = std::make_pair(GROUP, attr1_value);

  OperatorAttrs operator_attrs = {attr0, attr1};

  OperatorParams operator_param;
  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_args);
  OperatorVector op_vector = {op};

  size_t size = inputs_shape_.size();
  for (size_t i = 0; i < size; ++i) {
    mirror_ops_.push_back(op_vector);
  }
  mirror_ops_.clear();
  return SUCCESS;
}

Status VirtualDatasetInfo::InferForwardCommunication() { return SUCCESS; }

Status VirtualDatasetInfo::InferTensorMap() {
  for (size_t i = 0; i < strategy_->GetInputNumber(); i++) {
    std::vector<int32_t> tensor_map_index;
    tensor_map_index.push_back((int32_t)(LAST_INDEX(SizeToUint(dev_matrix_shape_.size()))));
    for (size_t j = 1; j < strategy_->GetInputDim()[i].size(); ++j) {
      tensor_map_index.push_back(MAP_NONE);
    }
    inputs_tensor_map_.push_back(tensor_map_index);
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferTensorInfo() {
  for (size_t i = 0; i < strategy_->GetInputNumber(); i++) {
    MS_LOG(INFO) << name_ << ": InferTensorInfo " << i << ",  size " << strategy_->GetInputNumber();
    TensorLayout tensor_layout_in;
    if (tensor_layout_in.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(i), inputs_shape_.at(i)) != SUCCESS) {
      return FAILED;
    }
    TensorInfo tensor_info_in(tensor_layout_in);
    inputs_tensor_info_.push_back(tensor_info_in);
    outputs_tensor_info_.push_back(tensor_info_in);
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::GetAttrs() { return SUCCESS; }

Status VirtualDatasetInfo::Init(const StrategyPtr& strategy) {
  if (InitWithManualRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithManualRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

void VirtualDatasetInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = true;
  }
}

Status VirtualDatasetInfo::SetCostUnderStrategy(const StrategyPtr& strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status VirtualDatasetInfo::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GetAttrs failed";
    return FAILED;
  }

  CheckGlobalDeviceManager();
  is_auto_parallel_ = true;
  size_t total_dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  StrategyPtr sp;
  std::vector<Dimensions> strategy;
  for (auto& shape : inputs_shape_) {
    Shape temp;
    temp.emplace_back(SizeToInt(total_dev_num));
    (void)temp.insert(temp.end(), shape.size() - 1, 1);
    strategy.push_back(temp);
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);

  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << ": Successfully generated batch-parallel-strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << ": Generating batch-parallel-strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferAsLossDivisor() {
  // no need to insert div op
  as_loss_divisor_ = 1;
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
