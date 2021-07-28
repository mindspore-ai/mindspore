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

#include "frontend/parallel/ops_info/virtual_dataset_info.h"

#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status VirtualDatasetInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  if (stra.size() < 1) {
    MS_LOG(ERROR) << name_ << ": Strategy size must be larger than 1.";
    return FAILED;
  }
  used_devices_ = int64_t(std::accumulate(stra[0].begin(), stra[0].end(), 1, std::multiplies<int64_t>()));
  for (size_t i = 0; i < stra.size(); ++i) {
    bool find_shard_dim = false;
    int64_t current_stra_shard_num = 1;
    for (auto dim : stra[i]) {
      if (dim == 1) {
        continue;
      }
      if (find_shard_dim) {
        MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
        return FAILED;
      } else {
        find_shard_dim = true;
        current_stra_shard_num = dim;
      }
    }
    if (i == 0) {
      shard_num_ = current_stra_shard_num;
    } else if (current_stra_shard_num != 1 && current_stra_shard_num != shard_num_) {
      MS_LOG(ERROR) << name_
                    << ": For each dataset input, the shard strategy can be not shard, "
                       "or shard in one dim with the same shard size between each input. "
                       "Current shard size is: "
                    << current_stra_shard_num << ". The previous shard size is " << shard_num_;
      return FAILED;
    }
    if (stra[i].size() > stra[max_size_strategy_dim_].size()) {
      max_size_strategy_dim_ = i;
    }
  }
  if (std::find(stra[max_size_strategy_dim_].begin(), stra[max_size_strategy_dim_].end(), shard_num_) ==
      stra[max_size_strategy_dim_].end()) {
    MS_LOG(ERROR) << name_
                  << ": For each dataset input, the shard strategy can be not shard, "
                     "or shard in one dim with the same shard size between each input."
                     " If using shard, the max length input must be shard, "
                     "but the strategy of the max length input is: "
                  << stra[max_size_strategy_dim_];
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra[max_size_strategy_dim_];
  return SUCCESS;
}

Status VirtualDatasetInfo::InferMirrorOps() { return SUCCESS; }

Status VirtualDatasetInfo::InferForwardCommunication() { return SUCCESS; }

Status VirtualDatasetInfo::InferTensorMap() {
  auto slice_dim_iter = std::find(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), shard_num_);
  if (slice_dim_iter == dev_matrix_shape_.end()) {
    MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
    return FAILED;
  }
  size_t slice_dim = size_t(slice_dim_iter - dev_matrix_shape_.begin());
  auto stra = strategy_->GetInputDim();
  for (size_t i = 0; i < stra.size(); i++) {
    Shape tensor_map_index;
    for (auto dim : stra[i]) {
      if (dim == 1) {
        tensor_map_index.push_back(MAP_NONE);
      } else if (dim == shard_num_) {
        tensor_map_index.push_back(dev_matrix_shape_.size() - 1 - slice_dim);
      } else {
        MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
        return FAILED;
      }
    }
    inputs_tensor_map_.push_back(tensor_map_index);
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::GetAttrs() { return SUCCESS; }

Status VirtualDatasetInfo::Init(const StrategyPtr &strategy) {
  repeated_num_in_dev_matrix_right_ = false;
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
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

Status VirtualDatasetInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status VirtualDatasetInfo::GenerateStrategies(int64_t stage_id) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  StrategyPtr sp;
  Strategys strategy;
  if (!ParallelContext::GetInstance()->dataset_strategy().empty()) {
    strategy = ParallelContext::GetInstance()->dataset_strategy();
  } else {
    bool full_batch = ParallelContext::GetInstance()->full_batch();
    size_t total_dev_num;
    if (full_batch) {
      total_dev_num = 1;
    } else {
      total_dev_num = stage_device_size_;
    }
    for (auto &shape : inputs_shape_) {
      Shape temp;
      temp.emplace_back(SizeToLong(total_dev_num));
      (void)temp.insert(temp.end(), shape.size() - 1, 1);
      strategy.push_back(temp);
    }
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);
  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << ": Successfully dataset strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << ": Generating dataset strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}

std::vector<StrategyPtr> VirtualDatasetInfo::GenerateOpStrategies(int64_t) {
  std::vector<StrategyPtr> sp_vector;
  return sp_vector;
}

Status VirtualDatasetInfo::InferAsLossDivisor() {
  // no need to insert div op
  as_loss_divisor_ = 1;
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
