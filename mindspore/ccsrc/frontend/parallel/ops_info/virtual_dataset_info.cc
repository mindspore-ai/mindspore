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
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"
#include "include/common/utils/parallel_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {

Status VirtualDatasetInfo::GetSquashedStrategyAndShape(const StrategyPtr &stra,
                                                       std::vector<std::vector<int64_t>> *squashed_stra,
                                                       std::vector<std::vector<int64_t>> *squashed_shape) {
  if (stra == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }
  if (inputs_shape_new_.empty()) {
    *squashed_stra = stra->GetInputDim();
    *squashed_shape = inputs_shape_;
    if (squashed_stra->empty()) {
      MS_LOG(ERROR) << name_ << ": Strategy size must be larger than 1.";
      return FAILED;
    }
  } else {
    if (!stra->HasTupleInTupleStrategy()) {
      MS_LOG(ERROR) << name_ << ": The strategy must be tuple in tuple.";
      return FAILED;
    }
    NewStrategies new_stra = stra->GetInputNewDim();
    if (new_stra.empty()) {
      MS_LOG(ERROR) << name_ << ": Strategy size must be larger than 1.";
      return FAILED;
    }
    // Squash shapevalue and shapelist into one vector
    // ((1,2), ((1,2), (1,2))) -> ((1,2), (1,2), (1,2))
    for (size_t i = 0; i < new_stra.size(); ++i) {
      if (new_stra[i]->is_list() != inputs_shape_new_[i]->is_list()) {
        MS_LOG(ERROR) << name_ << ": The strategy and shape must be both list or both value.";
        return FAILED;
      }
      auto shape_element = inputs_shape_new_[i]->GetAllElements();
      auto stra_element = new_stra[i]->GetAllElements();
      squashed_stra->insert(squashed_stra->end(), stra_element.begin(), stra_element.end());
      squashed_shape->insert(squashed_shape->end(), shape_element.begin(), shape_element.end());
    }
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::CheckStrategy(const StrategyPtr &strategy) {
  std::vector<std::vector<int64_t>> squashed_stra;
  std::vector<std::vector<int64_t>> squashed_shape;
  if (GetSquashedStrategyAndShape(strategy, &squashed_stra, &squashed_shape) != SUCCESS) {
    return FAILED;
  }
  if (CheckStrategyByVector(squashed_stra, squashed_shape) != SUCCESS) {
    return FAILED;
  }
  used_devices_ =
    int64_t(std::accumulate(squashed_stra[0].begin(), squashed_stra[0].end(), 1, std::multiplies<int64_t>()));
  for (size_t i = 0; i < squashed_stra.size(); ++i) {
    bool find_shard_dim = false;
    int64_t current_stra_shard_num = 1;
    for (auto dim : squashed_stra[i]) {
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
    if (shard_num_ == 1) {
      shard_num_ = current_stra_shard_num;
    } else if (current_stra_shard_num != 1 && current_stra_shard_num != shard_num_) {
      MS_LOG(ERROR) << name_
                    << ": For each dataset input, the shard strategy can be not shard, "
                       "or shard in one dim with the same shard size between each input. "
                       "Current shard size is: "
                    << current_stra_shard_num << ". The previous shard size is " << shard_num_;
      return FAILED;
    }
    max_size_strategy_ = squashed_stra[max_size_strategy_dim_];
    if (squashed_stra[i].size() > squashed_stra[max_size_strategy_dim_].size()) {
      max_size_strategy_dim_ = i;
      max_size_strategy_ = squashed_stra[i];
    }
  }
  if (!squashed_stra[max_size_strategy_dim_].empty() &&
      std::find(squashed_stra[max_size_strategy_dim_].begin(), squashed_stra[max_size_strategy_dim_].end(),
                shard_num_) == squashed_stra[max_size_strategy_dim_].end()) {
    MS_LOG(ERROR) << name_
                  << ": For each dataset input, the shard strategy can be not shard, "
                     "or shard in one dim with the same shard size between each input."
                     " If using shard, the max length input must be shard, "
                     "but the strategy of the max length input is: "
                  << squashed_stra[max_size_strategy_dim_];
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferDevMatrixShape() {
  if (max_size_strategy_.empty()) {
    dev_matrix_shape_ = strategy_->GetInputDim()[max_size_strategy_dim_];
  } else {
    dev_matrix_shape_ = max_size_strategy_;
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferMirrorOps() { return SUCCESS; }

Status VirtualDatasetInfo::InferForwardCommunication() { return SUCCESS; }

ShapeBasePtr VirtualDatasetInfo::ObtainTensorMap(const ShapeBasePtr &stra, const size_t &slice_dim,
                                                 const Shape &dev_mat) {
  size_t dev_mat_size = dev_mat.size();
  if (stra->is_list()) {
    std::vector<ShapeBasePtr> tensor_map;
    for (size_t i = 0; i < stra->size(); i++) {
      tensor_map.emplace_back(ObtainTensorMap(stra->GetElement(SizeToLong(i)), slice_dim, dev_mat));
    }
    return std::make_shared<ShapeList>(tensor_map);
  }
  Shape tensor_map_index;
  for (auto dim : stra->GetValue()) {
    if (dim == 1) {
      tensor_map_index.push_back(MAP_NONE);
    } else if (dim == shard_num_) {
      if (repeated_calc_num_ > 1 && repeated_num_in_dev_matrix_right_ && is_auto_parallel_) {
        tensor_map_index.push_back(dev_mat_size - slice_dim);
      } else {
        tensor_map_index.push_back(dev_mat_size - 1 - slice_dim);
      }
    } else {
      MS_LOG(EXCEPTION) << name_ << ": The dataset shard strategy only support shard in one dim.";
    }
  }
  return std::make_shared<ShapeValue>(tensor_map_index);
}

Status VirtualDatasetInfo::InferTensorMapNew() {
  auto dev_mat_origin = max_size_strategy_;
  auto slice_dim_iter = std::find(dev_mat_origin.begin(), dev_mat_origin.end(), shard_num_);
  if (!dev_mat_origin.empty() && slice_dim_iter == dev_mat_origin.end()) {
    MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
    return FAILED;
  }
  size_t slice_dim = size_t(slice_dim_iter - dev_mat_origin.begin());
  auto stra = strategy_->GetInputNewDim();
  for (size_t i = 0; i < stra.size(); i++) {
    auto tensor_map = ObtainTensorMap(stra[i], slice_dim, dev_mat_origin);
    inputs_tensor_map_new_.push_back(tensor_map);
    outputs_tensor_map_new_.push_back(tensor_map);
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InferTensorMap() {
  if (!inputs_shape_new_.empty()) {
    return InferTensorMapNew();
  }
  auto dev_mat_origin = strategy_->GetInputDim()[max_size_strategy_dim_];
  auto slice_dim_iter = std::find(dev_mat_origin.begin(), dev_mat_origin.end(), shard_num_);
  if (!dev_mat_origin.empty() && slice_dim_iter == dev_mat_origin.end()) {
    MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
    return FAILED;
  }
  size_t slice_dim = size_t(slice_dim_iter - dev_mat_origin.begin());
  auto stra = strategy_->GetInputDim();
  for (size_t i = 0; i < stra.size(); i++) {
    Shape tensor_map_index;
    for (auto dim : stra[i]) {
      if (dim == 1) {
        tensor_map_index.push_back(MAP_NONE);
      } else if (dim == shard_num_) {
        if (repeated_calc_num_ > 1 && repeated_num_in_dev_matrix_right_ && is_auto_parallel_) {
          tensor_map_index.push_back(dev_mat_origin.size() - slice_dim);
        } else {
          tensor_map_index.push_back(dev_mat_origin.size() - 1 - slice_dim);
        }
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

Status VirtualDatasetInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
                                const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                                const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts) {
  repeated_num_in_dev_matrix_right_ = false;
  if (ParallelContext::GetInstance()->dataset_repeat_dim_right()) {
    repeated_num_in_dev_matrix_right_ = true;
  }
  if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status VirtualDatasetInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  is_auto_parallel_ = true;
  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }
  is_auto_parallel_ = false;
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

void VirtualDatasetInfo::ReComputeBatchSplitFlagList() {
  size_t inputs_shape_size;
  if (inputs_shape_new_.empty()) {
    inputs_shape_size = inputs_shape_.size();
  } else {
    inputs_shape_size = inputs_shape_new_.size();
  }
  for (size_t i = 0; i < inputs_shape_size; i++) {
    split_flag_list_[i] = true;
  }
}

Status VirtualDatasetInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> VirtualDatasetInfo::GenerateOpStrategies(int64_t stage_id) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  StrategyPtr sp;
  Strategies strategy;
  if (!ParallelContext::GetInstance()->dataset_strategy().empty()) {
    strategy = ParallelContext::GetInstance()->dataset_strategy();
  } else {
    bool full_batch = ParallelContext::GetInstance()->full_batch();
    int64_t total_dev_num;
    if (full_batch) {
      total_dev_num = 1;
    } else {
      total_dev_num = stage_device_size_;
    }
    if (inputs_shape_new_.empty()) {
      for (auto &shape : inputs_shape_) {
        Shape temp;
        if (!shape.empty()) {
          temp.emplace_back(total_dev_num);
          (void)temp.insert(temp.cend(), shape.size() - 1, 1);
        }
        strategy.push_back(temp);
      }
    } else {
      for (auto &shape : inputs_shape_new_) {
        Shape temp;
        if (!shape->empty()) {
          temp.emplace_back(total_dev_num);
          (void)temp.insert(temp.cend(), shape->GetBatchValue() - 1, 1);
        }
        strategy.push_back(temp);
      }
    }
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status VirtualDatasetInfo::InferAsLossDivisor() {
  // no need to insert div op
  as_loss_divisor_ = 1;
  return SUCCESS;
}

REGISTER(VirtualDatasetInfo);
}  // namespace parallel
}  // namespace mindspore
