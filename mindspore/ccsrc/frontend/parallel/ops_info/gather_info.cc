/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/gather_info.h"

#include <vector>
#include <numeric>
#include <functional>
#include <utility>
#include <algorithm>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/common/utils/parallel_context.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_context.h"
#include "distributed/embedding_cache/embedding_cache_utils.h"
#endif

namespace mindspore {
namespace parallel {
Status GatherInfo::GetManualSplitWithoutOffsetAttr() {
  auto manual_split_without_offset_iter = attrs_.find("manual_split");
  if (manual_split_without_offset_iter != attrs_.end()) {
    manual_split_ = true;
    MS_EXCEPTION_IF_NULL(manual_split_without_offset_iter->second);
    if (manual_split_without_offset_iter->second->cast<ValueTuplePtr>() == nullptr) {
      MS_LOG(ERROR) << name_ << ": Manual split without offset strategy's format is wrong! Need ValueSequence";
      return FAILED;
    }
    std::vector<ValuePtr> value_vector = manual_split_without_offset_iter->second->cast<ValueTuplePtr>()->value();
    MS_LOG(INFO) << name_ << ": manual split with offset is " << manual_split_without_offset_iter->second->ToString();

    int64_t offset = 0;
    for (auto &ele : value_vector) {
      index_offsets_.push_back(offset);
      if (!ele->isa<Int64Imm>()) {
        MS_LOG(ERROR) << name_ << ": The element of manual split must be int64_t";
        return FAILED;
      }
      int64_t param_split_shape = static_cast<int64_t>(GetValue<int64_t>(ele));
      if (param_split_shape <= 0) {
        MS_LOG(ERROR) << name_ << ": The value of manual split must be positive, but got " << param_split_shape;
        return FAILED;
      }
      param_split_shapes_.push_back(param_split_shape);
      offset += param_split_shape;
    }
    if (param_split_shapes_.empty()) {
      MS_LOG(ERROR) << name_ << ": Failed to extract param split's split info";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status GatherInfo::GetManualSplitAttr() {
  auto manual_split_with_offset_iter = attrs_.find("manual_split_with_offset");
  if (manual_split_with_offset_iter != attrs_.end()) {
    manual_split_ = true;
    auto var = manual_split_with_offset_iter->second->cast<ValueTuplePtr>();
    if (var == nullptr) {
      MS_LOG(ERROR) << name_ << ": Manual split with offset strategy's format is wrong! Need ValueSequence";
      return FAILED;
    }

    MS_LOG(INFO) << name_ << ": manual split with offset strategy " << var->ToString();
    for (auto &ele : var->value()) {
      if (!ele->isa<ValueSequence>()) {
        MS_LOG(ERROR) << name_ << ": Manual split with offset strategy's format is wrong! Need ValueSequence";
        return FAILED;
      }
      std::vector<ValuePtr> value_vector = ele->cast<ValueTuplePtr>()->value();
      if (value_vector.size() != 2) {
        MS_LOG(ERROR) << name_ << ": Size of manual split with offset's element must be 2";
        return FAILED;
      }
      int64_t param_split_row = (GetValue<int64_t>(value_vector[0]));
      int64_t offset = (GetValue<int64_t>(value_vector[1]));
      if ((param_split_row <= 0) || (offset < 0)) {
        MS_LOG(ERROR) << name_ << ": The value of param split shape must be positive, "
                      << "and the offset must be greater than or equal to 0";
        return FAILED;
      }
      param_split_shapes_.push_back(param_split_row);
      index_offsets_.push_back(offset);
    }

    if (param_split_shapes_.empty()) {
      MS_LOG(ERROR) << name_ << ": Failed to extract param split with offset's split info";
      return FAILED;
    }
    if (std::any_of(index_offsets_.begin(), index_offsets_.end(), [](const int64_t &offset) { return offset < 0; })) {
      MS_LOG(ERROR) << name_ << ": Index offset must not be less than 0";
      return FAILED;
    }
    return SUCCESS;
  }

  if (GetManualSplitWithoutOffsetAttr() != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

Status GatherInfo::GetAttrs() {
  // get axis, the third input is the axis, is a ValueNode, embeddinglookup doesn't have axis.
  if (target_ != CPU) {
    if (input_value_.at(2) == nullptr) {
      MS_LOG(ERROR) << name_ << ": the third input value is nullptr, is not a ValueNode!";
      return FAILED;
    }
    auto axis = GetValue<int64_t>(input_value_.at(2));
    // if axis is negative then convert it to positive
    auto params_shape = inputs_shape_.at(0);
    if (params_shape.size() == 0) {
      MS_LOG(ERROR) << name_ << ": params can not be a scalar!";
      return FAILED;
    }
    if (axis < 0) {
      axis += SizeToLong(inputs_shape_[0].size());
    }
    axis_ = axis;
  }

  auto target_iter = attrs_.find(TARGET);
  if (target_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(target_iter->second);
    if (target_iter->second->isa<StringImm>()) {
      target_ = target_iter->second->cast<StringImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of target is not a string.";
    }
  }

  if (GetManualSplitAttr() != SUCCESS) {
    return FAILED;
  }

  if (manual_split_ && (axis_ != 0)) {
    MS_LOG(ERROR) << name_ << ": The axis must be 0 if manual split, bug got " << axis_;
    return FAILED;
  }

  if (std::find(inputs_shape_[1].begin(), inputs_shape_[1].end(), -1) != inputs_shape_[1].end()) {
    dynamic_shape_indices_ = true;
  }
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    dynamic_shape_indices_ = true;
  }
#endif
  return SUCCESS;
}

// constraint: the field dimension of indices is the last dimension
// parameter's dim >= 1, indices' dim >= 1, axis == 0
// parameter's strategy: [a, b, ..., c], indices' strategy: [1, ..., 1, a]
// output's strategy: [1, ..., 1, a, b, ..., c]
// dev_matrix: [a, b, ..., c]
// can not support repeated calculation
Status GatherInfo::CheckManualSplit(const Strategies &strategy) {
  if (strategy.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << strategy.size();
    return FAILED;
  }
  Dimensions param_strategy = strategy[0];
  Dimensions indices_strategy = strategy[1];

  if (indices_strategy.size() < 1) {
    MS_LOG(ERROR) << name_ << ": The size of indices strategy must be positive, but got " << indices_strategy.size();
    return FAILED;
  }

  auto product_i = std::accumulate(indices_strategy.begin(), indices_strategy.end(), 1, std::multiplies<int64_t>());
  size_t indices_split_dim = indices_strategy.size() - 1;  // only the last dim of indices can be split
  if (product_i != indices_strategy[indices_split_dim]) {
    MS_LOG(ERROR) << name_ << ": Only the last dim of indices can be split, but got " << indices_strategy;
    return FAILED;
  }

  if (param_strategy[0] != indices_strategy[indices_split_dim]) {
    MS_LOG(ERROR) << name_ << ": The param_strategy[0] " << param_strategy[0]
                  << " must be equal to indices_strategy[-1] " << indices_strategy[indices_split_dim];
    return FAILED;
  }

  if (indices_strategy[indices_split_dim] != SizeToLong(param_split_shapes_.size())) {
    MS_LOG(ERROR) << name_ << ": The indices_strategy[-1] " << indices_strategy[indices_split_dim]
                  << " must be equal to manual split size " << param_split_shapes_.size();
    return FAILED;
  }

  int64_t min_param_slice_row = inputs_shape_[1][indices_split_dim] / indices_strategy[indices_split_dim];
  bool invalid = std::any_of(param_split_shapes_.begin(), param_split_shapes_.end(),
                             [&min_param_slice_row](int64_t v) { return v < min_param_slice_row; });
  if (invalid) {
    MS_LOG(ERROR) << name_ << ": The split value " << param_split_shapes_
                  << " must be larger than or equal to indices field slice size " << min_param_slice_row;
    return FAILED;
  }

  if (inputs_shape_[0][0] < inputs_shape_[1][indices_split_dim]) {
    MS_LOG(ERROR) << name_ << ": The param's row size " << inputs_shape_[0][0]
                  << " is smaller than indices' field size " << inputs_shape_[1][indices_split_dim];
    return FAILED;
  }

  // Don't support repeated calc
  auto product_p = std::accumulate(param_strategy.begin(), param_strategy.end(), 1, std::multiplies<int64_t>());
  if (product_p < stage_device_size_) {
    MS_LOG(ERROR) << name_ << ": Manual split doesn't support repeated calc";
    return FAILED;
  }

  int64_t split_shape_sum = std::accumulate(param_split_shapes_.begin(), param_split_shapes_.end(), 0,
                                            [](int64_t s, int64_t shape) { return s + shape; });
  if (split_shape_sum != inputs_shape_[0][0]) {
    MS_LOG(ERROR) << name_ << ": Sum of split shapes " << split_shape_sum << " must be equal to param_shape[0] "
                  << inputs_shape_[0][0];
    return FAILED;
  }
  return SUCCESS;
}

Status GatherInfo::CheckSplitAxisStrategy(const StrategyPtr &strategy) {
  auto input_dim = strategy->GetInputDim();
  auto param_strategy = input_dim.at(0);
  auto index_strategy = input_dim.at(1);
  // param_strategy(axis) != 1, index can't be split
  auto product_i = std::accumulate(index_strategy.begin(), index_strategy.end(), 1, std::multiplies<int64_t>());
  if ((param_strategy.at(LongToSize(axis_)) != 1) && (product_i != 1)) {
    FILTER_LOG(is_auto_parallel_) << name_ << ": param is split at dim (axis)" << axis_ << " ,index can't be split.";
    return FAILED;
  }

  // param_strategy(axis) != 1, and axis != 0, don't support repeated calc
  auto product_p = std::accumulate(param_strategy.begin(), param_strategy.end(), 1, std::multiplies<int64_t>());
  if ((product_p != stage_device_size_) && (param_strategy.at(LongToSize(axis_)) != 1) && (axis_ != 0)) {
    FILTER_LOG(is_auto_parallel_) << name_ << ": Invalid strategy. Don't support repeated calc.";
    return FAILED;
  }

  if ((product_p != stage_device_size_) && (param_strategy.at(LongToSize(axis_)) != 1) && (axis_ == 0)) {
    if ((param_strategy.size() == 2) && (param_strategy[1] != 1)) {
      FILTER_LOG(is_auto_parallel_) << name_
                                    << ": axis(0) is split, and param_strategy[1] != 1, don't support"
                                       " repeated calc.";
      return FAILED;
    }
    MS_LOG(INFO) << name_ << ": split axis(0) and repeat calculation";
  }
  return SUCCESS;
}

// return true: axis is 0, and split the first dimension of parameter and the first dimension of indices
// otherwise return false
bool GatherInfo::ShardBatchAndAxis(const Strategies &strategy) const {
  if (axis_ != 0) {
    return false;
  }

  if (strategy.size() != 2) {
    return false;
  }

  Dimensions param_strategy = strategy[0];
  Dimensions indices_strategy = strategy[1];
  if ((param_strategy.size() != 2) || (indices_strategy.size() != 2)) {
    return false;
  }

  if ((param_strategy[1] != 1) || (indices_strategy[1] != 1)) {
    return false;
  }

  if (param_strategy[0] * indices_strategy[0] != stage_device_size_) {
    return false;
  }

  if ((param_strategy[0] == stage_device_size_) || (indices_strategy[0] == stage_device_size_)) {
    return false;
  }

  return true;
}

void GatherInfo::SetAttribute(const StrategyPtr &strategy) {
  auto input_dim = strategy->GetInputDim();
  auto param_strategy = input_dim.at(0);
  // axis=0, index_shape(0)%param_strategy(0) must be 0
  Shape index_shape = inputs_shape_.at(1);
  if ((axis_ == 0) && (index_shape.at(0) % param_strategy.at(0) != 0) && !dynamic_shape_indices_) {
    MS_LOG(INFO) << name_ << ": index_shape(0) can't be divided by param_strategy(0), use allreduce in forward";
    axis_split_forward_allreduce_ = true;
  } else if (is_auto_parallel_) {
    // in auto parallel mode, this function will be called many times, so need to reset the flags
    axis_split_forward_allreduce_ = false;
  }

  auto product_param = std::accumulate(param_strategy.begin(), param_strategy.end(), 1, std::multiplies<int>());
  // Cast 1: If repeated calculation, need to set repeated num to the left of dev-matrix. For example,
  // parameter strategy is [8, 1], indices strategy is [1, 1], dev num is 16,
  // and dev_matrix is [2, 1, 8, 1, 1], the communication groups are [0, 8] and [0, 1, 2, 3, 4, 5, 6, 7], they
  // can communicate normally, and dev0 to dev7 have the all parameters.
  // Cast 2: If not repeated calculation(such as data parallel), need to set repeated num to the right,
  // as it's easy to introduce the redistribution after or before gather operation, influencing the performance.
  if (product_param == stage_device_size_ || product_param == 1) {
    repeated_num_in_dev_matrix_right_ = true;
  } else {
    repeated_num_in_dev_matrix_right_ = false;
  }
  MS_LOG(INFO) << "Set repeated_num_in_dev_matrix_right for gather to " << repeated_num_in_dev_matrix_right_;
}

Status GatherInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // param slice shape preferably 32Byte aligned
  auto param_shape = inputs_shape_.at(0);
  auto input_dim = strategy->GetInputDim();
  auto param_strategy = input_dim.at(0);
  auto slice_shape = param_shape.at(param_shape.size() - 1) / param_strategy.at(param_strategy.size() - 1);
  if ((target_ != CPU) && (slice_shape % 8 != 0) && (slice_shape != 1)) {
    MS_LOG(WARNING) << "Gather: Last dim of param slice shape is not 32Byte aligned.";
  }

  if (manual_split_) {
    if (CheckManualSplit(strategy->GetInputDim()) != SUCCESS) {
      return FAILED;
    }
    // when using manual_split, no need to check belowings.
    return SUCCESS;
  }

  // parameter not split axis
  if (param_strategy.at(LongToSize(axis_)) == 1) {
    return SUCCESS;
  }

  // only support 1-dim and 2-dim param
  if (inputs_shape_.at(0).size() != 1 && inputs_shape_.at(0).size() != 2) {
    MS_LOG(ERROR) << name_ << ": Don't support param dim " << inputs_shape_.at(0).size();
    return FAILED;
  }

  // don't support scalar index
  if (inputs_shape_.at(1).size() == 0) {
    MS_LOG(ERROR) << name_ << ": Don't support scalar index.";
    return FAILED;
  }

  if (ShardBatchAndAxis(strategy->GetInputDim())) {
    shard_batch_and_axis_ = true;
    axis_split_forward_allreduce_ = true;
    MS_LOG(INFO) << name_ << ": Sharding batch and axis, and the forward use allreduce";
    return SUCCESS;
  } else if (is_auto_parallel_) {
    // in auto parallel mode, this function will be called many times, so need to reset the flags
    shard_batch_and_axis_ = false;
    axis_split_forward_allreduce_ = false;
  }

  // axis != 0, param_shape(0)%(param_strategy(0)*param_strategy(axis)) must be 0
  if (axis_ != 0 && param_shape.at(0) % (param_strategy.at(0) * param_strategy.at(LongToSize(axis_))) != 0) {
    MS_LOG(ERROR) << name_ << ": param_shape(0) can't be divided by (param_strategy(0)*param_strategy(axis)).";
    return FAILED;
  }

  if (CheckSplitAxisStrategy(strategy) != SUCCESS) {
    return FAILED;
  }

  // According to the strategy, set the private members.
  SetAttribute(strategy);

  return SUCCESS;
}

Status GatherInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy == nullptr) {
    MS_LOG(INFO) << name_ << ": The output strategy is null";
    return SUCCESS;
  }

  if (CheckStrategyValue(out_strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid output strategy";
    return FAILED;
  }

  if (manual_split_) {
    MS_LOG(ERROR) << name_ << ": The manual split do not support set output strategy";
    return FAILED;
  }

  if (axis_ != 0) {
    MS_LOG(ERROR) << name_ << ": The output strategy only support axis = 0, but the axis is " << axis_;
    return FAILED;
  }

  if (dynamic_shape_indices_) {
    MS_LOG(ERROR) << name_ << ": The dynamic shape do not support set output strategy";
    return FAILED;
  }

  auto in_stra = strategy_->GetInputDim();
  auto param_strategy = in_stra[0];
  if (inputs_shape_[1].empty() || (inputs_shape_[1][0] % param_strategy[0] != 0)) {
    MS_LOG(ERROR) << name_ << ": index_shape[0] can't be divided by param_strategy[0], can not set output strategy";
    return FAILED;
  }

  if (param_strategy[LongToSize(axis_)] == 1) {
    MS_LOG(ERROR) << name_ << ": The axis is not split, can not set output strategy";
    return FAILED;
  }

  auto index_strategy = in_stra[1];

  // only for axis == 0
  auto allreduce_strategy = index_strategy;
  (void)allreduce_strategy.insert(allreduce_strategy.end(), param_strategy.begin() + 1, param_strategy.end());
  auto reduce_scatter_strategy = allreduce_strategy;
  reduce_scatter_strategy[0] *= param_strategy[0];

  auto out_stra = out_strategy->GetInputDim()[0];
  if (out_stra == allreduce_strategy) {
    axis_split_forward_allreduce_ = true;
    MS_LOG(INFO) << name_ << ": The output strategy is " << out_stra << ", forward use allreduce";
    return SUCCESS;
  } else if (out_stra == reduce_scatter_strategy) {
    axis_split_forward_allreduce_ = false;
    MS_LOG(INFO) << name_ << ": The output strategy is " << out_stra << ", forward use reduce scatter";
    return SUCCESS;
  }

  MS_LOG(ERROR) << name_ << ": The output strategy " << out_stra << " is invalid, it must be " << allreduce_strategy
                << " or " << reduce_scatter_strategy;
  return FAILED;
}

Status GatherInfo::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_a_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_a_group;
  if (CreateGroupByTensorMap(input_a_tensor_map, &input_a_group) != SUCCESS) {
    ReportError(name_ + " : Create group for input a failed.");
    return FAILED;
  }

  OperatorVector op_for_input_a, op_for_input_b, op_for_axis;
  if (input_a_group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror group is empty.";
    return SUCCESS;
  } else {
    op_for_input_a = CreateMirrorOps(input_a_group[0].name(), input_a_group[0].GetDevNum());
    MS_LOG(INFO) << name_ << " : Create the mirror ops for input a success, group is " << input_a_group[0].name();
  }

  mirror_ops_.push_back(op_for_input_a);
  mirror_ops_.push_back(op_for_input_b);
  mirror_ops_.push_back(op_for_axis);

  return SUCCESS;
}

Status GatherInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  out_dev_matrix_shape_.clear();
  // infer input dev_matrix_shape
  auto param_stra = strategy_->GetInputDim();
  auto param_strategy = param_stra.at(0);
  auto index_strategy = param_stra.at(1);

  if (manual_split_) {
    dev_matrix_shape_ = param_strategy;
    out_dev_matrix_shape_ = dev_matrix_shape_;
    return SUCCESS;
  }

  if (shard_batch_and_axis_) {
    dev_matrix_shape_ = {index_strategy[0], param_strategy[0]};
    // if forward use reducescatter, the output's dev matrix is {index_strategy[0] * param_strategy[0]}
    if (axis_split_forward_allreduce_) {
      out_dev_matrix_shape_ = dev_matrix_shape_;
    } else {
      out_dev_matrix_shape_ = {index_strategy[0] * param_strategy[0]};
    }
    MS_LOG(INFO) << name_ << ": Sharding batch and axis, the dev matrix is " << dev_matrix_shape_
                 << ", out dev matrix is " << out_dev_matrix_shape_;
    return SUCCESS;
  }

  dev_matrix_shape_ = param_strategy;

  // param_strategy(axis) is 1
  if (param_strategy.at(LongToSize(axis_)) == 1) {
    (void)dev_matrix_shape_.insert(dev_matrix_shape_.end(), index_strategy.begin(), index_strategy.end());
  }

  // infer out dev_matrix_shape
  // axis is not 0, split axis
  if (axis_ != 0 && param_strategy.at(LongToSize(axis_)) != 1) {
    for (size_t i = 1; i < param_strategy.size(); ++i) {
      if (i == LongToSize(axis_)) {
        out_dev_matrix_shape_.push_back(1);
      } else {
        out_dev_matrix_shape_.push_back(param_strategy.at(i));
      }
    }
    out_dev_matrix_shape_.push_back(param_strategy.at(0) * param_strategy.at(LongToSize(axis_)));
  } else {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  auto param_product = std::accumulate(param_strategy.begin(), param_strategy.end(), 1, std::multiplies<int64_t>());
  auto index_product = std::accumulate(index_strategy.begin(), index_strategy.end(), 1, std::multiplies<int64_t>());
  if (param_product * index_product < stage_device_size_) {
    auto repeated_calc_num = stage_device_size_ / (param_product * index_product);
    if (repeated_num_in_dev_matrix_right_) {
      out_dev_matrix_shape_.push_back(repeated_calc_num);
    } else {
      (void)out_dev_matrix_shape_.insert(out_dev_matrix_shape_.begin(), repeated_calc_num);
    }
  }

  return SUCCESS;
}

void GatherInfo::InferInputsTensorMap() {
  // infer input tensor map
  // param_strategy(axis) is not 1
  size_t param_size = inputs_shape_.at(0).size();
  size_t index_size = inputs_shape_.at(1).size();
  size_t total_size = param_size + index_size;
  Shape tensor_map_index;
  Shape tensor_map_params;
  auto input_dim = strategy_->GetInputDim();
  auto param_strategy = input_dim.at(0);
  if (param_strategy.at(LongToSize(axis_)) != 1) {
    (void)tensor_map_index.insert(tensor_map_index.begin(), index_size, MAP_NONE);
    for (size_t i = 0; i < param_size; ++i) {
      tensor_map_params.push_back(SizeToLong(param_size - i - 1));
    }
  } else {
    // param_strategy(axis) is 1
    for (size_t i = 0; i < param_size; ++i) {
      tensor_map_params.push_back(SizeToLong(total_size - i - 1));
    }
    for (size_t i = 0; i < index_size; ++i) {
      tensor_map_index.push_back(SizeToLong(index_size - i - 1));
    }
  }
  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_params));
  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_index));
}

Shape GatherInfo::InferOutputsTensorMapSplitAxis() {
  Shape tensor_map_out;
  size_t param_size = inputs_shape_.at(0).size();
  size_t index_size = inputs_shape_.at(1).size();
  if (axis_ == 0) {
    if ((dynamic_shape_indices_ && target_ != CPU) || axis_split_forward_allreduce_) {
      // the output is repeat calculation
      (void)tensor_map_out.insert(tensor_map_out.end(), MAP_NONE);
    } else {
      (void)tensor_map_out.insert(tensor_map_out.end(), param_size - 1);
    }
    (void)tensor_map_out.insert(tensor_map_out.end(), index_size - 1, MAP_NONE);
    for (size_t i = 1; i < param_size; ++i) {
      tensor_map_out.push_back(param_size - 1 - i);
    }
  } else {
    for (size_t i = 0; i < param_size; ++i) {
      if (i == LongToSize(axis_)) {
        (void)tensor_map_out.insert(tensor_map_out.end(), index_size, MAP_NONE);
      } else {
        if (i == 0 && dynamic_shape_indices_ && target_ != CPU) {
          tensor_map_out.push_back(MAP_NONE);
        }
        tensor_map_out.push_back(SizeToLong(i));
      }
    }
  }
  return tensor_map_out;
}

void GatherInfo::InferOutputsTensorMap() {
  // infer output tensor map
  size_t param_size = inputs_shape_.at(0).size();
  size_t index_size = inputs_shape_.at(1).size();
  size_t total_size = param_size + index_size;
  Shape tensor_map_out;
  auto input_dim = strategy_->GetInputDim();
  auto param_strategy = input_dim.at(0);
  if (param_strategy.at(LongToSize(axis_)) == 1) {
    // param_strategy(axis) is 1
    for (size_t i = 0; i < param_size; ++i) {
      if (i == LongToSize(axis_)) {
        for (size_t j = 0; j < index_size; ++j) {
          tensor_map_out.push_back(SizeToLong(index_size - j - 1));
        }
      } else {
        tensor_map_out.push_back(SizeToLong(total_size - i - 1));
      }
    }
  } else {
    // param_strategy(axis) is not 1
    tensor_map_out = InferOutputsTensorMapSplitAxis();
  }
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
}

void GatherInfo::InferTensorMapForManualSplit() {
  Shape param_map;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    param_map.push_back(static_cast<int64_t>(size - i - 1));
  }

  size_t indices_size = inputs_shape_[1].size();
  Shape indices_map(indices_size, MAP_NONE);
  indices_map[indices_size - 1] = param_map[0];

  Shape out_map = param_map;
  (void)out_map.insert(out_map.begin(), indices_size - 1, MAP_NONE);

  (void)inputs_tensor_map_.emplace_back(std::move(param_map));
  (void)inputs_tensor_map_.emplace_back(std::move(indices_map));
  (void)outputs_tensor_map_.emplace_back(std::move(out_map));
}

Status GatherInfo::InferTensorMap() {
  if (manual_split_) {
    InferTensorMapForManualSplit();
    return SUCCESS;
  }

  if (shard_batch_and_axis_) {
    Shape param_tensor_map = {0, MAP_NONE};
    Shape indices_tensor_map = {1, MAP_NONE};
    Shape out_tensor_map;
    if (axis_split_forward_allreduce_) {
      out_tensor_map = {1, MAP_NONE, MAP_NONE};  // the dev matrix is (index_strategy[0], param_strategy[0])
    } else {
      out_tensor_map = {0, MAP_NONE, MAP_NONE};  // the dev matrix is (index_strategy[0] * param_strategy[0])
    }
    (void)inputs_tensor_map_.emplace_back(std::move(param_tensor_map));    // param
    (void)inputs_tensor_map_.emplace_back(std::move(indices_tensor_map));  // indices
    (void)outputs_tensor_map_.emplace_back(std::move(out_tensor_map));     // output
    return SUCCESS;
  }
  InferInputsTensorMap();
  InferOutputsTensorMap();
  return SUCCESS;
}

Status GatherInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape input_index_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);
  int64_t rank = g_device_manager->rank_index_in_stage();
  // infer tensor layout
  TensorLayout input_tensor_layout, input_index_layout, output_tensor_layout;
  if (manual_split_) {
    int64_t bias_size = 1;
    if (dev_matrix_shape_.size() > 1) {
      bias_size =
        std::accumulate(dev_matrix_shape_.begin() + 1, dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
    }
    if (bias_size == 0) {
      MS_LOG(ERROR) << name_ << ": Invalid device matrix " << dev_matrix_shape_;
      return FAILED;
    }
    input_shape[0] = param_split_shapes_[LongToSize(rank / bias_size)];
    input_shape[0] = input_shape[0] * dev_matrix_shape_[0];
  }
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(0), input_shape) != SUCCESS) ||
      (input_index_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(1), input_index_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(out_dev_matrix_shape_, outputs_tensor_map_.at(0), output_shape) !=
       SUCCESS)) {
    return FAILED;
  }

  if (manual_split_) {
    input_tensor_layout.set_uniform_split(false);
  }
  // infer tensor info
  TensorInfo input_tensor_info(input_tensor_layout);
  TensorInfo input_index_info(input_index_layout);
  TensorInfo output_tensor_info(output_tensor_layout);

  inputs_tensor_info_.push_back(input_tensor_info);
  inputs_tensor_info_.push_back(input_index_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status GatherInfo::InferBias() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();
  auto input_shape = inputs_shape_.at(0);
  auto params_strategy = strategy_->GetInputDim().at(0);

  if (shard_batch_and_axis_) {
    slice_size_ = input_shape[0] / params_strategy[0];
    bias_ = rank % params_strategy[0] * slice_size_;
    MS_LOG(INFO) << name_ << ": Sharding batch and axis, the rank is " << rank << ", slice size is " << slice_size_
                 << ", bias is " << bias_;
    return SUCCESS;
  }

  // axis don't split
  if (params_strategy.at(LongToSize(axis_)) == 1) {
    bias_ = 0;
    return SUCCESS;
  }
  // params_size=1, axis=0
  if ((input_shape.size() == 1) && (axis_ == 0)) {
    slice_size_ = input_shape.at(0) / params_strategy.at(0);
    // if repeated calculation, because the repeated num in the right of dev-matrix, so rank need to div repeated num
    if (repeated_calc_num_ > 1) {
      if (repeated_num_in_dev_matrix_right_) {
        rank = rank / repeated_calc_num_;
      } else {
        rank = rank % params_strategy[0];
      }
    }
    bias_ = rank * slice_size_;
    return SUCCESS;
  }
  // params_size=2, axis=0
  if ((input_shape.size() == 2) && (axis_ == 0)) {
    slice_size_ = input_shape.at(0) / params_strategy.at(0);
    // if repeated calculation, because the repeated num in the right of dev-matrix, so rank need to div repeated num
    if (repeated_calc_num_ > 1) {
      if (repeated_num_in_dev_matrix_right_) {
        rank = rank / repeated_calc_num_;
      } else {
        rank = rank % (params_strategy[0] * params_strategy[1]);
      }
    }
#if defined(__linux__) && defined(WITH_BACKEND)
    if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
      if (ps::PSContext::instance()->enable_distributed_mindrt()) {
        bias_ = static_cast<int64_t>(embedding_cache_table_manager.cache_indices_lower_bound());
      }
      return SUCCESS;
    }
#endif
    bias_ = rank / params_strategy.at(1) * slice_size_;
    return SUCCESS;
  }
  // params_size=2, axis=1
  if ((input_shape.size() == 2) && (axis_ == 1)) {
    slice_size_ = input_shape.at(1) / params_strategy.at(1);
    bias_ = rank % params_strategy.at(1) * slice_size_;
    return SUCCESS;
  }
  MS_LOG(ERROR) << name_ << ": Don't support params_size:" << input_shape.size() << " axis:" << axis_;
  return FAILED;
}

Status GatherInfo::InferOffset() {
  CheckGlobalDeviceManager();
  size_t rank = LongToSize(g_device_manager->rank_index_in_stage());

  MS_EXCEPTION_IF_NULL(strategy_);
  auto param_strategy = strategy_->GetInputDim()[0];

  int64_t bias_size = 1;
  if (param_strategy.size() > 1) {
    bias_size = std::accumulate(param_strategy.begin() + 1, param_strategy.end(), 1, std::multiplies<int64_t>());
  }
  MS_EXCEPTION_IF_ZERO("bias_size", LongToSize(bias_size));
  size_t index = rank / LongToSize(bias_size);
  if (index < index_offsets_.size()) {
    index_offset_ = index_offsets_[index];
    MS_LOG(INFO) << name_ << ": Device rank " << rank << ", Index Offset: " << index_offset_;
    return SUCCESS;
  }

  MS_LOG(ERROR) << name_ << ": Get index offset failed, index offset size is" << index_offsets_.size();
  return FAILED;
}

Status GatherInfo::InferGroup() {
  size_t dim = LongToSize(axis_);

  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;

  // the dev_matrix[0] is repeated_calc_num, so the dim need to add 1
  if ((repeated_calc_num_ > 1) && !repeated_num_in_dev_matrix_right_) {
    dim = dim + 1;
  }

  if (shard_batch_and_axis_) {
    dim = 1;
    MS_LOG(INFO) << name_ << ": Sharding batch and axis, the group dim is " << dim;
  }

  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(dim), &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed.";
    return FAILED;
  }
  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The group is empty";
    return SUCCESS;
  }

  MS_LOG(INFO) << name_ << ": The group ranks is " << group_devices;
  if (g_device_manager->CreateGroup(group_devices, &group_) != SUCCESS) {
    MS_LOG(ERROR) << "The node " << cnode_->fullname_with_scope() << " create reduce group failed in table row split.";
    return FAILED;
  }
  return SUCCESS;
}

Status GatherInfo::InferForwardCommunication() {
  if (manual_split_) {
    return SUCCESS;
  }

  forward_op_.clear();
  auto input_dim = strategy_->GetInputDim();
  auto param_strategy = input_dim.at(0);
  // don't split axis or target is not CPU, no need forward communication
  if (target_ != CPU || param_strategy.at(LongToSize(axis_)) == 1) {
    return SUCCESS;
  }
  // split axis
  Attr attr_group;
  OperatorName operator_name;
  if (axis_split_forward_allreduce_) {
    operator_name = ALL_REDUCE;
  } else {
    operator_name = REDUCE_SCATTER;
  }

  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Group failed.";
    return FAILED;
  }
  if (group_.name().empty()) {
    return SUCCESS;
  }
  attr_group = std::make_pair(GROUP, MakeValue(group_.name()));
  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  OperatorAttrs attrs = {attr_op, attr_group};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(operator_name, args);

  forward_op_.push_back(op);
  return SUCCESS;
}

Status GatherInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }
  if (manual_split_ && target_ != CPU) {
    if (InferOffset() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer Bias failed.";
      return FAILED;
    }
    auto sub_node =
      gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(index_offset_)});
    auto gather_v2_node =
      gen_g.PushBack({gen_g.NewOpInst(replace_op_name_), gen_g.virtual_input_node(), sub_node, CreatInt64Imm(axis_)});
    std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(sub_node, 2),
                                                               std::make_pair(gather_v2_node, 1)};
    replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
      std::make_pair(input_nodes, gather_v2_node));
    return SUCCESS;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Bias failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto gather_v2 =
    gen_g.PushBack({gen_g.NewOpInst(replace_op_name_), gen_g.virtual_input_node(), minimum, CreatInt64Imm(axis_)});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gather_v2});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype});
  auto expand_dims = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), cast, CreatInt64Imm(axis_ - 1)});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gather_v2, expand_dims});
  // don't need expand dim, if param_size = 1
  if (inputs_shape_.at(0).size() == 1) {
    mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gather_v2, cast});
  }
  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Group failed.";
    return FAILED;
  }
  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_.name()));
  OperatorAttrs attrs = {attr_op, attr_group};
  AnfNodePtr reduce_op;
  if (dynamic_shape_indices_ || axis_split_forward_allreduce_) {
    reduce_op = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});
  } else {
    reduce_op = gen_g.PushBack({gen_g.NewOpInst(REDUCE_SCATTER, attrs), mul});
  }
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(sub, 2), std::make_pair(gather_v2, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, reduce_op));

  return SUCCESS;
}

ReplaceGraphPtr GatherInfo::replace_graph(const CNodePtr &cnode) {
  if (manual_split_ && target_ != CPU) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
    }
    return replace_graph_;
  }
  auto input_dim = strategy_->GetInputDim();
  auto param_strategy = input_dim.at(0);
  // target_ == CPU, no need to replace graph
  if (target_ == CPU) {
    return nullptr;
  }
  if (param_strategy.at(LongToSize(axis_)) != 1 && ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

Status GatherInfo::ComputeReplaceOp() {
  int64_t bias = 0;
  if (manual_split_) {
    if (InferOffset() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer offset failed.";
      return FAILED;
    }
    bias = index_offset_;
  } else {
    if (InferBias() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer offset failed.";
      return FAILED;
    }
    bias = bias_;
  }

  OperatorName op_name = EMBEDDING_LOOKUP;
  OperatorAttrs attrs;
  Attr param_offset = std::make_pair("offset", MakeValue(bias));
  OperatorParams params = {std::make_pair(param_offset, 3)};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(op_name, args);
  replace_op_.push_back(op);

  return SUCCESS;
}

Status GatherInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  // only target_ == CPU, we need to replace op
  if (target_ == CPU && ComputeReplaceOp() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": ComputeReplaceOp failed.";
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status GatherInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    FILTER_LOG(is_auto_parallel_) << name_ << ": Init for cost model failed.";
    return FAILED;
  }
  auto param_strategy = strategy_->GetInputDim().at(0);
  // cost model set axis and strategy
  auto gather_cost = std::dynamic_pointer_cast<GatherCost>(operator_cost());
  gather_cost->set_axis(axis_);
  gather_cost->set_strategy(param_strategy);
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status GatherInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GatherInfo::GenerateOpStrategies(int64_t stage_id) {
  if (manual_split_) {
    MS_LOG(EXCEPTION) << name_ << ": Manual split does not support to search strategy";
  }
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

std::shared_ptr<Strategies> GatherInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Get attr failed";
  }
  if (manual_split_) {
    MS_LOG(EXCEPTION) << name_ << ": Manual split does not support to generate batch strategy";
  }

  Dimensions param_strategy(inputs_shape_[0].size(), 1);
  Dimensions index_strategy;
  index_strategy.push_back(stage_device_size_);
  for (size_t i = 1; i < inputs_shape_[1].size(); i++) {
    index_strategy.push_back(1);
  }
  Strategies strategy_v = {param_strategy, index_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

REGISTER(GatherInfo);
REGISTER(SparseGatherV2Info);
REGISTER(EmbeddingLookupInfo);
}  // namespace parallel
}  // namespace mindspore
