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

#include "frontend/parallel/tensor_layout/redistribution_operator_infer.h"
#include <utility>
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
Status RedistributionOperatorInfer::Init(const TensorLayout &tensor_layout, const Map &out_tensor_map,
                                         RankList dev_list, bool is_cost_model) {
  in_tensor_map_ = tensor_layout.tensor_map();
  dev_mat_ = tensor_layout.device_arrangement();

  if (in_tensor_map_.GetDimSize() == 0 || out_tensor_map.GetDimSize() != in_tensor_map_.GetDimSize()) {
    MS_LOG(ERROR) << "Invalid input when initialize RedistributionOperatorInfer!";
    return Status::FAILED;
  }

  cur_tensor_layout_ = tensor_layout;
  out_tensor_map_ = out_tensor_map;
  dev_list_ = std::move(dev_list);

  operator_list_.clear();
  operator_vector_.clear();
  output_info_vector_.clear();

  if (constructor_.Init(dev_list_, dev_mat_.array(), is_cost_model) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Init constructor failed";
    return Status::FAILED;
  }
  constructor_.UpdateTensorShape(cur_tensor_layout_.slice_shape().array());

  size_t key = 0;
  Shape map = in_tensor_map_.array();
  for (int64_t item : map) {
    map_[key++] = item;
  }

  is_cost_model_ = is_cost_model;
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::InferRedistributionOperator() {
  this->constructor_.UpdateTensorShape(cur_tensor_layout_.slice_shape().array());
  while (!map_.empty()) {
    size_t len_global = operator_list_.size();

    while (!map_.empty()) {
      size_t len_split_by_axis = operator_list_.size();
      // split_by_axis operation
      if (InferSplitByAxis() == Status::FAILED) {
        return Status::FAILED;
      }
      // permute_by_axis operation
      while (!map_.empty()) {
        size_t len_permute_by_axis = operator_list_.size();
        if (InferPermuteByAxis() == Status::FAILED) {
          return Status::FAILED;
        }
        if (len_permute_by_axis == operator_list_.size()) {
          break;
        }
      }
      if (len_split_by_axis == operator_list_.size()) {
        break;
      }
    }
    // concat_by_axis operation
    if (InferConcatByAxis() == Status::FAILED) {
      return Status::FAILED;
    }
    // break loop structure with concat_by_axis
    if (len_global == operator_list_.size() && !map_.empty()) {
      size_t index = map_.begin()->first;
      int64_t in_dim = map_[index];
      map_[index] = NONE;
      Args args = {SizeToLong(index), in_dim, dev_mat_.GetDimByReverseIdx(LongToSize(in_dim))};
      if (InsertOperator(CONCAT_BY_AXIS, args) == Status::FAILED) {
        return Status::FAILED;
      }
    }
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::InferSplitByAxis() {
  for (auto iter = map_.begin(); iter != map_.end();) {
    uint64_t index = iter->first;
    int64_t in_dim = iter->second;
    int64_t out_dim = out_tensor_map_.GetDimByIdx(index);
    if (in_dim == out_dim) {
      iter = map_.erase(iter);
      continue;
    }
    if (in_dim == NONE &&
        !std::any_of(map_.begin(), map_.end(),
                     [out_dim](const RedistributionOperatorMap::value_type &a) { return a.second == out_dim; })) {
      Args args = {dev_mat_.GetDimByReverseIdx(LongToUlong(out_dim)), UlongToLong(index), out_dim};
      if (InsertOperator(SPLIT_BY_AXIS, args) == Status::FAILED) {
        MS_LOG(ERROR) << "Insert SplitByAxis Error!";
        return Status::FAILED;
      }
      iter = map_.erase(iter);
    } else {
      (void)++iter;
    }
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::InferPermuteByAxis() {
  for (auto iter = map_.begin(); iter != map_.end();) {
    uint64_t index = iter->first;
    int64_t in_dim = iter->second;
    int64_t out_dim = out_tensor_map_.GetDimByIdx(index);
    if (in_dim == out_dim) {
      iter = map_.erase(iter);
      continue;
    }
    if (in_dim == NONE &&
        std::any_of(map_.begin(), map_.end(),
                    [out_dim](const RedistributionOperatorMap::value_type &a) { return a.second == out_dim; })) {
      int64_t cat_dim = in_tensor_map_.GetIndexByValue(out_dim);
      int64_t dev_num = dev_mat_.GetDimByReverseIdx(LongToSize(out_dim));
      if (ParallelContext::GetInstance()->enable_all2all() && !ParallelContext::GetInstance()->do_transform()) {
        int64_t dev_dim = in_tensor_map_.GetDimByIdx(LongToUlong(cat_dim));
        Args args_alltoall = {dev_mat_.GetDimByReverseIdx(LongToUlong(dev_dim)), UlongToLong(index), cat_dim, dev_dim,
                              dev_num};
        if (InsertOperator(PERMUTE_BY_AXIS, args_alltoall) == Status::FAILED) {
          MS_LOG(ERROR) << "Insert PermuteByAxis Error!";
          return Status::FAILED;
        }
      } else {
        Args args_allconcat = {cat_dim, out_dim, dev_num};
        Args args_allsplit = {dev_num, UlongToLong(index), out_dim};
        if (InsertOperator(CONCAT_BY_AXIS, args_allconcat) == Status::FAILED) {
          MS_LOG(ERROR) << "Insert ConcatByAxis Error!";
          return Status::FAILED;
        }
        if (InsertOperator(SPLIT_BY_AXIS, args_allsplit) == Status::FAILED) {
          MS_LOG(ERROR) << "Insert SplitByAxis Error!";
          return Status::FAILED;
        }
      }
      iter = map_.erase(iter);
      map_[LongToSize(cat_dim)] = NONE;
    } else {
      (void)++iter;
    }
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::InferConcatByAxis() {
  for (auto iter = map_.begin(); iter != map_.end();) {
    uint64_t index = iter->first;
    int64_t in_dim = iter->second;
    int64_t out_dim = out_tensor_map_.GetDimByIdx(index);
    if (in_dim != NONE && out_tensor_map_.GetIndexByValue(in_dim) == NONE) {
      Args args = {SizeToLong(index), in_dim, dev_mat_.GetDimByReverseIdx(LongToSize(in_dim))};
      if (InsertOperator(CONCAT_BY_AXIS, args) == Status::FAILED) {
        MS_LOG(ERROR) << "Insert ConcatByAxis Error!";
        return Status::FAILED;
      }
      if (out_dim == NONE) {
        iter = map_.erase(iter);
      } else {
        iter->second = NONE;
        (void)++iter;
      }
    } else {
      (void)++iter;
    }
  }
  return Status::SUCCESS;
}

// Transfer communicative operators into primitives and insert them into vector
Status RedistributionOperatorInfer::InsertOperator(const OperatorName &name, const Args &args) {
  OperatorR op = std::make_pair(name, args);
  OperatorC op_cost = std::make_pair(op, cur_tensor_layout_.slice_shape().array());
  operator_list_.push_back(op_cost);
  if (construct_op_flag_) {
    if (name == SPLIT_BY_AXIS) {
      if (TransferSplitByAxis(args) == Status::FAILED) {
        return Status::FAILED;
      }
    } else if (name == PERMUTE_BY_AXIS) {
      if (TransferPermuteByAxis(args) == Status::FAILED) {
        return Status::FAILED;
      }
    } else {
      if (TransferConcatByAxis(args) == Status::FAILED) {
        return Status::FAILED;
      }
    }
    constructor_.UpdateTensorShape(cur_tensor_layout_.slice_shape().array());
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::TransferSplitByAxis(const Args &args) {
  if (args.size() < TRANSFER_SPLIT_ARGS_SIZE) {
    MS_LOG(ERROR) << "args size should not be less than 3!";
    return Status::FAILED;
  }
  size_t index = LongToSize(args[TRANSFER_PERMUTE_SPLIT_DIM_INDEX]);
  if (constructor_.StridedSliceOP(args) != Status::SUCCESS) {
    return Status::FAILED;
  } else {
    operator_vector_.push_back(constructor_.GetOperator());
    output_info_vector_.push_back(std::make_pair(false, 0));
  }
  if (cur_tensor_layout_.UpdateTensorMap(index, args[TRANSFER_PERMUTE_CONCAT_DIM_INDEX]) == Status::FAILED) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::TransferPermuteByAxis(const Args &args) {
  if (args.size() < TRANSFER_PERMUTE_ARGS_SIZE) {
    MS_LOG(ERROR) << "args size should not be less than 5!";
    return Status::FAILED;
  }
  if (constructor_.AlltoAllOP(args) != Status::SUCCESS) {
    return Status::FAILED;
  } else {
    operator_vector_.push_back(constructor_.GetOperator());
    output_info_vector_.push_back(std::make_pair(false, 0));
  }
  size_t index = LongToSize(args[TRANSFER_PERMUTE_SPLIT_DIM_INDEX]);
  int64_t val = args[TRANSFER_PERMUTE_CONCAT_DIM_INDEX];
  int64_t out_dim = out_tensor_map_.GetDimByIdx(index);

  if (cur_tensor_layout_.UpdateTensorMap(LongToSize(val), NONE) == Status::FAILED) {
    return Status::FAILED;
  }
  if (cur_tensor_layout_.UpdateTensorMap(index, out_dim) == Status::FAILED) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status RedistributionOperatorInfer::TransferConcatByAxis(const Args &args) {
  if (args.size() < TRANSFER_CONCAT_ARGS_SIZE) {
    MS_LOG(ERROR) << "args size should not be less than 3!";
    return Status::FAILED;
  }
  int64_t tensor_dim = args[TRANSFER_CONCAT_TENSOR_DIM_INDEX];
  int64_t dev_dim = args[TRANSFER_CONCAT_DEV_DIM_INDEX];
  int64_t split_count = args[TRANSFER_CONCAT_SPLIT_COUNT_INDEX];
  if (constructor_.AllGatherOP(dev_dim) != Status::SUCCESS) {
    return Status::FAILED;
  } else {
    operator_vector_.push_back(constructor_.GetOperator());
    (void)output_info_vector_.emplace_back(false, 0);
  }
  if (tensor_dim != 0) {
    if (constructor_.SplitOP(split_count) != Status::SUCCESS) {
      return Status::FAILED;
    } else {
      operator_vector_.push_back(constructor_.GetOperator());
      (void)output_info_vector_.emplace_back(true, split_count);
    }
    if (constructor_.ConcatOP(tensor_dim) != Status::SUCCESS) {
      return Status::FAILED;
    } else {
      operator_vector_.push_back(constructor_.GetOperator());
      (void)output_info_vector_.emplace_back(false, 0);
    }
  }
  if (cur_tensor_layout_.UpdateTensorMap(LongToSize(tensor_dim), NONE) == Status::FAILED) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
