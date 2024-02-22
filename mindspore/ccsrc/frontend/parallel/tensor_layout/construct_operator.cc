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

#include "frontend/parallel/tensor_layout/construct_operator.h"

#include <functional>
#include <numeric>
#include <algorithm>
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
Status ConstructOperator::Init(const RankList &dev_list, const Shape &dev_matrix_shape, bool is_cost_model) {
  dev_size_ = dev_matrix_shape.size();
  dev_matrix_shape_ = dev_matrix_shape;
  dev_list_ = dev_list;
  is_cost_model_ = is_cost_model;
  return Status::SUCCESS;
}

// skip redistribution for reshape operator
OperatorVector ConstructOperator::SkipRedisReshapeOP(const Shape &shape) const {
  OperatorAttrs attrs;
  ValuePtr param_value = MakeValue(shape);
  Attr param = std::make_pair(SHAPE, param_value);
  OperatorParams params = {std::make_pair(param, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(RESHAPE, args);
  OperatorVector opvector;
  opvector.push_back(op);
  return opvector;
}

Status ConstructOperator::ReshapeOP(const Shape &shape) {
  int64_t prod = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  int64_t prod_expect = std::accumulate(tensor_shape_.begin(), tensor_shape_.end(), 1, std::multiplies<int64_t>());
  if (prod > 0 && prod_expect > 0 && prod != prod_expect) {
    ValuePtr ptr = MakeValue(shape);
    MS_EXCEPTION_IF_NULL(ptr);
    MS_LOG(ERROR) << "Invalid tensor shape " << ptr->ToString()
                  << " when construct Reshape operator! Expect production is " << prod_expect << " which shape is "
                  << tensor_shape_;
    return Status::INVALID_ARGUMENT;
  }
  OperatorAttrs attrs;
  ValuePtr param_value = MakeValue(shape);
  Attr param = std::make_pair(SHAPE, param_value);
  OperatorParams params = {std::make_pair(param, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  op_ = std::make_pair(RESHAPE, args);
  return Status::SUCCESS;
}

Operator CreateStridedSliceOp(int64_t value, const Shape &begin, const Shape &end, const Shape &strides) {
  ValuePtr param_begin_value = MakeValue(begin);
  Param param_begin = std::make_pair(std::make_pair(BEGIN, param_begin_value), STRIDED_SLICE_BEGIN_INDEX + 1);
  ValuePtr param_end_value = MakeValue(end);
  Param param_end = std::make_pair(std::make_pair(END, param_end_value), STRIDED_SLICE_END_INDEX + 1);

  ValuePtr param_strides_value = MakeValue(strides);
  Param param_strides = std::make_pair(std::make_pair(STRIDES, param_strides_value), STRIDED_SLICE_STRIDES_INDEX + 1);

  ValuePtr begin_mask = MakeValue(value);
  Param param_begin_mask = std::make_pair(std::make_pair(BEGIN_MASK, begin_mask), STRIDED_SLICE_BEGIN_MASK_INDEX + 1);
  ValuePtr end_mask = MakeValue(value);
  Param param_end_mask = std::make_pair(std::make_pair(END_MASK, end_mask), STRIDED_SLICE_END_MASK_INDEX + 1);
  ValuePtr ellipsis_mask = MakeValue(value);
  Param param_ellipsis_mask =
    std::make_pair(std::make_pair(ELLIPSIS_MASK, ellipsis_mask), STRIDED_SLICE_ELLIPSIS_MASK_INDEX + 1);
  ValuePtr new_axis_mask = MakeValue(value);
  Param param_new_axis_mask =
    std::make_pair(std::make_pair(NEW_AXIS_MASK, new_axis_mask), STRIDED_SLICE_NEW_AXIS_MASK_INDEX + 1);
  ValuePtr shrink_axis_mask = MakeValue(value);
  Param param_shrink_axis_mask =
    std::make_pair(std::make_pair(SHRINK_AXIS_MASK, shrink_axis_mask), STRIDED_SLICE_SHRINK_AXIS_MASK_INDEX + 1);

  OperatorParams params = {param_begin,    param_end,           param_strides,       param_begin_mask,
                           param_end_mask, param_ellipsis_mask, param_new_axis_mask, param_shrink_axis_mask};
  OperatorAttrs attrs;
  OperatorArgs op_args = std::make_pair(attrs, params);

  return std::make_pair(STRIDED_SLICE, op_args);
}

Status ConstructOperator::StridedSliceOP(const Args &args) {
  if (args.size() < STRIDED_SLICE_ARGS_SIZE) {
    MS_LOG(ERROR) << "args size should not be less than 3!";
    return Status::FAILED;
  }
  int64_t split_count = args[TRANSFER_PERMUTE_SPLIT_COUNT_INDEX];
  if (split_count <= 0) {
    MS_LOG(ERROR) << "split_count should not be less than 0!";
    return Status::FAILED;
  }
  int64_t split_dim = args[TRANSFER_PERMUTE_SPLIT_DIM_INDEX];
  int64_t dev_dim = args[TRANSFER_PERMUTE_CONCAT_DIM_INDEX];
  std::vector<Group> group_list;

  if (CreateGroupByDim(dev_size_ - LongToSize(dev_dim) - 1, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << "stride slice op: create group failed";
    return FAILED;
  } else if (group_list.empty()) {  // this group only has one device, don't need do StridedSlice
    MS_LOG(INFO) << "no need stride slice op";
    return SUCCESS;
  }

  Group group = group_list[0];
  size_t rank;
  if (group.GetIndex(&rank) == Status::FAILED) {
    return Status::FAILED;
  }
  size_t size = tensor_shape_.size();
  Shape begin(size);
  Shape end(size);
  Shape strides(size, 1);
  size_t index = 0;
  for (auto num : tensor_shape_) {
    if (index != LongToSize(split_dim)) {
      begin[index] = 0;
      end[index] = num;
    } else {
      if (num % split_count != 0) {
        MS_LOG(ERROR) << "Tensor with shape " << this->tensor_shape_ << " can not be split into " << split_count
                      << " slices in the dimension " << split_dim << " when construct StridedSlice operator";
        return Status::INVALID_ARGUMENT;
      }
      int64_t count = num / split_count;
      begin[index] = SizeToLong(rank) * count;
      end[index] = (SizeToLong(rank) + 1) * count;
    }
    index++;
  }

  op_ = CreateStridedSliceOp(DEFAULT, begin, end, strides);

  return Status::SUCCESS;
}

Status ConstructOperator::AllGatherOP(int64_t dev_dim) {
  if ((LongToSize(dev_dim) >= dev_size_) || (dev_dim < 0)) {
    MS_LOG(ERROR) << "Invalid device dimension " << dev_dim << " when construct AllGather operator!";
    return Status::INVALID_ARGUMENT;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(dev_size_ - LongToSize(dev_dim) - 1, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << "AllGather op: create group failed";
    return FAILED;
  } else if (group_list.empty()) {  // this group only has one device, don't need do allgather
    MS_LOG(INFO) << "no need all gather op";
    return SUCCESS;
  }

  std::string group_name = group_list[0].name();
  ValuePtr attr_value = MakeValue(group_name);
  Attr attr = std::make_pair(GROUP, attr_value);
  auto group_devices = group_list[0].GetDevicesList();
  std::vector<int64_t> group_ranks;
  (void)std::transform(group_devices.begin(), group_devices.end(), std::back_inserter(group_ranks),
                       [](const Device &dev) { return dev.rank(); });
  ValuePtr attr_ranks_value = MakeValue(group_ranks);
  Attr attr_ranks = std::make_pair(GROUP_RANKS, attr_ranks_value);
  OperatorAttrs attrs = {attr, attr_ranks};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  op_ = std::make_pair(ALL_GATHER, args);
  return Status::SUCCESS;
}

Status ConstructOperator::ConcatOP(int64_t concat_dim) {
  if (LongToSize(concat_dim) >= tensor_shape_.size() || concat_dim < 0) {
    MS_LOG(ERROR) << "Invalid tensor dimension " << concat_dim << " when construct Concat operator!";
    return Status::INVALID_ARGUMENT;
  }
  ValuePtr attr_value = MakeValue(concat_dim);
  Attr attr = std::make_pair(AXIS, attr_value);
  OperatorAttrs attrs = {attr};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  op_ = std::make_pair(CONCAT, args);
  return Status::SUCCESS;
}

Status ConstructOperator::SplitOP(int64_t split_count) {
  // tensor_shape_ can not be validated here
  if (split_count <= 0) {
    MS_LOG(ERROR) << "Invalid split count when construct Split operator!";
    return Status::FAILED;
  }
  OperatorAttrs attrs;
  ValuePtr attr_value_axis = MakeValue(DEFAULT);
  Attr attr_axis = std::make_pair(AXIS, attr_value_axis);
  ValuePtr attr_value_split = MakeValue(split_count);
  Attr attr_split = std::make_pair(OUTPUT_NUM, attr_value_split);
  attrs = {attr_axis, attr_split};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  op_ = std::make_pair(SPLIT, args);
  return Status::SUCCESS;
}

Status ConstructOperator::AlltoAllOP(const Args &args) {
  if (args.size() < TRANSFER_PERMUTE_ARGS_SIZE) {
    MS_LOG(ERROR) << "args size should not be less than 5!";
    return Status::FAILED;
  }
  int64_t split_count = args[TRANSFER_PERMUTE_SPLIT_COUNT_INDEX];
  int64_t split_dim = args[TRANSFER_PERMUTE_SPLIT_DIM_INDEX];
  int64_t concat_dim = args[TRANSFER_PERMUTE_CONCAT_DIM_INDEX];
  int64_t dev_dim = args[TRANSFER_PERMUTE_DEV_DIM_INDEX];
  if (split_count <= 0) {
    MS_LOG(ERROR) << "Invalid split count when construct AlltoAll operator!";
    return Status::FAILED;
  }
  if (tensor_shape_[LongToSize(split_dim)] % split_count != 0) {
    MS_LOG(ERROR) << "Tensor can not be split into " << split_count << " slices in the dimension " << split_dim
                  << "when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }
  if (LongToSize(concat_dim) >= tensor_shape_.size() || concat_dim < 0) {
    MS_LOG(ERROR) << "Invalid split count " << split_count << " when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }
  if ((LongToSize(dev_dim) >= dev_size_) || (dev_dim < 0)) {
    MS_LOG(ERROR) << "Invalid device dimension " << dev_dim << " when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(dev_size_ - LongToSize(dev_dim) - 1, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << "AlltoAll op: create group failed";
    return FAILED;
  } else if (group_list.empty()) {  // this group only has one device, don't need do alltoall
    MS_LOG(INFO) << "no need all to all op";
    return SUCCESS;
  }

  std::string group_name = group_list[0].name();
  ValuePtr attr_value_group = MakeValue(group_name);
  Attr attr_group = std::make_pair(GROUP, attr_value_group);
  ValuePtr attr_value_split_count = MakeValue(split_count);
  Attr attr_split_count = std::make_pair(SPLIT_COUNT, attr_value_split_count);
  ValuePtr attr_value_split_dim = MakeValue(split_dim);
  Attr attr_split_dim = std::make_pair(SPLIT_DIM, attr_value_split_dim);
  ValuePtr attr_value_concat_dim = MakeValue(concat_dim);
  Attr attr_concat_dim = std::make_pair(CONCAT_DIM, attr_value_concat_dim);
  OperatorAttrs attrs = {attr_split_count, attr_split_dim, attr_concat_dim, attr_group};
  OperatorParams params;
  OperatorArgs op_args = std::make_pair(attrs, params);
  op_ = std::make_pair(ALL_TO_ALL, op_args);
  return Status::SUCCESS;
}

Status ConstructOperator::CreateGroupByDim(size_t axis, std::vector<Group> *group) {
  MS_EXCEPTION_IF_NULL(group);
  auto rank = ParallelContext::GetInstance()->global_rank();
  if (!ParallelContext::GetInstance()->do_transform()) {
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    rank = g_device_manager->global_rank();
  }
  DeviceMatrix dev_matrix(rank, dev_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(axis), &group_devices) != SUCCESS) {
    return FAILED;
  }
  // this group only has one device, don't need create the group
  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "the group is empty";
    return SUCCESS;
  }
  if (is_cost_model_ || ParallelContext::GetInstance()->do_transform()) {
    Group g;
    std::vector<Device> dev_list;
    (void)std::transform(group_devices.begin(), group_devices.end(), std::back_inserter(dev_list),
                         [](auto &rank_id) { return Device(rank_id); });
    (void)g.Init("fake_group", dev_list);
    group->push_back(g);
    if (ParallelContext::GetInstance()->do_transform()) {
      return SUCCESS;
    }
    return g_device_manager->CheckDeviceList(group_devices);
  }
  Group g;
  if (g_device_manager->CreateGroup(group_devices, &g) != SUCCESS) {
    MS_LOG(ERROR) << "Create communication group in redistribution failed, the rank_list is: " << group_devices;
    return FAILED;
  }
  group->push_back(g);
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
