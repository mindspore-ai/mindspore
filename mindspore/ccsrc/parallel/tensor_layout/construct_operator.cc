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

#include "parallel/tensor_layout/construct_operator.h"

#include <functional>
#include <numeric>

namespace mindspore {
namespace parallel {
Status ConstructOperator::Init(const RankList &dev_list, const Shape &dev_matrix_shape) {
  dev_size_ = dev_matrix_shape.size();
  dev_matrix_shape_ = dev_matrix_shape;
  dev_list_ = dev_list;
  return Status::SUCCESS;
}

Status ConstructOperator::ReshapeOP(Shape shape) {
  int32_t prod = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  int32_t prod_expect = std::accumulate(tensor_shape_.begin(), tensor_shape_.end(), 1, std::multiplies<int>());
  if (prod != prod_expect) {
    ValuePtr ptr = MakeValue(shape);
    MS_EXCEPTION_IF_NULL(ptr);
    MS_LOG(ERROR) << "Invalid tensor shape " << ptr->ToString() << "when construct Reshape operator!";
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

Operator CreateStridedSliceOp(int32_t value, const Shape &begin, const Shape &end, const Shape &strides) {
  ValuePtr attr_value = MakeValue(value);
  Attr attr_begin_mask = std::make_pair(BEGIN_MASK, attr_value);
  Attr attr_end_mask = std::make_pair(END_MASK, attr_value);
  Attr attr_ellipsis_mask = std::make_pair(ELLIPSIS_MASK, attr_value);
  Attr attr_new_axis_mask = std::make_pair(NEW_AXIS_MASK, attr_value);
  Attr attr_shrink_axis_mask = std::make_pair(SHRINK_AXIS_MASK, attr_value);
  OperatorAttrs attrs = {attr_begin_mask, attr_end_mask, attr_ellipsis_mask, attr_new_axis_mask, attr_shrink_axis_mask};

  ValuePtr param_begin_value = MakeValue(begin);
  Param param_begin = std::make_pair(std::make_pair(BEGIN, param_begin_value), 2);
  ValuePtr param_end_value = MakeValue(end);
  Param param_end = std::make_pair(std::make_pair(END, param_end_value), 3);

  ValuePtr param_strides_value = MakeValue(strides);
  Param param_strides = std::make_pair(std::make_pair(STRIDES, param_strides_value), 4);
  OperatorParams params = {param_begin, param_end, param_strides};
  OperatorArgs op_args = std::make_pair(attrs, params);

  return std::make_pair(STRIDED_SLICE, op_args);
}

Status ConstructOperator::StridedSliceOP(Args args) {
  if (args.size() < 3) {
    MS_LOG(ERROR) << "args size should not be less than 3!";
    return Status::FAILED;
  }
  int32_t split_count = args[0];
  if (split_count <= 0) {
    MS_LOG(ERROR) << "split_count should not be less than 0!";
    return Status::FAILED;
  }
  int32_t split_dim = args[1];
  int32_t dev_dim = args[2];
  std::vector<Group> group_list;

  if (CreateGroupByDim(dev_size_ - IntToSize(dev_dim) - 1, &group_list) != SUCCESS) {
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
    if (index != IntToSize(split_dim)) {
      begin[index] = 0;
      end[index] = num;
    } else {
      if (num % split_count != 0) {
        MS_LOG(ERROR) << "Tensor can not be split into " << split_count << " slices in the dimension " << split_dim
                      << "! when construct StridedSlice operator";
        return Status::INVALID_ARGUMENT;
      }
      int32_t count = num / split_count;
      begin[index] = SizeToInt(rank) * count;
      end[index] = (SizeToInt(rank) + 1) * count;
    }
    index++;
  }

  op_ = CreateStridedSliceOp(DEFAULT, begin, end, strides);

  return Status::SUCCESS;
}

Status ConstructOperator::AllGatherOP(int32_t dev_dim) {
  if ((IntToSize(dev_dim) >= dev_size_) || (dev_dim < 0)) {
    MS_LOG(ERROR) << "Invalid device dimension " << dev_dim << " when construct AllGather operator!";
    return Status::INVALID_ARGUMENT;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(dev_size_ - IntToSize(dev_dim) - 1, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << "AllGather op: create group failed";
    return FAILED;
  } else if (group_list.empty()) {  // this group only has one device, don't need do allgather
    MS_LOG(INFO) << "no need all gather op";
    return SUCCESS;
  }

  std::string group_name = group_list[0].name();
  ValuePtr attr_value = MakeValue(group_name);
  Attr attr = std::make_pair(GROUP, attr_value);
  OperatorAttrs attrs = {attr};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  op_ = std::make_pair(ALL_GATHER, args);
  return Status::SUCCESS;
}

Status ConstructOperator::ConcatOP(int32_t concat_dim) {
  if (IntToSize(concat_dim) >= tensor_shape_.size()) {
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

Status ConstructOperator::SplitOP(int32_t split_count) {
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

Status ConstructOperator::AlltoAllOP(Args args) {
  if (args.size() < 4) {
    MS_LOG(ERROR) << "args size should not be less than 4!";
    return Status::FAILED;
  }
  int32_t split_count = args[0];
  int32_t split_dim = args[1];
  int32_t concat_dim = args[2];
  int32_t dev_dim = args[3];
  if (split_count <= 0) {
    MS_LOG(ERROR) << "Invalid split count when construct AlltoAll operator!";
    return Status::FAILED;
  }
  if (tensor_shape_[IntToSize(split_dim)] % split_count != 0) {
    MS_LOG(ERROR) << "Tensor can not be split into " << split_count << " slices in the dimension " << split_dim
                  << "when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }
  if (IntToSize(concat_dim) >= tensor_shape_.size()) {
    MS_LOG(ERROR) << "Invalid split count " << split_count << " when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }
  if ((IntToSize(dev_dim) >= dev_size_) || (dev_dim < 0)) {
    MS_LOG(ERROR) << "Invalid device dimension " << dev_dim << " when construct AlltoAll operator!";
    return Status::INVALID_ARGUMENT;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(dev_size_ - IntToSize(dev_dim) - 1, &group_list) != SUCCESS) {
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
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  int32_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, dev_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUint(axis), &group_devices) != SUCCESS) {
    return FAILED;
  }
  // this group only has one device, don't need create the group
  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "the group is empty";
    return SUCCESS;
  }

  Group g = g_device_manager->CreateGroup(group_devices);
  group->push_back(g);
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
