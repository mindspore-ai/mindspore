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

#include "frontend/parallel/ops_info/conv2d_info.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
namespace {
ValuePtr MakeListValue(const std::vector<int64_t> &v) {
  std::vector<ValuePtr> list;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(list), [](int64_t ele) { return MakeValue(ele); });
  return std::make_shared<ValueSequeue>(list);
}

ValuePtr MakeTupleListValue(const Shapes &v) {
  std::vector<ValuePtr> tuple;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(tuple),
                       [](const std::vector<int64_t> &list) { return MakeListValue(list); });
  return std::make_shared<ValueTuple>(tuple);
}
}  // namespace
Status Conv2DInfo::GetAttrsBase() {
  // format
  format_ = GetStringAttr(FORMAT);
  if (format_ != NCHW) {
    MS_LOG(ERROR) << name_ << ": The format must be 'NCHW', but got " << format_;
    return FAILED;
  }

  // out_channel
  out_channel_ = GetIntAttr(OUT_CHANNEL);
  if (out_channel_ <= 0) {
    MS_LOG(ERROR) << name_ << ": The attr of out_channel is invalid";
    return FAILED;
  }

  // kernel_size
  auto kernel_size_iter = attrs_.find(KERNEL_SIZE);
  if (kernel_size_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attribution of " << KERNEL_SIZE;
    return FAILED;
  }

  MS_EXCEPTION_IF_NULL(kernel_size_iter->second);
  if (kernel_size_iter->second->isa<Int64Imm>()) {
    int64_t kernel_size = kernel_size_iter->second->cast<Int64ImmPtr>()->value();
    kernel_size_ = {kernel_size, kernel_size};
  } else if (kernel_size_iter->second->isa<ValueTuple>() || kernel_size_iter->second->isa<ValueList>()) {
    kernel_size_ = GetValue<std::vector<int64_t>>(kernel_size_iter->second);
    if (kernel_size_.size() != 2) {
      MS_LOG(ERROR) << name_ << ": The size of kernel_size'tuple must be 2, but got " << kernel_size_.size();
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": The kernel_size must be int or tuple";
    return FAILED;
  }

  // mode
  mode_ = GetIntAttr(MODE);
  if (mode_ != 1) {
    MS_LOG(ERROR) << name_ << ": The mode must be 1, but got " << mode_;
    return FAILED;
  }

  // pad_mode
  pad_mode_ = GetIntAttr(PAD_MODE);
  if (pad_mode_ < 0 || pad_mode_ > 2) {
    MS_LOG(ERROR) << name_ << ": The pad_mode must be in the range of [0, 2], but got " << pad_mode_;
    return FAILED;
  }

  // pad_list
  pad_list_ = GetTupleIntAttr(PAD_LIST);
  if (pad_list_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of pad_list must be 4, but got " << pad_list_.size();
    return FAILED;
  }

  // stride
  stride_ = GetTupleIntAttr(STRIDE);
  if (stride_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of stride must be 4, but got " << stride_.size();
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of stride must be 1, but got (" << stride_[0] << ", "
                  << stride_[1] << ")";
    return FAILED;
  }

  // dilation
  dilation_ = GetTupleIntAttr(DILATION);
  if (dilation_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of dilation must be 4, but got " << dilation_.size();
    return FAILED;
  }

  // group
  group_ = GetIntAttr(GROUP);

  MS_LOG(INFO) << name_ << ": The out channel is " << out_channel_ << ", kernel size is " << kernel_size_
               << ", mode is " << mode_ << ", pad mode is " << pad_mode_ << ", pad list is " << pad_list_
               << ", stride is " << stride_ << ", dilation is " << dilation_ << ", group is " << group_
               << ", format is " << format_;

  return SUCCESS;
}

Status Conv2DInfo::GetAttrs() { return GetAttrsBase(); }

Status Conv2DInfo::CheckHWStrategyBase(int64_t h_strategy, int64_t w_strategy) const {
  if (outputs_shape_[0][2] % h_strategy != 0) {
    MS_LOG(ERROR) << name_
                  << ": Do not support to split h dimension when out_shape of h dimension is not divisible by strategy "
                     "of h dimension";
    return FAILED;
  }

  if (outputs_shape_[0][3] % w_strategy != 0) {
    MS_LOG(ERROR) << name_
                  << ": Do not support to split w dimension when out_shape of w dimension is not divisible by strategy "
                     "of w dimension";
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategySameMode(int64_t h_strategy, int64_t w_strategy) {
  int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
  int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;

  // H dimension
  if (kernel_size_[0] > stride_[2] && h_strategy > 1) {
    MS_LOG(ERROR) << name_ << ": The 'same' mode do not support to split H when kernel_size > stride";
    return FAILED;
  }

  if (h_strategy > 1 && (kernel_size_[0] <= stride_[2] && h_slice_shape % stride_[2] != 0)) {
    MS_LOG(ERROR) << name_
                  << ": The 'same' mode do not support to split H when kernel_size <= stride but slice shape "
                     "is not divisible by stride ";
    return FAILED;
  }

  // W dimension
  if (w_strategy > 1 && (kernel_size_[1] <= stride_[3] && w_slice_shape % stride_[3] != 0)) {
    MS_LOG(ERROR) << name_
                  << ": The 'same' mode do not support to split W when kernel_size <= stride but slice shape "
                     "is not divisible by stride ";
    return FAILED;
  }

  if (w_strategy > 1 && (kernel_size_[1] > stride_[3])) {
    if (inputs_shape_[0][3] % stride_[3] != 0) {
      MS_LOG(ERROR) << name_
                    << ": The 'same' mode do not support to split W when kernel_size > stride but w shape is not "
                       "divisible by stride";
      return FAILED;
    }

    if (w_slice_shape <= ((kernel_size_[1] - stride_[3] + 1) / 2)) {
      MS_LOG(ERROR) << name_
                    << ": The 'same' mode do not support to split W when kernel_size > stride but w slice shape is "
                       "smaller than or equal to (k - s + 1) / 2";
      return FAILED;
    }

    if (kernel_size_[1] - stride_[3] == 1) {
      MS_LOG(ERROR) << name_ << ": The 'same' mode do not support to split W when kernel_size > stride but k - s == 1";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategyValidMode(int64_t h_strategy, int64_t w_strategy) {
  int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
  int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;

  if ((kernel_size_[0] > stride_[2] && h_strategy > 1) || (kernel_size_[1] > stride_[3] && w_strategy > 1)) {
    MS_LOG(ERROR) << name_ << ": The 'valid' mode do not support to split H or W when kernel_size > stride";
    return FAILED;
  }

  if (kernel_size_[0] <= stride_[2] && h_slice_shape % stride_[2] != 0) {
    MS_LOG(ERROR) << name_
                  << ": The 'valid' mode do not support to split H when kernel_size <= stride but slice shape is "
                     "not divisible by stride ";
    return FAILED;
  }

  if (kernel_size_[1] <= stride_[3] && w_slice_shape % stride_[3] != 0) {
    MS_LOG(ERROR) << name_
                  << ": The 'valid' mode do not support to split W when kernel_size <= stride but slice shape is "
                     "not divisible by stride ";
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (CheckHWStrategyBase(h_strategy, w_strategy) != SUCCESS) {
    return FAILED;
  }

  if (pad_mode_ == 0) {  // 'pad' mode
    MS_LOG(ERROR) << name_ << ": The 'pad' mode do not support to split H or W";
    return FAILED;
  }

  if (pad_mode_ == 1) {  // 'same' mode
    return CheckHWStrategySameMode(h_strategy, w_strategy);
  }

  if (pad_mode_ == 2) {  // 'valid' mode
    return CheckHWStrategyValidMode(h_strategy, w_strategy);
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckStrategyBase(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy.size() != 4 || weight_strategy.size() != 4) {
    MS_LOG(ERROR) << name_
                  << ": The size of input strategy or weight strategy must be 4, but the size of input strategy is "
                  << input_strategy.size() << ", the size of weight strategy is " << weight_strategy.size();
    return FAILED;
  }

  if (weight_strategy[2] != 1 || weight_strategy[3] != 1) {
    MS_LOG(ERROR) << name_ << ": The kernel size can not be split, but the strategy for kernel size is ("
                  << weight_strategy[2] << ", " << weight_strategy[3] << ")";
    return FAILED;
  }

  if (weight_strategy[0] > 1) {
    out_channel_shard_ = true;
    new_out_channel_ = out_channel_ / weight_strategy[0];
  } else {
    out_channel_shard_ = false;
    new_out_channel_ = out_channel_;
  }

  int64_t input_except_n_shards =
    std::accumulate(input_strategy.begin() + 1, input_strategy.end(), 1, std::multiplies<int64_t>());
  int64_t weight_shards =
    std::accumulate(weight_strategy.begin() + 1, weight_strategy.end(), 1, std::multiplies<int64_t>());

  bool is_data_parallel = (input_except_n_shards * weight_shards == 1);
  if (!is_data_parallel) {
    if (std::any_of(dilation_.begin(), dilation_.end(), [](int64_t value) { return value != 1; })) {
      MS_LOG(ERROR) << name_ << ": If it is not data parallel, the value of dilation must be 1, but got " << dilation_;
      return FAILED;
    }

    if (group_ != 1) {
      MS_LOG(ERROR) << name_ << ": If it is not data parallel, the group must be 1, but got " << group_;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status Conv2DInfo::CheckStrategy(const StrategyPtr &strategy) {
  need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy[1] != weight_strategy[1]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-in for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-in for weight strategy is " << weight_strategy[1];
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  // kernel size larger than stride and the w dimension is split, need to exchange overlap
  if ((kernel_size_[1] > stride_[3]) && (input_strategy[3] > 1)) {
    need_exchange_overlap_ = true;
  }

  return SUCCESS;
}

Status Conv2DInfo::InferDevMatrixShape() {
  // the strategy is ((n, i, h, w), (o, i, 1, 1))
  // the dev matrix is (n, i, h, w, o)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  dev_matrix_shape_.push_back(stra[1][0]);
  w_dimension_shard_num_ = stra[0][3];
  input_slice_shape_ = GetSliceShape(inputs_shape_[0], stra[0]);
  return SUCCESS;
}

Status Conv2DInfo::InferRankBias() {
  // the Conv2D operator:
  // the origin dev_matrix is [n, i, h, w, o]
  // if repeated calculation and repeated num in the left of dev matrix, the dev_matrix is [repeated_num, n, i, h, w, o]
  // if repeated calculation and repeated num in the right of dev matrix, the dev_matrix is [n, i, h, w, o,
  // repeated_num]
  //
  // the Conv2DBackpropInput's origin dev_matrix is [n, o, h, w, i], w dimension's relative position is the same as
  // Conv2D, the rank_bias_ is the position of the current rank in the w dimension of the dev_matrix(have not split h
  // dimension)
  if (!need_exchange_overlap_) {
    MS_LOG(INFO) << name_ << ": No need to infer rank bias";
    return SUCCESS;
  }

  uint64_t w_index_in_dev_matrix = 3;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    w_index_in_dev_matrix += 1;
  }

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(w_index_in_dev_matrix, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() <= 1) {
    MS_LOG(INFO) << name_ << ": The devices' size of w dimension is " << group_devices.size()
                 << ", no need to infer rank bias";
    return SUCCESS;
  }

  if (group_devices.size() != LongToSize(w_dimension_shard_num_)) {
    MS_LOG(ERROR) << name_ << ": The devices' size of w dimension is " << group_devices.size()
                  << ", but the shard num of w dimension is " << w_dimension_shard_num_;
    return FAILED;
  }

  std::vector<int64_t>::iterator it = std::find(group_devices.begin(), group_devices.end(), rank);
  if (it == group_devices.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the current rank in device list of w dimension, the current rank is "
                  << rank << ", the device list is " << group_devices;
    return FAILED;
  }

  rank_bias_ = std::distance(group_devices.begin(), it);
  if (it == group_devices.begin()) {
    left_rank_bias_ = -1;
    right_rank_bias_ = rank_bias_ + 1;

    left_rank_id_ = -1;
    right_rank_id_ = *(it + 1);
  } else if (it == group_devices.end() - 1) {
    left_rank_bias_ = rank_bias_ - 1;
    right_rank_bias_ = -1;

    left_rank_id_ = *(it - 1);
    right_rank_id_ = -1;
  } else {
    left_rank_bias_ = rank_bias_ - 1;
    right_rank_bias_ = rank_bias_ + 1;

    left_rank_id_ = *(it - 1);
    right_rank_id_ = *(it + 1);
  }
  MS_LOG(INFO) << name_ << ": The current rank is " << rank << ", the device list of w dimension is " << group_devices
               << ", the rank bias is " << rank_bias_ << ", the left rank bias is " << left_rank_bias_
               << ", the right rank bias is " << right_rank_bias_ << ", the left rank id is " << left_rank_id_
               << ", the right rank id is " << right_rank_id_;
  return SUCCESS;
}

int64_t Conv2DInfo::ComputeOverlapLeftSizeByRankBias(int64_t rank_bias) {
  int64_t left_pad = pad_list_[2];
  int64_t w_dimension_input_shape = inputs_shape_[0][3];
  int64_t w_dimension_output_shape = outputs_shape_[0][3];
  int64_t w_stride = stride_[3];

  return left_pad +
         (w_dimension_input_shape - w_dimension_output_shape * w_stride) * rank_bias / w_dimension_shard_num_;
}

int64_t Conv2DInfo::ComputeOverlapRightSizeByRankBias(int64_t rank_bias) {
  int64_t left_pad = pad_list_[2];
  int64_t w_dimension_input_shape = inputs_shape_[0][3];
  int64_t w_dimension_output_shape = outputs_shape_[0][3];
  int64_t w_kernel_size = kernel_size_[1];
  int64_t w_stride = stride_[3];

  return (rank_bias + 1) * (w_dimension_output_shape * w_stride - w_dimension_input_shape) / w_dimension_shard_num_ +
         w_kernel_size - w_stride - left_pad;
}

void Conv2DInfo::InferOverlapSize() {
  if (!need_exchange_overlap_) {
    MS_LOG(INFO) << name_ << ": No need to infer overlap size";
    return;
  }

  overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(rank_bias_);
  overlap_right_size_ = ComputeOverlapRightSizeByRankBias(rank_bias_);

  if (rank_bias_ == 0) {  // it has not left rank
    left_rank_overlap_left_size_ = 0;
    left_rank_overlap_right_size_ = 0;
    right_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(right_rank_bias_);
    right_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(right_rank_bias_);
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {  // it has not right rank
    left_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(left_rank_bias_);
    left_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = 0;
    right_rank_overlap_right_size_ = 0;
  } else {  // it has left rank and right rank
    left_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(left_rank_bias_);
    left_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(right_rank_bias_);
    right_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(right_rank_bias_);
  }

  MS_LOG(INFO) << name_ << ": the left overlap size of current rank is " << overlap_left_size_
               << ", the right overlap size of current rank is " << overlap_right_size_
               << ", the left overlap size of left rank is " << left_rank_overlap_left_size_
               << ", the right overlap size of left rank is " << left_rank_overlap_right_size_
               << ", the left overlap size of right rank is " << right_rank_overlap_left_size_
               << ", the right overlap size of right rank is " << right_rank_overlap_right_size_;
}

Status Conv2DInfo::InferTensorMap() {
  // input_strategy: ((n, i, h, w), (o, i, 1, 1))
  // output_strategy: ((n, o, h, w),)
  // dev_matrix: (n, i, h, w, o)
  TensorMap input_tensor_map = {4, 3, 2, 1};
  TensorMap weight_tensor_map = {0, 3, -1, -1};
  TensorMap output_tensor_map = {4, 0, 2, 1};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)inputs_tensor_map_.emplace_back(std::move(weight_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

// Conv2d: dev_matrix is (n, i, h, w, o), if in channel is split, it need to insert all reduce
// Conv2DBackpropInputInfo: dev_matrix is (n, o, h, w, i), if out channel is split, it need to insert all reduce
Status Conv2DInfo::InferForwardCommunication() {
  forward_op_.clear();
  size_t relevant_dim_index = IN_CHANNEL_INDEX;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    // if repeated calculation and repeated num in the left of dev matrix, the index of relevant dimension should add 1
    relevant_dim_index += 1;
  }

  if (dev_matrix_shape_[relevant_dim_index] == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(relevant_dim_index, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed";
    return FAILED;
  }

  if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  Operator op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward all reduce is " << group_list[0].name();

  return SUCCESS;
}

void Conv2DInfo::InferNewPadList() {
  new_pad_list_ = pad_list_;
  if (rank_bias_ == 0) {                                  // the first rank
    new_pad_list_[3] = 0;                                 // no need the right pad
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {  // the last rank
    new_pad_list_[2] = 0;                                 // no need the left pad
  } else {                                                // the middle rank
    new_pad_list_[2] = 0;                                 // no need the left pad
    new_pad_list_[3] = 0;                                 // no need the right pad
  }
  MS_LOG(INFO) << name_ << ": the new pad list is " << new_pad_list_;
}

void Conv2DInfo::InferSendRecvFlag() {
  if (rank_bias_ == 0) {  // the first rank
    left_need_send_ = false;
    left_need_recv_ = false;
    right_need_send_ = (right_rank_overlap_left_size_ > 0);
    right_need_recv_ = (overlap_right_size_ > 0);         // no need the right pad
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {  // the last rank
    left_need_send_ = (left_rank_overlap_right_size_ > 0);
    left_need_recv_ = (overlap_left_size_ > 0);
    right_need_send_ = false;
    right_need_recv_ = false;
  } else {  // the middle rank
    left_need_send_ = (left_rank_overlap_right_size_ > 0);
    left_need_recv_ = (overlap_left_size_ > 0);
    right_need_send_ = (right_rank_overlap_left_size_ > 0);
    right_need_recv_ = (overlap_right_size_ > 0);
  }
  MS_LOG(INFO) << name_ << ": The left need send is " << left_need_send_ << ", the left need recv is "
               << left_need_recv_ << ", the right need send is " << right_need_send_ << ", the right need recv is "
               << right_need_recv_;

  if (left_need_send_) {
    if (left_rank_overlap_right_size_ >= input_slice_shape_[3]) {
      MS_LOG(EXCEPTION) << name_ << ": Do not support left overlap size(" << left_rank_overlap_right_size_
                        << ") larger than or equal to slice shape in w dimension(" << input_slice_shape_[3] << ")";
    }
    send_rank_ids_.push_back(left_rank_id_);
  }

  if (right_need_send_) {
    if (right_rank_overlap_left_size_ >= input_slice_shape_[3]) {
      MS_LOG(EXCEPTION) << name_ << ": Do not support left overlap size(" << right_rank_overlap_left_size_
                        << ") larger than or equal to slice shape in w dimension(" << input_slice_shape_[3] << ")";
    }
    send_rank_ids_.push_back(right_rank_id_);
  }

  if (left_need_recv_) {
    recv_rank_ids_.push_back(left_rank_id_);
  }

  if (right_need_recv_) {
    recv_rank_ids_.push_back(right_rank_id_);
  }

  MS_LOG(INFO) << name_ << ": The send rank ids is " << send_rank_ids_ << ", the recv rank ids is " << recv_rank_ids_;
}

void Conv2DInfo::InferOverlapShapes() {
  if (left_need_recv_) {
    Shape left_recv_shape = input_slice_shape_;
    left_recv_shape[3] = overlap_left_size_;
    recv_shapes_.push_back(left_recv_shape);
  }

  if (right_need_recv_) {
    Shape right_recv_shape = input_slice_shape_;
    right_recv_shape[3] = overlap_right_size_;
    recv_shapes_.push_back(right_recv_shape);
  }

  if (left_need_send_) {
    Shape left_send_shape = input_slice_shape_;
    left_send_shape[3] = left_rank_overlap_right_size_;
    send_shapes_.push_back(left_send_shape);
  }

  if (right_need_send_) {
    Shape right_send_shape = input_slice_shape_;
    right_send_shape[3] = right_rank_overlap_left_size_;
    send_shapes_.push_back(right_send_shape);
  }
  MS_LOG(INFO) << name_ << ": the recv shapes is " << recv_shapes_ << ", the send shapes is " << send_shapes_;
}

void Conv2DInfo::InferStridedSliceAttrs() {
  if (left_need_send_) {
    left_strided_slice_begin_ = {0, 0, 0, 0};
    left_strided_slice_end_ = input_slice_shape_;
    left_strided_slice_end_[3] = left_rank_overlap_right_size_;
    left_strided_slice_strides_ = {1, 1, 1, 1};
    MS_LOG(INFO) << name_ << ": The left strided slice begin is " << left_strided_slice_begin_ << ", end is "
                 << left_strided_slice_end_;
  }

  if (right_need_send_) {
    right_strided_slice_begin_ = {0, 0, 0, 0};
    right_strided_slice_begin_[3] = input_slice_shape_[3] - right_rank_overlap_left_size_;
    right_strided_slice_end_ = input_slice_shape_;
    right_strided_slice_strides_ = {1, 1, 1, 1};
    MS_LOG(INFO) << name_ << ": The right strided slice begin is " << right_strided_slice_begin_ << ", end is "
                 << right_strided_slice_end_;
  }
}

void Conv2DInfo::InferNewOperatorAttrs() {
  InferNewPadList();

  InferSendRecvFlag();

  InferOverlapShapes();

  InferStridedSliceAttrs();
}

OperatorAttrs Conv2DInfo::CreateNeighborExchangeAttrs(const CNodePtr &cnode) {
  auto type = cnode->Type();
  MS_EXCEPTION_IF_NULL(type);
  auto tensor_type = type->cast<mindspore::TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto dtype = tensor_type->element();
  MS_EXCEPTION_IF_NULL(dtype);
  Attr send_ranks = {SEND_RNAK_IDS, MakeListValue(send_rank_ids_)};
  Attr recv_ranks = {RECV_RNAK_IDS, MakeListValue(recv_rank_ids_)};
  Attr send_shapes = {SEND_SHAPES, MakeTupleListValue(send_shapes_)};
  Attr recv_shapes = {RECV_SHAPES, MakeTupleListValue(recv_shapes_)};
  Attr recv_type = {RECV_TYPE, dtype};
  OperatorAttrs attrs = {send_ranks, recv_ranks, recv_shapes, send_shapes, recv_type};
  return attrs;
}

OperatorAttrs Conv2DInfo::CreateConv2DAttrs() {
  Attr out_channel = {OUT_CHANNEL, MakeValue(new_out_channel_)};
  Attr kernel_size = {KERNEL_SIZE, MakeValue(kernel_size_)};
  Attr mode = {MODE, MakeValue(mode_)};
  Attr pad_mode = {PAD_MODE, MakeValue("pad")};
  Attr pad = {PAD, MakeValue(new_pad_list_)};
  Attr stride = {STRIDE, MakeValue(stride_)};
  Attr dilation = {DILATION, MakeValue(dilation_)};
  Attr group = {GROUP, MakeValue(group_)};
  Attr data_format = {DATA_FORMAT, MakeValue(format_)};

  OperatorAttrs attrs;
  if (name_.find(CONV2D_INFO) != std::string::npos) {
    attrs = {out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, data_format};
  } else {  // Conv2DTranspose
    attrs = {out_channel, kernel_size, pad_mode, pad, pad, mode, stride, dilation, group, data_format};
  }

  return attrs;
}

std::string Conv2DInfo::ReplaceNodeName() const {
  if (name_.find(CONV2D_INFO) != std::string::npos) {
    return CONV2D;
  }

  if (name_.find(CONV2D_BACK_PROP_INPUT_INFO) != std::string::npos) {
    return CONV2D_BACK_PROP_INPUT;
  }

  if (name_.find(CONV2D_TRANSPOSE_INFO) != std::string::npos) {
    return CONV2D_TRANSPOSE;
  }

  MS_LOG(EXCEPTION) << "Invalid name: " << name_;
}

AnfNodePtr Conv2DInfo::GenerateConv2DNode(const AnfNodePtr &new_input, const CNodePtr &cnode) {
  auto conv2d_attrs = CreateConv2DAttrs();
  auto node_name = ReplaceNodeName();

  // conv2d
  if (name_.find(CONV2D_INFO) != std::string::npos) {
    if (cnode->size() < 3) {
      MS_LOG(EXCEPTION) << name_ << ": The size of cnode is invalid: " << cnode->size();
    }
    return gen_g_.PushBack({gen_g_.NewOpInst(node_name, conv2d_attrs), new_input, cnode->input(2)});
  }

  // conv2dtranspose
  if (cnode->size() < 4) {
    MS_LOG(EXCEPTION) << name_ << ": The size of cnode is invalid: " << cnode->size();
  }
  return gen_g_.PushBack({gen_g_.NewOpInst(node_name, conv2d_attrs), new_input, cnode->input(2), cnode->input(3)});
}

void Conv2DInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  if (gen_g_.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << "GenerateGraph Init failed";
  }

  if (!left_need_send_ && !right_need_send_) {
    MS_LOG(EXCEPTION) << name_ << ": Now do not support left no need to send and right no need to send";
  }

  if (!left_need_recv_ && !right_need_recv_) {
    MS_LOG(EXCEPTION) << name_ << ": Now do not support left no need to recv and right no need to recv";
  }

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes;
  std::vector<AnfNodePtr> make_tuple_a_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  if (left_need_send_) {
    auto slice_left_begin = CreateTuple(left_strided_slice_begin_);
    auto slice_left_end = CreateTuple(left_strided_slice_end_);
    auto slice_left_strided = CreateTuple(left_strided_slice_strides_);
    auto slice_left = gen_g_.PushBack({gen_g_.NewOpInst(STRIDED_SLICE), gen_g_.virtual_input_node(), slice_left_begin,
                                       slice_left_end, slice_left_strided});
    make_tuple_a_inputs.push_back(slice_left);
    input_nodes.push_back(std::make_pair(slice_left, 1));
  }
  if (right_need_send_) {
    auto slice_right_begin = CreateTuple(right_strided_slice_begin_);
    auto slice_right_end = CreateTuple(right_strided_slice_end_);
    auto slice_right_strided = CreateTuple(right_strided_slice_strides_);
    auto slice_right = gen_g_.PushBack({gen_g_.NewOpInst(STRIDED_SLICE), gen_g_.virtual_input_node(), slice_right_begin,
                                        slice_right_end, slice_right_strided});
    make_tuple_a_inputs.push_back(slice_right);
    input_nodes.push_back(std::make_pair(slice_right, 1));
  }

  auto make_tuple_a = graph->NewCNode(make_tuple_a_inputs);
  auto alltoall_attrs = CreateNeighborExchangeAttrs(cnode);
  auto alltoall_v = gen_g_.PushBack({gen_g_.NewOpInst(NEIGHBOREXCHANGE, alltoall_attrs), make_tuple_a});

  AnfNodePtr conv2d;
  Attr concat_axis = {AXIS, MakeValue(-1)};
  OperatorAttrs concat_attrs = {concat_axis};

  if (left_need_recv_) {
    std::vector<AnfNodePtr> tuple_getitem_l_inputs = {NewValueNode(prim::kPrimTupleGetItem), alltoall_v,
                                                      CreatInt64Imm(0)};
    auto tuple_getitem_l = graph->NewCNode(tuple_getitem_l_inputs);
    std::vector<AnfNodePtr> make_tuple_l_inputs = {NewValueNode(prim::kPrimMakeTuple), tuple_getitem_l,
                                                   cnode->input(1)};
    auto make_tuple_l = graph->NewCNode(make_tuple_l_inputs);
    auto concat_l = gen_g_.PushBack({gen_g_.NewOpInst(CONCAT, concat_attrs), make_tuple_l});

    if (right_need_recv_) {
      std::vector<AnfNodePtr> tuple_getitem_r_inputs = {NewValueNode(prim::kPrimTupleGetItem), alltoall_v,
                                                        CreatInt64Imm(1)};
      auto tuple_getitem_r = graph->NewCNode(tuple_getitem_r_inputs);
      std::vector<AnfNodePtr> make_tuple_r_inputs = {NewValueNode(prim::kPrimMakeTuple), concat_l, tuple_getitem_r};
      auto make_tuple_r = graph->NewCNode(make_tuple_r_inputs);
      auto concat_r = gen_g_.PushBack({gen_g_.NewOpInst(CONCAT, concat_attrs), make_tuple_r});
      conv2d = GenerateConv2DNode(concat_r, cnode);
    } else {
      conv2d = GenerateConv2DNode(concat_l, cnode);
    }
  } else {  // left no need recv, and right need recv
    std::vector<AnfNodePtr> tuple_getitem_r_inputs_1 = {NewValueNode(prim::kPrimTupleGetItem), alltoall_v,
                                                        CreatInt64Imm(0)};
    auto tuple_getitem_r_1 = graph->NewCNode(tuple_getitem_r_inputs_1);
    std::vector<AnfNodePtr> make_tuple_r_inputs_1 = {NewValueNode(prim::kPrimMakeTuple), gen_g_.virtual_input_node(),
                                                     tuple_getitem_r_1};
    auto make_tuple_r_1 = graph->NewCNode(make_tuple_r_inputs_1);
    input_nodes.push_back(std::make_pair(make_tuple_r_1, 1));

    auto concat_r_1 = gen_g_.PushBack({gen_g_.NewOpInst(CONCAT, concat_attrs), make_tuple_r_1});
    conv2d = GenerateConv2DNode(concat_r_1, cnode);
  }

  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, conv2d));
}

ReplaceGraphPtr Conv2DInfo::replace_graph(const CNodePtr &cnode) {
  if (!need_exchange_overlap_) {
    if (!out_channel_shard_) {
      return nullptr;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    prim->set_attr(OUT_CHANNEL, MakeValue(new_out_channel_));
    return nullptr;
  }

  if (InferRankBias() != SUCCESS) {
    return nullptr;
  }

  InferOverlapSize();

  InferNewOperatorAttrs();

  ComputeReplaceGraph(cnode);
  return replace_graph_;
}

void Conv2DInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status Conv2DInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> Conv2DInfo::GenerateOpStrategies(int64_t stage_id) {
  Strategys strategy = {{stage_device_size_, 1, 1, 1}, {1, 1, 1, 1}};
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status Conv2DInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status Conv2DInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status Conv2DBackpropInputInfo::GetOutShape() {
  if (input_value_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": The size of input value must be 3, but got " << input_value_.size();
    return FAILED;
  }

  if (input_value_[2] == nullptr) {
    MS_LOG(ERROR) << name_ << ": The input_value_[2] is nullptr";
    return FAILED;
  }

  std::vector<ValuePtr> elements;
  auto value_tuple = input_value_[2]->cast<ValueTuplePtr>();
  if (value_tuple == nullptr) {
    MS_LOG(ERROR) << name_ << ": Input_value_[2] must be ValueTuplePtr.";
    return FAILED;
  }
  elements = value_tuple->value();
  if (elements.size() != 4) {
    MS_LOG(ERROR) << name_ << ": Elements size must be 4, but got " << elements.size();
    return FAILED;
  }

  for (auto &element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t ele_value = element->cast<Int64ImmPtr>()->value();
      out_shape_.push_back(ele_value);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of shape must be int";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Conv2DBackpropInputInfo::GetAttrs() {
  if (GetAttrsBase() != SUCCESS) {
    return FAILED;
  }

  return GetOutShape();
}

Status Conv2DBackpropInputInfo::CheckStrategy(const StrategyPtr &strategy) {
  need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy[1] != weight_strategy[0]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-out for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-out for weight strategy is " << weight_strategy[0];
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  // kernel size larger than stride and the w dimension is split, need to exchange overlap
  if ((kernel_size_[1] > stride_[3]) && (input_strategy[3] > 1)) {
    need_exchange_overlap_ = true;
  }
  return SUCCESS;
}

Status Conv2DBackpropInputInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (CheckHWStrategyBase(h_strategy, w_strategy) != SUCCESS) {
    return FAILED;
  }

  if (pad_mode_ != 1) {  // only support same mode
    MS_LOG(ERROR) << name_ << ": Do not support the pad mode " << pad_mode_ << " when split H or W dimension";
    return FAILED;
  }

  if (h_strategy > 1) {
    MS_LOG(ERROR) << name_ << ": Do not support to split h dimension";
    return FAILED;
  }

  if (w_strategy > 1 && inputs_shape_[0][3] * stride_[3] != outputs_shape_[0][3]) {
    MS_LOG(ERROR) << name_ << ": Do not support to split w dimension when in_shape * stride != out_shape";
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DBackpropInputInfo::InferDevMatrixShape() {
  // the strategy is ((n, o, h, w), (o, i, 1, 1))
  // the dev matrix is (n, o, h, w, i)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  dev_matrix_shape_.push_back(stra[1][1]);

  Shape out_strategy = stra[0];
  out_strategy[1] = stra[1][1];

  out_slice_shape_ = out_shape_;
  if (out_shape_.size() != out_strategy.size()) {
    MS_LOG(ERROR) << name_ << ": The size of out shape is " << out_shape_.size()
                  << ", but the size of output strategy is " << out_strategy.size();
    return FAILED;
  }

  for (size_t i = 0; i < out_slice_shape_.size(); ++i) {
    if (out_slice_shape_[i] % out_strategy[i] != 0) {
      MS_LOG(ERROR) << name_ << ": The output can not be split by strategy. The shape of output is " << out_slice_shape_
                    << ", but the strategy of output is " << out_strategy;
      return FAILED;
    }
    out_slice_shape_[i] = out_slice_shape_[i] / out_strategy[i];
  }

  w_dimension_shard_num_ = stra[0][3];
  input_slice_shape_ = GetSliceShape(inputs_shape_[0], stra[0]);
  MS_LOG(INFO) << name_ << ": The output slice shape is " << out_slice_shape_;
  return SUCCESS;
}

Status Conv2DBackpropInputInfo::InferTensorMap() {
  // input_strategy: ((n, o, h, w), (o, i, 1, 1))
  // output_strategy: ((n, i, h, w),)
  // dev_matrix: (n, o, h, w, i)
  TensorMap input_tensor_map = {4, 3, 2, 1};
  TensorMap weight_tensor_map = {3, 0, -1, -1};
  TensorMap output_tensor_map = {4, 0, 2, 1};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)inputs_tensor_map_.emplace_back(std::move(weight_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status Conv2DBackpropInputInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
    return FAILED;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    std::vector<Group> group;
    if (CreateGroupByTensorMap(inputs_tensor_map_[i], &group) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Create group failed, the input index is " << i;
      mirror_ops_.clear();
      return FAILED;
    }

    OperatorVector mirror_op;
    if (group.empty()) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    group_is_empty = false;
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
    return SUCCESS;
  }

  OperatorVector tmp_mirror_op;  // tmp mirror op for 'out_shape'
  mirror_ops_.push_back(tmp_mirror_op);
  return SUCCESS;
}

void Conv2DBackpropInputInfo::UpdateOutShape() {
  auto cnode = cnode_;
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != 4) {
    MS_LOG(EXCEPTION) << name_ << ": The size of cnode's inputs must be 4, but got " << cnode->size();
  }

  if (!IsValueNode<ValueTuple>(cnode->input(3))) {
    MS_LOG(EXCEPTION) << name_ << ": The cnode's input[3] is not value node";
  }

  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  ValuePtr out_shape = MakeValue(out_slice_shape_);
  AnfNodePtr val = NewValueNode(out_shape);
  (void)manager->Replace(cnode->input(3), val);
  MS_LOG(INFO) << name_ << ": Update the output shape " << out_slice_shape_;
}

int64_t Conv2DBackpropInputInfo::ComputeOverlapLeftSizeByRankBias(int64_t rank_bias) {
  // 1. the first rank: 0
  // 2. the last rank:
  //    size of origin data required by current rank: a = ceil((o/n + k - o + w*s - s - x)/s)
  //    data size of the current rank: b = w/n
  //    return a - b = ceil((o/n + k - o + w*s - s - x)/s) - w/n
  // 3. the middle rank:
  //    r*w/n - ceil((r*o/n - k + x + 1)/s)
  if (rank_bias == 0) {  // the first rank
    return 0;
  }

  int64_t w_output_shape = outputs_shape_[0][3];
  int64_t w_input_shape = inputs_shape_[0][3];
  int64_t w_kernel_size = kernel_size_[1];
  int64_t w_stride = stride_[3];
  int64_t left_pad = pad_list_[2];
  if (rank_bias == w_dimension_shard_num_ - 1) {  // the last rank
    return DoubleToLong(std::ceil(LongToDouble(w_output_shape / w_dimension_shard_num_ + w_kernel_size -
                                               w_output_shape + w_input_shape * w_stride - w_stride - left_pad) /
                                  LongToDouble(w_stride))) -
           w_input_shape / w_dimension_shard_num_;
  }

  // the middle rank
  return rank_bias * w_input_shape / w_dimension_shard_num_ -
         DoubleToLong(
           std::ceil(LongToDouble(rank_bias * w_output_shape / w_dimension_shard_num_ - w_kernel_size + left_pad + 1) /
                     LongToDouble(w_stride)));
}

int64_t Conv2DBackpropInputInfo::ComputeOverlapRightSizeByRankBias(int64_t rank_bias) {
  // 1. the first rank: ceil((o/n + x)/s) - w/n
  // 2. the last rank: 0
  // 3. the middle rank: ceil((r*o/n + o/n + x)/s) - r*w/n - w/n
  int64_t w_output_shape = outputs_shape_[0][3];
  int64_t w_input_shape = inputs_shape_[0][3];
  int64_t w_stride = stride_[3];
  int64_t left_pad = pad_list_[2];

  if (rank_bias == 0) {  // the first rank
    return DoubleToLong(
             std::ceil(LongToDouble(w_output_shape / w_dimension_shard_num_ + left_pad) / LongToDouble(w_stride))) -
           w_input_shape / w_dimension_shard_num_;
  }

  if (rank_bias == w_dimension_shard_num_ - 1) {  // the last rank
    return 0;
  }

  // the middle rank
  return DoubleToLong(std::ceil(LongToDouble(rank_bias * w_output_shape / w_dimension_shard_num_ +
                                             w_output_shape / w_dimension_shard_num_ + left_pad) /
                                LongToDouble(w_stride))) -
         (rank_bias + 1) * w_input_shape / w_dimension_shard_num_;
}

void Conv2DBackpropInputInfo::InferNewPadList() {
  // 1. compute the size of origin data required by current rank:
  //    1) the first rank: ceil((o/n + x) / s)
  //    2) the last rank: ceil((o/n + k - o + ws - s - x) / s)
  //    3) the middle rank: ceil((r*o/n + o/n + x) / s) - ceil((r*o/n - k + x + 1) / s)
  //
  // 2. compute the real left pad
  //    1) the first rank: k - x - 1
  //    2) the last rank:
  //       if (o/n + k - o + ws - s - x) is divisible by s, real_left_pad = s - 1.
  //       otherwise, real_left_pad = (o/n + k - o + ws - s - x) % s - 1
  //    3) the middle rank:
  //       if (r*on - k + x + 1) is divisible by s, real_left_pad = 0.
  //       otherwise, real_left_pad = s - (r*on - k + x + 1) % s
  int64_t w_output_shape = outputs_shape_[0][3];
  int64_t w_input_shape = inputs_shape_[0][3];
  int64_t w_kernel_size = kernel_size_[1];
  int64_t w_stride = stride_[3];
  int64_t left_pad = pad_list_[2];
  int64_t current_rank_required_size = 0;
  int64_t real_left_pad = 0;

  if (rank_bias_ == 0) {  // the first rank
    current_rank_required_size = DoubleToLong(
      std::ceil(LongToDouble(w_output_shape / w_dimension_shard_num_ + left_pad) / LongToDouble(w_stride)));

    real_left_pad = w_kernel_size - left_pad - 1;
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {  // the last rank
    current_rank_required_size =
      DoubleToLong(std::ceil(LongToDouble(w_output_shape / w_dimension_shard_num_ + w_kernel_size - w_output_shape +
                                          w_input_shape * w_stride - w_stride - left_pad) /
                             LongToDouble(w_stride)));

    int64_t tmp = w_output_shape / w_dimension_shard_num_ + w_kernel_size - w_output_shape + w_input_shape * w_stride -
                  w_stride - left_pad;
    if (tmp % w_stride == 0) {
      real_left_pad = w_stride - 1;
    } else {
      real_left_pad = tmp % w_stride - 1;
    }
  } else {  // the middle rank
    current_rank_required_size =
      DoubleToLong(std::ceil(LongToDouble(rank_bias_ * w_output_shape / w_dimension_shard_num_ +
                                          w_output_shape / w_dimension_shard_num_ + left_pad) /
                             LongToDouble(w_stride))) -
      DoubleToLong(
        std::ceil(LongToDouble(rank_bias_ * w_output_shape / w_dimension_shard_num_ - w_kernel_size + left_pad + 1) /
                  LongToDouble(w_stride)));

    int64_t tmp = rank_bias_ * w_output_shape / w_dimension_shard_num_ - w_kernel_size + left_pad + 1;
    if (tmp % w_stride == 0) {
      real_left_pad = 0;
    } else {
      real_left_pad = w_stride - tmp % w_stride;
    }
  }

  // 3. compute the pad_add: (current_rank_required_size - 1) * s + k - o/n
  int64_t pad_all =
    (current_rank_required_size - 1) * w_stride + w_kernel_size - w_output_shape / w_dimension_shard_num_;

  // 4. compute new left pad: k - real_left_pad - 1
  new_pad_list_ = pad_list_;
  new_pad_list_[2] = w_kernel_size - real_left_pad - 1;

  // 5. compute new right pad: pad_all - new_left_pad
  new_pad_list_[3] = pad_all - new_pad_list_[2];

  MS_LOG(INFO) << name_ << ": the new pad list is " << new_pad_list_ << ", the required size of current rank is "
               << current_rank_required_size << ", new pad all is " << pad_all;
}

void Conv2DBackpropInputInfo::ReplaceNodeInputOrAttrs() { UpdateOutShape(); }
}  // namespace parallel
}  // namespace mindspore
