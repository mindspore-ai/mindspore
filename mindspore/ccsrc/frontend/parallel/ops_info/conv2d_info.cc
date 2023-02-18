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
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/common/utils/parallel_context.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status Conv2DInfo::CheckAttrsBase() {
  if (format_ != NCHW) {
    MS_LOG(ERROR) << name_ << ": The format must be 'NCHW', but got " << format_;
    return FAILED;
  }

  if (kernel_size_.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of kernel_size'tuple must be 2, but got " << kernel_size_.size();
    return FAILED;
  }

  if (pad_list_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of pad_list must be 4, but got " << pad_list_.size();
    return FAILED;
  }

  if (stride_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of stride must be 4, but got " << stride_.size();
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of stride must be 1, but got (" << stride_[0] << ", "
                  << stride_[1] << ")";
    return FAILED;
  }

  if (dilation_.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of dilation must be 4, but got " << dilation_.size();
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The out channel is " << out_channel_ << ", kernel size is " << kernel_size_
               << ", mode is " << mode_ << ", pad mode is " << pad_mode_ << ", pad list is " << pad_list_
               << ", stride is " << stride_ << ", dilation is " << dilation_ << ", group is " << group_
               << ", format is " << format_ << ", the kernel size use dilation is " << kernel_size_use_dilation_;
  return SUCCESS;
}

std::vector<int64_t> Conv2DInfo::GetStrideAttr() { return GetTupleIntAttr(STRIDE); }

std::vector<int64_t> Conv2DInfo::GetDilationAttr() { return GetTupleIntAttr(DILATION); }

Status Conv2DInfo::GetAttrsBase() {
  // format
  format_ = GetStringAttr(FORMAT);

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
    kernel_size_ = Shape(inputs_shape_[1].size() - 2, kernel_size);
  } else if (kernel_size_iter->second->isa<ValueTuple>() || kernel_size_iter->second->isa<ValueList>()) {
    kernel_size_ = GetValue<std::vector<int64_t>>(kernel_size_iter->second);
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

  // stride
  stride_ = GetStrideAttr();

  // dilation
  dilation_ = GetDilationAttr();

  for (size_t i = 0; i < kernel_size_.size(); ++i) {
    kernel_size_use_dilation_.push_back(dilation_[i + 2] * (kernel_size_[i] - 1) + 1);
  }

  // group
  group_ = GetIntAttr(GROUP);

  return SUCCESS;
}

void Conv2DInfo::AdjustPadList() {
  // adjust the pad list for 'pad' mode
  // because the output_len = (in_len + pad_all - k) / s, so the useless_len = (in_len + pad_all - k) % s
  // and need to adjust the bottom_pad/right_pad if useless_len != 0
  if (pad_mode_ != 0 || pad_list_adjusted_) {
    return;
  }

  int64_t useless_len_2th_dim =
    (inputs_shape_[0][2] + pad_list_[0] + pad_list_[1] - kernel_size_use_dilation_[0]) % stride_[2];
  int64_t useless_len_3th_dim =
    (inputs_shape_[0][3] + pad_list_[2] + pad_list_[3] - kernel_size_use_dilation_[1]) % stride_[3];
  if (useless_len_2th_dim == 0 && useless_len_3th_dim == 0) {
    return;
  }

  if (useless_len_2th_dim > pad_list_[1]) {
    MS_LOG(EXCEPTION) << name_ << ": The useless len for 2th dim (" << useless_len_2th_dim
                      << ") can not larger than pad_list[1] (" << pad_list_[1] << ")";
  }
  if (useless_len_3th_dim > pad_list_[3]) {
    MS_LOG(EXCEPTION) << name_ << ": The useless len for 3th dim (" << useless_len_3th_dim
                      << ") can not larger than pad_list[3] (" << pad_list_[3] << ")";
  }
  pad_list_[1] -= useless_len_2th_dim;
  pad_list_[3] -= useless_len_3th_dim;
  pad_list_adjusted_ = true;
  MS_LOG(INFO) << name_ << ": After adjusting, the pad_list is " << pad_list_;
}

Status Conv2DInfo::GetAttrs() {
  if (GetAttrsBase() != SUCCESS || CheckAttrsBase() != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategyBase(int64_t h_strategy, int64_t w_strategy) const {
  if (outputs_shape_[0][2] % h_strategy != 0) {
    FILTER_LOG(is_auto_parallel_) << name_
                                  << ": Do not support to split 2th dimension when out_shape of 2th dimension is not"
                                     " divisible by strategy of 2th dimension";
    return FAILED;
  }

  if (outputs_shape_[0][3] % w_strategy != 0) {
    FILTER_LOG(is_auto_parallel_) << name_
                                  << ": Do not support to split 3th dimension when out_shape of 3th dimension is not"
                                     " divisible by strategy of 3th dimension";
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategyValidMode(int64_t h_strategy, int64_t w_strategy) {
  int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
  int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;

  if ((kernel_size_use_dilation_[0] > stride_[2] && h_strategy > 1) ||
      (kernel_size_use_dilation_[1] > stride_[3] && w_strategy > 1)) {
    FILTER_LOG(is_auto_parallel_) << name_
                                  << ": The 'valid' mode do not support to split 2th or 3th dimension when"
                                     " kernel_size_use_dilation_ > stride";
    return FAILED;
  }

  if (kernel_size_use_dilation_[0] <= stride_[2] && h_slice_shape % stride_[2] != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_
      << ": The 'valid' mode do not support to split 2th when kernel_size_use_dilation_ <= stride but slice shape is "
         "not divisible by stride ";
    return FAILED;
  }

  if (kernel_size_use_dilation_[1] <= stride_[3] && w_slice_shape % stride_[3] != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_
      << ": The 'valid' mode do not support to split 3th when kernel_size_use_dilation_ <= stride but slice shape is "
         "not divisible by stride ";
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategyPadModeByDimension(int64_t strategy, int64_t dimension_id) {
  if (strategy == 1) {
    return SUCCESS;
  }

  int64_t h_or_w_input_shape = 0, h_or_w_output_shape = 0, h_or_w_kernel_size = 0, h_or_w_stride = 0, pad_all = 0;
  if (dimension_id == 2) {
    h_or_w_input_shape = inputs_shape_[0][2];
    h_or_w_output_shape = outputs_shape_[0][2];
    h_or_w_kernel_size = kernel_size_use_dilation_[0];
    h_or_w_stride = stride_[2];
    pad_all = pad_list_[0] + pad_list_[1];
  } else {
    h_or_w_input_shape = inputs_shape_[0][3];
    h_or_w_output_shape = outputs_shape_[0][3];
    h_or_w_kernel_size = kernel_size_use_dilation_[1];
    h_or_w_stride = stride_[3];
    pad_all = pad_list_[2] + pad_list_[3];
  }

  // kernel size <= stride, no need to exchange
  if (h_or_w_kernel_size <= h_or_w_stride) {
    if (pad_all != 0) {
      FILTER_LOG(is_auto_parallel_) << name_ << ": The 'pad' or 'same' mode do not support to split " << dimension_id
                                    << "th dimension when kernel_size <= stride and pad != 0";
      return FAILED;
    }
    if ((h_or_w_input_shape / strategy) % h_or_w_stride != 0) {
      FILTER_LOG(is_auto_parallel_) << name_ << ": The 'pad' or 'same' mode do not support to split " << dimension_id
                                    << "th dimension when kernel_size <= stride and input's slice % stride != 0";
      return FAILED;
    }
    return SUCCESS;
  }

  // kernel_size > stride, need to exchange
  if ((h_or_w_input_shape + pad_all - h_or_w_kernel_size) % h_or_w_stride != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_ << ": The 'pad' or 'same' mode do not support to split " << dimension_id
      << "th dimension when kernel_size > stride and input_shape + pad_all - k is not divisible by stride";
    return FAILED;
  }

  if ((h_or_w_output_shape * h_or_w_stride - h_or_w_input_shape) % strategy != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_ << ": The 'pad' or 'same' mode do not support to split " << dimension_id
      << "th dimension when kernel_size > stride and output_shape * s - input_shape is not divisible by stride";
    return FAILED;
  }

  // if the h/w dimension is split, and the pad mode is not "valid", need to exchange overlap
  if (dimension_id == 2) {
    h_dim_need_exchange_overlap_ = true;
  } else if (dimension_id == 3) {
    w_dim_need_exchange_overlap_ = true;
  }
  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategyPadMode(int64_t h_strategy, int64_t w_strategy) {
  AdjustPadList();
  if (CheckHWStrategyPadModeByDimension(h_strategy, 2) != SUCCESS) {
    return FAILED;
  }

  if (CheckHWStrategyPadModeByDimension(w_strategy, 3) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status Conv2DInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (CheckHWStrategyBase(h_strategy, w_strategy) != SUCCESS) {
    return FAILED;
  }

  if (pad_mode_ == 0 || pad_mode_ == 1) {  // 'pad' mode or 'same' mode
    return CheckHWStrategyPadMode(h_strategy, w_strategy);
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

  Dimensions weight_strategy = stra[1];
  if (weight_strategy[0] > 1) {
    out_channel_shard_ = true;
    new_out_channel_ = out_channel_ / weight_strategy[0];
  } else {
    out_channel_shard_ = false;
    new_out_channel_ = out_channel_;
  }

  if (group_ != 1 && (weight_strategy[0] != 1 || weight_strategy[1] != 1)) {
    MS_LOG(ERROR) << name_ << ": The group is " << group_
                  << ", the cout and cin can not be split, but the shard num of cout is " << weight_strategy[0]
                  << ", the shard num of cin is " << weight_strategy[1];
    return FAILED;
  }
  return SUCCESS;
}

Status Conv2DInfo::CheckStrategy(const StrategyPtr &strategy) {
  h_dim_need_exchange_overlap_ = false;
  w_dim_need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy.size() != 4 || weight_strategy.size() != 4) {
    MS_LOG(ERROR) << name_
                  << ": The size of input strategy or weight strategy must be 4, but the size of input strategy is "
                  << input_strategy.size() << ", the size of weight strategy is " << weight_strategy.size();
    return FAILED;
  }

  if (input_strategy[1] != weight_strategy[1]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-in for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-in for weight strategy is " << weight_strategy[1];
    return FAILED;
  }

  if (weight_strategy[2] != 1 || weight_strategy[3] != 1) {
    MS_LOG(ERROR) << name_ << ": The kernel size can not be split, but the strategy for kernel size is ("
                  << weight_strategy[2] << ", " << weight_strategy[3] << ")";
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Conv2DInfo::InferDevMatrixShape() {
  // conv2d: the strategy is ((n, i, a, b), (o, i, 1, 1))
  // conv3d: the strategy is ((n, i, a, b, 1), (o, i, 1, 1, 1))
  // the dev matrix is (n, i, a, b, o)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  dev_matrix_shape_ = {stra[0][0], stra[0][1], stra[0][2], stra[0][3]};
  dev_matrix_shape_.push_back(stra[1][0]);
  h_dimension_shard_num_ = stra[0][2];
  w_dimension_shard_num_ = stra[0][3];
  input_slice_shape_ = GetSliceShape(inputs_shape_[0], stra[0]);
  return SUCCESS;
}

std::vector<int64_t> Conv2DInfo::GetAdjacentRankIdsAndBiases(int64_t rank_id, int64_t dimension) {
  std::vector<int64_t> ret;
  if (rank_id < 0) {
    ret = {-1, -1, -1, -1, -1};
    return ret;
  }

  MS_LOG(INFO) << name_ << ": The rank id is " << rank_id << ", the dimension is " << dimension << "th dimension";

  uint64_t index_in_dev_matrix = 0;
  int64_t dimension_shard_num = 1;
  if (dimension == 2) {
    index_in_dev_matrix = 2;
    dimension_shard_num = h_dimension_shard_num_;
  } else {
    index_in_dev_matrix = 3;
    dimension_shard_num = w_dimension_shard_num_;
  }

  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    index_in_dev_matrix += 1;
  }

  DeviceMatrix dev_matrix(rank_id, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(index_in_dev_matrix, &group_devices) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Get device along dim failed";
  }

  if (group_devices.size() <= 1) {
    MS_LOG(INFO) << name_ << ": The devices' size of " << dimension << "th dimension is " << group_devices.size()
                 << ", no need to infer rank bias";
    ret = {-1, -1, -1, -1, -1};
    return ret;
  }

  if (group_devices.size() != LongToSize(dimension_shard_num)) {
    MS_LOG(EXCEPTION) << name_ << ": The devices' size of " << dimension << "th dimension is " << group_devices.size()
                      << ", but the shard num of this dimension is " << dimension_shard_num;
  }

  std::vector<int64_t>::iterator it = std::find(group_devices.begin(), group_devices.end(), rank_id);
  if (it == group_devices.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the current rank in device list of " << dimension
                      << "th dimension, the current rank is " << rank_id << ", the device list is " << group_devices;
  }

  int64_t left_or_top_rank_id = -1;
  int64_t right_or_bottom_rank_id = -1;
  int64_t left_or_top_rank_bias = -1;
  int64_t right_or_bottom_rank_bias = -1;
  int64_t current_rank_bias = -1;
  current_rank_bias = std::distance(group_devices.begin(), it);
  if (it == group_devices.begin()) {
    left_or_top_rank_bias = -1;
    right_or_bottom_rank_bias = current_rank_bias + 1;

    left_or_top_rank_id = -1;
    right_or_bottom_rank_id = *(it + 1);
  } else if (it == group_devices.end() - 1) {
    left_or_top_rank_bias = current_rank_bias - 1;
    right_or_bottom_rank_bias = -1;

    left_or_top_rank_id = *(it - 1);
    right_or_bottom_rank_id = -1;
  } else {
    left_or_top_rank_bias = current_rank_bias - 1;
    right_or_bottom_rank_bias = current_rank_bias + 1;

    left_or_top_rank_id = *(it - 1);
    right_or_bottom_rank_id = *(it + 1);
  }

  ret = {left_or_top_rank_id, right_or_bottom_rank_id, left_or_top_rank_bias, right_or_bottom_rank_bias,
         current_rank_bias};
  return ret;
}

void Conv2DInfo::InferAdjacentRankInfo() {
  // the Conv2D operator:
  // the origin dev_matrix is [n, i, h, w, o]
  // if repeated calculation and repeated num in the left of dev matrix, the dev_matrix is [repeated_num, n, i, h, w, o]
  // if repeated calculation and repeated num in the right of dev matrix, the dev_matrix is [n, i, h, w, o,
  // repeated_num]
  //
  // the Conv2DBackpropInput's origin dev_matrix is [n, o, h, w, i], w dimension's relative position is the same as
  // Conv2D, the w_rank_bias_ is the position of the current rank in the w dimension of the dev_matrix(have not split h
  // dimension)

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  std::vector<int64_t> h_dim_rank_info = GetAdjacentRankIdsAndBiases(rank, 2);
  top_rank_id_ = h_dim_rank_info[0];
  bottom_rank_id_ = h_dim_rank_info[1];
  top_rank_bias_ = h_dim_rank_info[2];
  bottom_rank_bias_ = h_dim_rank_info[3];
  h_rank_bias_ = h_dim_rank_info[4];

  std::vector<int64_t> w_dim_rank_info = GetAdjacentRankIdsAndBiases(rank, 3);
  left_rank_id_ = w_dim_rank_info[0];
  right_rank_id_ = w_dim_rank_info[1];
  left_rank_bias_ = w_dim_rank_info[2];
  right_rank_bias_ = w_dim_rank_info[3];
  w_rank_bias_ = w_dim_rank_info[4];

  std::vector<int64_t> top_w_dim_rank_info = GetAdjacentRankIdsAndBiases(top_rank_id_, 3);
  top_left_rank_id_ = top_w_dim_rank_info[0];
  top_right_rank_id_ = top_w_dim_rank_info[1];

  std::vector<int64_t> bottom_w_dim_rank_info = GetAdjacentRankIdsAndBiases(bottom_rank_id_, 3);
  bottom_left_rank_id_ = bottom_w_dim_rank_info[0];
  bottom_right_rank_id_ = bottom_w_dim_rank_info[1];

  all_to_all_group_ = g_device_manager->world_group();  // use world group temporarily
  MS_LOG(INFO) << name_ << ": The current rank is " << rank << ", the top rank id is " << top_rank_id_
               << ", the top right rank id is " << top_right_rank_id_ << ", the right rank id is " << right_rank_id_
               << ", the bottom right rank id is " << bottom_right_rank_id_ << ", the bottom rank id is "
               << bottom_rank_id_ << ", the bottom left rank id is " << bottom_left_rank_id_ << ", the left rank id is "
               << left_rank_id_ << ", the top left rank id is " << top_left_rank_id_ << ", the top rank bias is "
               << top_rank_bias_ << ", the bottom rank bias is " << bottom_rank_bias_ << ", the left rank bias is "
               << left_rank_bias_ << ", the right rank bias is " << right_rank_bias_ << ", the h dim rank bias is "
               << h_rank_bias_ << ", the w dim rank bias is " << w_rank_bias_;
}

int64_t Conv2DInfo::ComputeOverlapTopSizeByRankBias(int64_t rank_bias) {
  int64_t top_pad = pad_list_[0];
  int64_t h_dimension_input_shape = inputs_shape_[0][2];
  int64_t h_dimension_output_shape = outputs_shape_[0][2];
  int64_t h_stride = stride_[2];

  return top_pad + (h_dimension_input_shape - h_dimension_output_shape * h_stride) * rank_bias / h_dimension_shard_num_;
}

int64_t Conv2DInfo::ComputeOverlapBottomSizeByRankBias(int64_t rank_bias) {
  int64_t top_pad = pad_list_[0];
  int64_t h_dimension_input_shape = inputs_shape_[0][2];
  int64_t h_dimension_output_shape = outputs_shape_[0][2];
  int64_t h_kernel_size = kernel_size_use_dilation_[0];
  int64_t h_stride = stride_[2];

  return (rank_bias + 1) * (h_dimension_output_shape * h_stride - h_dimension_input_shape) / h_dimension_shard_num_ +
         h_kernel_size - h_stride - top_pad;
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
  int64_t w_kernel_size = kernel_size_use_dilation_[1];
  int64_t w_stride = stride_[3];

  return (rank_bias + 1) * (w_dimension_output_shape * w_stride - w_dimension_input_shape) / w_dimension_shard_num_ +
         w_kernel_size - w_stride - left_pad;
}

void Conv2DInfo::InferOverlapSizeForHDim() {
  if (!h_dim_need_exchange_overlap_) {
    overlap_top_size_ = 0;
    overlap_bottom_size_ = 0;
    bottom_rank_overlap_top_size_ = 0;
    top_rank_overlap_bottom_size_ = 0;
    return;
  }

  if (h_rank_bias_ == 0) {
    // it has not top rank
    overlap_top_size_ = 0;
    overlap_bottom_size_ = ComputeOverlapBottomSizeByRankBias(h_rank_bias_);
    top_rank_overlap_bottom_size_ = 0;
    bottom_rank_overlap_top_size_ = ComputeOverlapTopSizeByRankBias(bottom_rank_bias_);
  } else if (h_rank_bias_ == h_dimension_shard_num_ - 1) {
    // it has not bottom rank
    overlap_top_size_ = ComputeOverlapTopSizeByRankBias(h_rank_bias_);
    overlap_bottom_size_ = 0;
    top_rank_overlap_bottom_size_ = ComputeOverlapBottomSizeByRankBias(top_rank_bias_);
    bottom_rank_overlap_top_size_ = 0;
  } else {
    // it has left rank and right rank
    overlap_top_size_ = ComputeOverlapTopSizeByRankBias(h_rank_bias_);
    overlap_bottom_size_ = ComputeOverlapBottomSizeByRankBias(h_rank_bias_);
    top_rank_overlap_bottom_size_ = ComputeOverlapBottomSizeByRankBias(top_rank_bias_);
    bottom_rank_overlap_top_size_ = ComputeOverlapTopSizeByRankBias(bottom_rank_bias_);
  }
}

void Conv2DInfo::InferOverlapSizeForWDim() {
  if (!w_dim_need_exchange_overlap_) {
    overlap_left_size_ = 0;
    overlap_right_size_ = 0;
    left_rank_overlap_right_size_ = 0;
    right_rank_overlap_left_size_ = 0;
    return;
  }

  if (w_rank_bias_ == 0) {
    // it has not left rank
    overlap_left_size_ = 0;
    overlap_right_size_ = ComputeOverlapRightSizeByRankBias(w_rank_bias_);
    left_rank_overlap_right_size_ = 0;
    right_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(right_rank_bias_);
  } else if (w_rank_bias_ == w_dimension_shard_num_ - 1) {
    // it has not right rank
    overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(w_rank_bias_);
    overlap_right_size_ = 0;
    left_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = 0;
  } else {
    // it has left rank and right rank
    overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(w_rank_bias_);
    overlap_right_size_ = ComputeOverlapRightSizeByRankBias(w_rank_bias_);
    left_rank_overlap_right_size_ = ComputeOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = ComputeOverlapLeftSizeByRankBias(right_rank_bias_);
  }
}

void Conv2DInfo::CheckHDimensionOverlapSizeNonNegative() {
  if (h_dimension_shard_num_ == 1) {
    MS_LOG(INFO) << name_ << ": The 2th dimension is not shard";
    return;
  }

  int64_t h_first_rank_bottom_size = ComputeOverlapBottomSizeByRankBias(0);
  if (h_first_rank_bottom_size < 0) {
    MS_LOG(EXCEPTION) << name_ << ": The bottom overlap size of 2th dimension rank bias 0 must be positive, but it is "
                      << h_first_rank_bottom_size;
  }

  for (int64_t h_rank_bias = 1; h_rank_bias < h_dimension_shard_num_ - 1; ++h_rank_bias) {
    auto top_size = ComputeOverlapTopSizeByRankBias(h_rank_bias);
    auto bottom_size = ComputeOverlapBottomSizeByRankBias(h_rank_bias);
    if (top_size < 0 || bottom_size < 0) {
      MS_LOG(EXCEPTION) << name_ << ": The overlap size of 2th dimension rank bias " << h_rank_bias
                        << " must be positive, but top overlap size is " << top_size << ", bottom overlap size is "
                        << bottom_size;
    }
  }

  int64_t h_last_rank_top_size = ComputeOverlapTopSizeByRankBias(h_dimension_shard_num_ - 1);
  if (h_last_rank_top_size < 0) {
    MS_LOG(EXCEPTION) << name_ << ": The top overlap size of 2th dimension last rank bias must be positive, but it is "
                      << h_last_rank_top_size;
  }
}

void Conv2DInfo::CheckWDimensionOverlapSizeNonNegative() {
  if (w_dimension_shard_num_ == 1) {
    MS_LOG(INFO) << name_ << ": The 3th dimension is not shard";
    return;
  }
  int64_t w_first_rank_right_size = ComputeOverlapRightSizeByRankBias(0);
  if (w_first_rank_right_size < 0) {
    MS_LOG(EXCEPTION) << name_ << ": The right overlap size of 3th dimension rank bias 0 must be positive, but it is "
                      << w_first_rank_right_size;
  }

  for (int64_t w_rank_bias = 1; w_rank_bias < w_dimension_shard_num_ - 1; ++w_rank_bias) {
    auto left_size = ComputeOverlapLeftSizeByRankBias(w_rank_bias);
    auto right_size = ComputeOverlapRightSizeByRankBias(w_rank_bias);
    if (left_size < 0 || right_size < 0) {
      MS_LOG(EXCEPTION) << name_ << ": The overlap size of 3th dimension rank bias " << w_rank_bias
                        << " must be positive, but left overlap size is " << left_size << ", right overlap size is "
                        << right_size;
    }
  }

  int64_t w_last_rank_left_size = ComputeOverlapLeftSizeByRankBias(w_dimension_shard_num_ - 1);
  if (w_last_rank_left_size < 0) {
    MS_LOG(EXCEPTION) << name_ << ": The left overlap size of 3th dimension last rank bias must be positive, but it is "
                      << w_last_rank_left_size;
  }
}

void Conv2DInfo::CheckOverlapSizeNonNegative() {
  CheckHDimensionOverlapSizeNonNegative();
  CheckWDimensionOverlapSizeNonNegative();
}

void Conv2DInfo::InferOverlapSize() {
  InferOverlapSizeForHDim();
  InferOverlapSizeForWDim();

  MS_LOG(INFO) << name_ << ": the left overlap size of current rank is " << overlap_left_size_
               << ", the right overlap size of current rank is " << overlap_right_size_
               << ", the right overlap size of left rank is " << left_rank_overlap_right_size_
               << ", the left overlap size of right rank is " << right_rank_overlap_left_size_
               << ", the top overlap size of current rank is " << overlap_top_size_
               << ", the bottom overlap size of current rank is " << overlap_bottom_size_
               << ", the bottom overlap size of top rank is " << top_rank_overlap_bottom_size_
               << ", the top overlap size of bottom rank is " << bottom_rank_overlap_top_size_;

  CheckOverlapSizeNonNegative();
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

// Conv2d/Conv3d: dev_matrix is (n, i, h, w, o), if in channel is split, it need to insert all reduce
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
    ReportError(name_ + ": Create group failed");
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
  if (h_dim_need_exchange_overlap_) {
    if (h_rank_bias_ == 0) {                                  // the first rank
      new_pad_list_[1] = 0;                                   // no need the bottom pad
    } else if (h_rank_bias_ == h_dimension_shard_num_ - 1) {  // the last rank
      new_pad_list_[0] = 0;                                   // no need the top pad
    } else {                                                  // the middle rank
      new_pad_list_[0] = 0;                                   // no need the top pad
      new_pad_list_[1] = 0;                                   // no need the bottom pad
    }
  }

  if (w_dim_need_exchange_overlap_) {
    if (w_rank_bias_ == 0) {                                  // the first rank
      new_pad_list_[3] = 0;                                   // no need the right pad
    } else if (w_rank_bias_ == w_dimension_shard_num_ - 1) {  // the last rank
      new_pad_list_[2] = 0;                                   // no need the left pad
    } else {                                                  // the middle rank
      new_pad_list_[2] = 0;                                   // no need the left pad
      new_pad_list_[3] = 0;                                   // no need the right pad
    }
  }

  MS_LOG(INFO) << name_ << ": the new pad list is " << new_pad_list_;
}

void Conv2DInfo::InferSendRankIds() {
  int64_t send_top_rank = top_rank_overlap_bottom_size_ > 0 ? top_rank_id_ : -1;
  int64_t send_bottom_rank = bottom_rank_overlap_top_size_ > 0 ? bottom_rank_id_ : -1;
  int64_t send_left_rank = left_rank_overlap_right_size_ > 0 ? left_rank_id_ : -1;
  int64_t send_right_rank = right_rank_overlap_left_size_ > 0 ? right_rank_id_ : -1;
  int64_t send_top_left_rank = (send_top_rank != -1 && send_left_rank != -1) ? top_left_rank_id_ : -1;
  int64_t send_top_right_rank = (send_top_rank != -1 && send_right_rank != -1) ? top_right_rank_id_ : -1;
  int64_t send_bottom_left_rank = (send_bottom_rank != -1 && send_left_rank != -1) ? bottom_left_rank_id_ : -1;
  int64_t send_bottom_right_rank = (send_bottom_rank != -1 && send_right_rank != -1) ? bottom_right_rank_id_ : -1;

  // the order of send or recv rank ids in the array is organized in the following format:
  // [top, top_right, right, bottom_right, bottom, bottom_left, left, top_left]
  send_rank_ids_ = {send_top_rank,    send_top_right_rank,   send_right_rank, send_bottom_right_rank,
                    send_bottom_rank, send_bottom_left_rank, send_left_rank,  send_top_left_rank};
}

void Conv2DInfo::InferRecvRankIds() {
  int64_t recv_top_rank = overlap_top_size_ > 0 ? top_rank_id_ : -1;
  int64_t recv_bottom_rank = overlap_bottom_size_ > 0 ? bottom_rank_id_ : -1;
  int64_t recv_left_rank = overlap_left_size_ > 0 ? left_rank_id_ : -1;
  int64_t recv_right_rank = overlap_right_size_ > 0 ? right_rank_id_ : -1;
  int64_t recv_top_left_rank = (recv_top_rank != -1 && recv_left_rank != -1) ? top_left_rank_id_ : -1;
  int64_t recv_top_right_rank = (recv_top_rank != -1 && recv_right_rank != -1) ? top_right_rank_id_ : -1;
  int64_t recv_bottom_left_rank = (recv_bottom_rank != -1 && recv_left_rank != -1) ? bottom_left_rank_id_ : -1;
  int64_t recv_bottom_right_rank = (recv_bottom_rank != -1 && recv_right_rank != -1) ? bottom_right_rank_id_ : -1;

  // the order of send or recv rank ids in the array is organized in the following format:
  // [top, top_right, right, bottom_right, bottom, bottom_left, left, top_left]
  recv_rank_ids_ = {recv_top_rank,    recv_top_right_rank,   recv_right_rank, recv_bottom_right_rank,
                    recv_bottom_rank, recv_bottom_left_rank, recv_left_rank,  recv_top_left_rank};
}

void Conv2DInfo::InferCommunicationAttrs() {
  // send ranks
  InferSendRankIds();

  // recv ranks
  InferRecvRankIds();

  // send lens
  int64_t send_top_len = top_rank_overlap_bottom_size_;
  int64_t send_bottom_len = bottom_rank_overlap_top_size_;
  int64_t send_left_len = left_rank_overlap_right_size_;
  int64_t send_right_len = right_rank_overlap_left_size_;

  // recv lens
  int64_t recv_top_len = overlap_top_size_;
  int64_t recv_bottom_len = overlap_bottom_size_;
  int64_t recv_left_len = overlap_left_size_;
  int64_t recv_right_len = overlap_right_size_;

  // the order of send or recv lens in the array is organized in the following format:
  // [top, bottom, left, right]
  send_lens_ = {send_top_len, send_bottom_len, send_left_len, send_right_len};
  recv_lens_ = {recv_top_len, recv_bottom_len, recv_left_len, recv_right_len};

  MS_LOG(INFO) << name_ << ": The send rank ids is " << send_rank_ids_ << ", the send lens is " << send_lens_
               << ", the recv rank ids is " << recv_rank_ids_ << ", the recv lens is " << recv_lens_;

  for (auto &send_len : send_lens_) {
    if (send_len < 0) {
      MS_LOG(EXCEPTION) << name_ << ": Send len less than 0 is not supported, but it is " << send_len;
    }
  }

  for (auto &recv_len : recv_lens_) {
    if (recv_len < 0) {
      MS_LOG(EXCEPTION) << name_ << ": Recv len less than 0 is not supported, but it is " << recv_len;
    }
  }

  int64_t h_slice_shape = input_slice_shape_[2];
  if (send_top_len > h_slice_shape || send_bottom_len > h_slice_shape || recv_top_len > h_slice_shape ||
      recv_bottom_len > h_slice_shape) {
    MS_LOG(EXCEPTION) << name_ << ": The send or recv len larger than slice shape of 2th dimension " << h_slice_shape;
  }

  int64_t w_slice_shape = input_slice_shape_[3];
  if (send_left_len > w_slice_shape || send_right_len > w_slice_shape || recv_left_len > w_slice_shape ||
      recv_right_len > w_slice_shape) {
    MS_LOG(EXCEPTION) << name_ << ": The send or recv len larger than slice shape of 3th dimension " << w_slice_shape;
  }
}

void Conv2DInfo::InferNewOperatorAttrs() {
  InferNewPadList();

  InferCommunicationAttrs();
}

OperatorAttrs Conv2DInfo::CreateNeighborExchangeV2Attrs() {
  // the type of send_rank_ids, recv_rank_ids, send_lens, recv_lens is list, is not tuple, can not use MakeValue
  // the MakeValue(vector) return a tuple
  Attr send_rank_ids = {SEND_RANK_IDS, MakeListValue(send_rank_ids_)};
  Attr send_lens = {SEND_LENS, MakeListValue(send_lens_)};
  Attr recv_rank_ids = {RECV_RANK_IDS, MakeListValue(recv_rank_ids_)};
  Attr recv_lens = {RECV_LENS, MakeListValue(recv_lens_)};
  Attr data_format = {DATA_FORMAT, MakeValue(NCHW)};
  Attr group = {GROUP, MakeValue(all_to_all_group_)};

  OperatorAttrs attrs = {send_rank_ids, send_lens, recv_rank_ids, recv_lens, data_format, group};
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
  if (name_.find(CONV2D_INFO) != std::string::npos || name_.find(CONV3D_INFO) != std::string::npos) {
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

  auto neighbor_exchange_v2_attrs = CreateNeighborExchangeV2Attrs();
  auto neighbor_exchange_v2_node =
    gen_g_.PushBack({gen_g_.NewOpInst(NEIGHBOREXCHANGEV2, neighbor_exchange_v2_attrs), gen_g_.virtual_input_node()});

  auto conv2d = GenerateConv2DNode(neighbor_exchange_v2_node, cnode);

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(neighbor_exchange_v2_node, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, conv2d));
}

ReplaceGraphPtr Conv2DInfo::replace_graph(const CNodePtr &cnode) {
  if (!w_dim_need_exchange_overlap_ && !h_dim_need_exchange_overlap_) {
    if (!out_channel_shard_) {
      return nullptr;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    prim->set_attr(OUT_CHANNEL, MakeValue(new_out_channel_));
    return nullptr;
  }

  InferAdjacentRankInfo();

  InferOverlapSize();

  InferNewOperatorAttrs();

  int64_t all_send_lens = std::accumulate(send_lens_.begin(), send_lens_.end(), 0);
  int64_t all_recv_lens = std::accumulate(recv_lens_.begin(), recv_lens_.end(), 0);
  if (all_send_lens + all_recv_lens == 0) {
    int64_t pad_mode = 0;  // 0 is "pad" mode
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    prim->set_attr(OUT_CHANNEL, MakeValue(new_out_channel_));
    prim->set_attr(PAD_MODE, MakeValue(pad_mode));  // need to use int64_t to define pad_mode
    prim->set_attr(PAD, MakeValue(new_pad_list_));
    MS_LOG(INFO) << name_ << ": the send lens and recv lens is 0, no need exchange data";
    return nullptr;
  }

  ComputeReplaceGraph(cnode);
  return replace_graph_;
}

void Conv2DInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status Conv2DInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> Conv2DInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  auto parallel_context = ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  auto search_mode = parallel_context->strategy_search_mode();
  // generate data parallel strategy when the search mode is not sharding propagation
  if (parallel_mode == parallel::kAutoParallel && search_mode != parallel::kShardingPropagation) {
    Shape input_strategy(inputs_shape_[0].size(), 1);
    input_strategy[0] = stage_device_size_;
    Strategies strategy = {input_strategy, Shape(inputs_shape_[1].size(), 1)};
    StrategyPtr data_parallel_sp = std::make_shared<Strategy>(stage_id, strategy);
    sp_vector.push_back(data_parallel_sp);
    return sp_vector;
  }

  // to generate the strategy for (N, C1, H, W, C2), the k1/k2 can not be split
  Shapes splittable_input = {{1, 1, 1, 1, 0}};  // keep C2 unsplittable and simply avoid redistribution
  Shape tmp_shape = inputs_shape_[0];
  if (name_.find(CONV2D_INFO) != std::string::npos) {  // conv2d: ((N, C-in, H, W), (C-out, C-in, k1, k2))
    tmp_shape.push_back(inputs_shape_[1][0]);          // the tmp shape is (N, C-in, H, W, C-out)
  } else {                                             // conv2d-transpose: ((N, C-out, H, W), (C-out, C-in, k1, k2))
    tmp_shape.push_back(inputs_shape_[1][1]);          // the tmp shape is (N, C-out, H, W, C-in)
  }
  Shapes tmp_inputs_shape = {tmp_shape};
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // set the inputs' strategies
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies replace_strategy;
    Dimensions tmp_strategy = sp->GetInputDim()[0];
    if (tmp_strategy.size() != 5) {
      MS_LOG(EXCEPTION) << name_ << ": The size of first tmp strategy must be 5, but got " << tmp_strategy.size();
    }

    Dimensions input0_strategy;
    Dimensions input1_strategy;

    if (name_.find(CONV2D_INFO) != std::string::npos) {  // conv2d
      input0_strategy = {tmp_strategy[0], tmp_strategy[1], tmp_strategy[2], tmp_strategy[3]};
      input1_strategy = {tmp_strategy[4], tmp_strategy[1], 1, 1};  // (C-out, C-in, k1, k2), the k1/k2 can not be split
    } else if (name_.find(CONV3D_INFO) != std::string::npos) {     // conv3d
      input0_strategy = {tmp_strategy[0], tmp_strategy[1], tmp_strategy[2], tmp_strategy[3], 1};
      input1_strategy = {tmp_strategy[4], tmp_strategy[1], 1, 1, 1};
    } else if (name_.find(CONV2D_TRANSPOSE) != std::string::npos ||
               name_.find(CONV2D_BACK_PROP_INPUT) != std::string::npos) {  // conv2d-transpose
      input0_strategy = {tmp_strategy[0], tmp_strategy[1], tmp_strategy[2], tmp_strategy[3]};
      input1_strategy = {tmp_strategy[1], tmp_strategy[4], 1, 1};
    }
    replace_strategy.push_back(input0_strategy);
    replace_strategy.push_back(input1_strategy);
    sp->ResetInputs(replace_strategy);
  }
  return sp_vector;
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
  if (GetAttrsBase() != SUCCESS || CheckAttrsBase() != SUCCESS) {
    return FAILED;
  }

  return GetOutShape();
}

Status Conv2DBackpropInputInfo::CheckStrategy(const StrategyPtr &strategy) {
  w_dim_need_exchange_overlap_ = false;
  h_dim_need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy.size() != 4 || weight_strategy.size() != 4) {
    MS_LOG(ERROR) << name_
                  << ": The size of input strategy or weight strategy must be 4, but the size of input strategy is "
                  << input_strategy.size() << ", the size of weight strategy is " << weight_strategy.size();
    return FAILED;
  }

  if (input_strategy[1] != weight_strategy[0]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-out for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-out for weight strategy is " << weight_strategy[0];
    return FAILED;
  }

  if (weight_strategy[2] != 1 || weight_strategy[3] != 1) {
    MS_LOG(ERROR) << name_ << ": The kernel size can not be split, but the strategy for kernel size is ("
                  << weight_strategy[2] << ", " << weight_strategy[3] << ")";
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  // if the h/w dimension is split, need to exchange overlap
  if (input_strategy[2] > 1) {
    h_dim_need_exchange_overlap_ = true;
  }

  if (input_strategy[3] > 1) {
    w_dim_need_exchange_overlap_ = true;
  }
  return SUCCESS;
}

Status Conv2DBackpropInputInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (CheckHWStrategyBase(h_strategy, w_strategy) != SUCCESS) {
    return FAILED;
  }

  if (pad_mode_ != 0 && pad_mode_ != 1) {  // only support pad mode and same mode
    FILTER_LOG(is_auto_parallel_) << name_ << ": Do not support the pad mode " << pad_mode_
                                  << " when split H or W dimension";
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

  h_dimension_shard_num_ = stra[0][2];
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
      ReportError(name_ + ": Create group failed, the input index is " + std::to_string(i));
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

int64_t Conv2DBackpropInputInfo::ComputeOverlapTopSizeByRankBias(int64_t rank_bias) {
  // 1. the first rank: 0
  // 2. the last rank:
  //    size of origin data required by current rank: a = ceil((o/n + k - o + h*s - s - x)/s)
  //    data size of the current rank: b = h/n
  //    return a - b = ceil((o/n + k - o + h*s - s - x)/s) - h/n
  // 3. the middle rank: (the x is top pad)
  //    r*h/n - ceil((r*o/n - k + x + 1)/s)
  if (rank_bias == 0) {  // the first rank
    return 0;
  }

  int64_t h_output_shape = outputs_shape_[0][2];
  int64_t h_input_shape = inputs_shape_[0][2];
  int64_t h_kernel_size = kernel_size_use_dilation_[0];
  int64_t h_stride = stride_[2];
  int64_t top_pad = pad_list_[0];
  if (rank_bias == h_dimension_shard_num_ - 1) {  // the last rank
    return DoubleToLong(std::ceil(LongToDouble(h_output_shape / h_dimension_shard_num_ + h_kernel_size -
                                               h_output_shape + h_input_shape * h_stride - h_stride - top_pad) /
                                  LongToDouble(h_stride))) -
           h_input_shape / h_dimension_shard_num_;
  }

  // the middle rank
  return rank_bias * h_input_shape / h_dimension_shard_num_ -
         DoubleToLong(
           std::ceil(LongToDouble(rank_bias * h_output_shape / h_dimension_shard_num_ - h_kernel_size + top_pad + 1) /
                     LongToDouble(h_stride)));
}

int64_t Conv2DBackpropInputInfo::ComputeOverlapBottomSizeByRankBias(int64_t rank_bias) {
  // 1. the first rank: ceil((o/n + x)/s) - h/n
  // 2. the last rank: 0
  // 3. the middle rank: ceil((r*o/n + o/n + x)/s) - r*h/n - h/n
  int64_t h_output_shape = outputs_shape_[0][2];
  int64_t h_input_shape = inputs_shape_[0][2];
  int64_t h_stride = stride_[2];
  int64_t top_pad = pad_list_[0];

  if (rank_bias == 0) {  // the first rank
    return DoubleToLong(
             std::ceil(LongToDouble(h_output_shape / h_dimension_shard_num_ + top_pad) / LongToDouble(h_stride))) -
           h_input_shape / h_dimension_shard_num_;
  }

  if (rank_bias == h_dimension_shard_num_ - 1) {  // the last rank
    return 0;
  }

  // the middle rank
  return DoubleToLong(std::ceil(LongToDouble(rank_bias * h_output_shape / h_dimension_shard_num_ +
                                             h_output_shape / h_dimension_shard_num_ + top_pad) /
                                LongToDouble(h_stride))) -
         (rank_bias + 1) * h_input_shape / h_dimension_shard_num_;
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
  int64_t w_kernel_size = kernel_size_use_dilation_[1];
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

void Conv2DBackpropInputInfo::InferNewPadListByDimension(const std::string &dimension) {
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
  //       if (r*o/n - k + x + 1) < 0, real_left_pad = -(r*o/n - k + x + 1);
  //       otherwise, if (r*o/n - k + x + 1) is divisible by s, real_left_pad = 0.
  //       otherwise, real_left_pad = s - (r*o/n - k + x + 1) % s
  int64_t current_rank_required_size = 0;
  int64_t real_top_or_left_pad = 0;
  int64_t h_or_w_output_shape = -1;
  int64_t h_or_w_input_shape = -1;
  int64_t h_or_w_kernel_size = -1;
  int64_t h_or_w_stride = -1;
  int64_t top_or_left_pad = -1;
  int64_t h_or_w_rank_bias = -1;
  int64_t h_or_w_dim_shard_num = -1;

  if (dimension == H_DIMENSION) {
    h_or_w_output_shape = outputs_shape_[0][2];
    h_or_w_input_shape = inputs_shape_[0][2];
    h_or_w_kernel_size = kernel_size_use_dilation_[0];
    h_or_w_stride = stride_[2];
    top_or_left_pad = pad_list_[0];
    h_or_w_rank_bias = h_rank_bias_;
    h_or_w_dim_shard_num = h_dimension_shard_num_;
  } else {
    h_or_w_output_shape = outputs_shape_[0][3];
    h_or_w_input_shape = inputs_shape_[0][3];
    h_or_w_kernel_size = kernel_size_use_dilation_[1];
    h_or_w_stride = stride_[3];
    top_or_left_pad = pad_list_[2];
    h_or_w_rank_bias = w_rank_bias_;
    h_or_w_dim_shard_num = w_dimension_shard_num_;
  }

  if (h_or_w_rank_bias == 0) {  // the first rank
    current_rank_required_size = DoubleToLong(std::ceil(
      LongToDouble(h_or_w_output_shape / h_or_w_dim_shard_num + top_or_left_pad) / LongToDouble(h_or_w_stride)));

    real_top_or_left_pad = h_or_w_kernel_size - top_or_left_pad - 1;
  } else if (h_or_w_rank_bias == h_or_w_dim_shard_num - 1) {  // the last rank
    current_rank_required_size = DoubleToLong(
      std::ceil(LongToDouble(h_or_w_output_shape / h_or_w_dim_shard_num + h_or_w_kernel_size - h_or_w_output_shape +
                             h_or_w_input_shape * h_or_w_stride - h_or_w_stride - top_or_left_pad) /
                LongToDouble(h_or_w_stride)));

    int64_t tmp = h_or_w_output_shape / h_or_w_dim_shard_num + h_or_w_kernel_size - h_or_w_output_shape +
                  h_or_w_input_shape * h_or_w_stride - h_or_w_stride - top_or_left_pad;
    if (tmp % h_or_w_stride == 0) {
      real_top_or_left_pad = h_or_w_stride - 1;
    } else {
      real_top_or_left_pad = tmp % h_or_w_stride - 1;
    }
  } else {  // the middle rank
    current_rank_required_size =
      DoubleToLong(std::ceil(LongToDouble(h_or_w_rank_bias * h_or_w_output_shape / h_or_w_dim_shard_num +
                                          h_or_w_output_shape / h_or_w_dim_shard_num + top_or_left_pad) /
                             LongToDouble(h_or_w_stride))) -
      DoubleToLong(std::ceil(LongToDouble(h_or_w_rank_bias * h_or_w_output_shape / h_or_w_dim_shard_num -
                                          h_or_w_kernel_size + top_or_left_pad + 1) /
                             LongToDouble(h_or_w_stride)));

    int64_t tmp =
      h_or_w_rank_bias * h_or_w_output_shape / h_or_w_dim_shard_num - h_or_w_kernel_size + top_or_left_pad + 1;
    if (tmp < 0) {
      real_top_or_left_pad = -tmp;
    } else if (tmp % h_or_w_stride == 0) {
      real_top_or_left_pad = 0;
    } else {
      real_top_or_left_pad = h_or_w_stride - tmp % h_or_w_stride;
    }
  }

  // 3. compute the pad_add: (current_rank_required_size - 1) * s + k - o/n
  int64_t pad_all =
    (current_rank_required_size - 1) * h_or_w_stride + h_or_w_kernel_size - h_or_w_output_shape / h_or_w_dim_shard_num;

  // 4. compute new left pad: k - real_left_pad - 1
  // 5. compute new right pad: pad_all - new_left_pad
  if (dimension == H_DIMENSION) {
    new_pad_list_[0] = h_or_w_kernel_size - real_top_or_left_pad - 1;
    new_pad_list_[1] = pad_all - new_pad_list_[0];
  } else {
    new_pad_list_[2] = h_or_w_kernel_size - real_top_or_left_pad - 1;
    new_pad_list_[3] = pad_all - new_pad_list_[2];
  }

  MS_LOG(INFO) << name_ << ": The dimension is " << dimension << ", the required size of current rank is "
               << current_rank_required_size << ", new pad all is " << pad_all;
}

void Conv2DBackpropInputInfo::InferNewPadList() {
  // init new pad list
  new_pad_list_ = pad_list_;

  // infer h dimension's new pad
  if (h_dim_need_exchange_overlap_) {
    InferNewPadListByDimension(H_DIMENSION);
  }

  // infer w dimension's new pad
  if (w_dim_need_exchange_overlap_) {
    InferNewPadListByDimension(W_DIMENSION);
  }

  MS_LOG(INFO) << name_ << ": The new pad list is " << new_pad_list_;
}

void Conv2DBackpropInputInfo::ReplaceNodeInputOrAttrs() { UpdateOutShape(); }

REGISTER(Conv2DInfo);
REGISTER(Conv2DBackpropInputInfo);
REGISTER(Conv2DTransposeInfo);
}  // namespace parallel
}  // namespace mindspore
