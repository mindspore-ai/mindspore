/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/max_avg_pool_3d_info.h"

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
// maxpool3d need to calculate pad list in same mode
std::vector<int64_t> MaxPool3DInfo::CalculatePadListInSameMode() {
  int64_t stride_d = stride_[2];
  int64_t stride_h = stride_[3];
  int64_t stride_w = stride_[4];
  if (stride_d != 0 && stride_h != 0 && stride_w != 0) {
    int64_t in_d = inputs_shape_[0][2];
    int64_t in_h = inputs_shape_[0][3];
    int64_t in_w = inputs_shape_[0][4];
    int64_t kernel_d = kernel_size_use_dilation_[0];
    int64_t kernel_h = kernel_size_use_dilation_[1];
    int64_t kernel_w = kernel_size_use_dilation_[2];

    int64_t tail_d = in_d % stride_d;
    int64_t tail_h = in_h % stride_h;
    int64_t tail_w = in_w % stride_w;
    int64_t pad_d = std::max((tail_d > 0 ? kernel_d - tail_d : kernel_d - stride_d), (int64_t)0);
    int64_t pad_h = std::max((tail_h > 0 ? kernel_h - tail_h : kernel_h - stride_h), (int64_t)0);
    int64_t pad_w = std::max((tail_w > 0 ? kernel_w - tail_w : kernel_w - stride_w), (int64_t)0);
    constexpr int twice = 2;
    std::vector<int64_t> pad_list;
    pad_list.push_back(static_cast<int64_t>(std::floor(pad_d / twice)));
    pad_list.push_back(pad_d - pad_list[0]);
    pad_list.push_back(static_cast<int64_t>(std::floor(pad_h / twice)));
    pad_list.push_back(pad_h - pad_list[2]);
    pad_list.push_back(static_cast<int64_t>(std::floor(pad_w / twice)));
    pad_list.push_back(pad_w - pad_list[4]);

    MS_LOG(INFO) << name_ << ": the origin pad list is " << pad_list_ << ", the calculated pad list is " << pad_list;
    return pad_list;
  }

  MS_LOG(EXCEPTION) << "For '" << name_
                    << "', stride_d or stride_h or stride_w must be non-zero, but got stride_d: " << stride_d
                    << ", stride_h: " << stride_h << ", stride_w: " << stride_w << ".";
}

Status MaxPool3DInfo::GetAttrs() {
  // format
  format_ = GetStringAttr(FORMAT);

  // kernel_size
  kernel_size_ = GetTupleIntAttr(KERNEL_SIZE);

  // pad_mode
  pad_mode_ = GetStringAttr(PAD_MODE);

  // pad_list
  pad_list_ = GetTupleIntAttr(PAD_LIST);

  // strides
  stride_ = GetTupleIntAttr(STRIDES);

  if (format_ != NCDHW) {
    MS_LOG(ERROR) << name_ << ": The format must be 'NCDHW', but got " << format_;
    return FAILED;
  }

  if (kernel_size_.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of kernel_size must be 5, but the kernel size is " << kernel_size_;
    return FAILED;
  }

  if (kernel_size_[0] != 1 || kernel_size_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of kernel_size must be 1, but the kernel_size is "
                  << kernel_size_;
    return FAILED;
  }

  if (pad_list_.size() != 6) {
    MS_LOG(ERROR) << name_ << ": The size of pad_list must be 6, but the pad_list is " << pad_list_;
    return FAILED;
  }

  if (stride_.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of strides must be 5, but the strides is " << stride_;
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of strides must be 1, but the strides is " << stride_;
    return FAILED;
  }

  if (pad_mode_ == SAME_UPPER) {
    pad_mode_ = SAME;
  }

  if (pad_mode_ == PAD_UPPER || pad_mode_ == CALCULATED_UPPER) {
    pad_mode_ = PAD;
  }

  if (pad_mode_ == VALID_UPPER) {
    pad_mode_ = VALID;
  }

  if (pad_mode_ != SAME && pad_mode_ != PAD && pad_mode_ != VALID) {
    MS_LOG(ERROR) << name_ << ": The pad mode must be 'same'/'valid'/'pad', but got " << pad_mode_;
    return FAILED;
  }

  kernel_size_use_dilation_ = {kernel_size_[2], kernel_size_[3], kernel_size_[4]};

  if (pad_mode_ == SAME) {
    pad_list_ = CalculatePadListInSameMode();
  }

  MS_LOG(INFO) << name_ << ", kernel size is " << kernel_size_ << ", pad mode is " << pad_mode_ << ", pad list is "
               << pad_list_ << ", strides is " << stride_ << ", format is " << format_;
  return SUCCESS;
}

Status MaxPool3DInfo::CheckHWStrategyBase(int64_t h_strategy, int64_t w_strategy) const {
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

Status MaxPool3DInfo::CheckHWStrategyValidMode(int64_t h_strategy, int64_t w_strategy) {
  int64_t h_slice_shape = inputs_shape_[0][2] / h_strategy;
  int64_t w_slice_shape = inputs_shape_[0][3] / w_strategy;

  if ((kernel_size_use_dilation_[0] > stride_[2] && h_strategy > 1) ||
      (kernel_size_use_dilation_[1] > stride_[3] && w_strategy > 1)) {
    FILTER_LOG(is_auto_parallel_) << name_
                                  << ": The 'valid' mode do not support to split 2th or 3th dimension when"
                                     " kernel_size > stride";
    return FAILED;
  }

  if (kernel_size_use_dilation_[0] <= stride_[2] && h_slice_shape % stride_[2] != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_
      << ": The 'valid' mode do not support to split 2th when kernel_size <= stride but slice shape is "
         "not divisible by stride ";
    return FAILED;
  }

  if (kernel_size_use_dilation_[1] <= stride_[3] && w_slice_shape % stride_[3] != 0) {
    FILTER_LOG(is_auto_parallel_)
      << name_
      << ": The 'valid' mode do not support to split 3th when kernel_size <= stride but slice shape is "
         "not divisible by stride ";
    return FAILED;
  }

  return SUCCESS;
}

Status MaxPool3DInfo::CheckHWStrategyPadModeByDimension(int64_t strategy, int64_t dimension_id) {
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

void MaxPool3DInfo::AdjustPadList() {
  // adjust the pad list for 'pad' mode
  // because the output_len = (in_len + pad_all - k) / s, so the useless_len = (in_len + pad_all - k) % s
  // and need to adjust the bottom_pad/right_pad if useless_len != 0
  if (pad_mode_ != PAD) {
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
  MS_LOG(INFO) << name_ << ": After adjusting, the pad_list is " << pad_list_;
}

Status MaxPool3DInfo::CheckHWStrategyPadMode(int64_t h_strategy, int64_t w_strategy) {
  AdjustPadList();
  if (CheckHWStrategyPadModeByDimension(h_strategy, 2) != SUCCESS) {
    return FAILED;
  }

  if (CheckHWStrategyPadModeByDimension(w_strategy, 3) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status MaxPool3DInfo::CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) {
  if (CheckHWStrategyBase(h_strategy, w_strategy) != SUCCESS) {
    return FAILED;
  }

  if (pad_mode_ == PAD || pad_mode_ == SAME) {  // 'pad' mode or 'same' mode
    return CheckHWStrategyPadMode(h_strategy, w_strategy);
  }

  if (pad_mode_ == VALID) {  // 'valid' mode
    return CheckHWStrategyValidMode(h_strategy, w_strategy);
  }

  return SUCCESS;
}

Status MaxPool3DInfo::CheckStrategyBase(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  return SUCCESS;
}

Status MaxPool3DInfo::CheckStrategy(const StrategyPtr &strategy) {
  h_dim_need_exchange_overlap_ = false;
  w_dim_need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  if (input_strategy.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of input strategy must be 5, but the input strategy is " << input_strategy;
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MaxPool3DInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra[0];
  h_dimension_shard_num_ = stra[0][2];
  w_dimension_shard_num_ = stra[0][3];
  input_slice_shape_ = GetSliceShape(inputs_shape_[0], stra[0]);
  return SUCCESS;
}

std::vector<int64_t> MaxPool3DInfo::GetAdjacentRankIdsAndBiases(int64_t rank_id, int64_t dimension) {
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

void MaxPool3DInfo::InferAdjacentRankInfo() {
  // the GetAdjacentRankIdsAndBiases will handle the repeated calculation

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

int64_t MaxPool3DInfo::ComputeOverlapTopSizeByRankBias(int64_t rank_bias) {
  int64_t top_pad = pad_list_[0];
  int64_t h_dimension_input_shape = inputs_shape_[0][2];
  int64_t h_dimension_output_shape = outputs_shape_[0][2];
  int64_t h_stride = stride_[2];

  return top_pad + (h_dimension_input_shape - h_dimension_output_shape * h_stride) * rank_bias / h_dimension_shard_num_;
}

int64_t MaxPool3DInfo::ComputeOverlapBottomSizeByRankBias(int64_t rank_bias) {
  int64_t top_pad = pad_list_[0];
  int64_t h_dimension_input_shape = inputs_shape_[0][2];
  int64_t h_dimension_output_shape = outputs_shape_[0][2];
  int64_t h_kernel_size = kernel_size_use_dilation_[0];
  int64_t h_stride = stride_[2];

  return (rank_bias + 1) * (h_dimension_output_shape * h_stride - h_dimension_input_shape) / h_dimension_shard_num_ +
         h_kernel_size - h_stride - top_pad;
}

int64_t MaxPool3DInfo::ComputeOverlapLeftSizeByRankBias(int64_t rank_bias) {
  int64_t left_pad = pad_list_[2];
  int64_t w_dimension_input_shape = inputs_shape_[0][3];
  int64_t w_dimension_output_shape = outputs_shape_[0][3];
  int64_t w_stride = stride_[3];

  return left_pad +
         (w_dimension_input_shape - w_dimension_output_shape * w_stride) * rank_bias / w_dimension_shard_num_;
}

int64_t MaxPool3DInfo::ComputeOverlapRightSizeByRankBias(int64_t rank_bias) {
  int64_t left_pad = pad_list_[2];
  int64_t w_dimension_input_shape = inputs_shape_[0][3];
  int64_t w_dimension_output_shape = outputs_shape_[0][3];
  int64_t w_kernel_size = kernel_size_use_dilation_[1];
  int64_t w_stride = stride_[3];

  return (rank_bias + 1) * (w_dimension_output_shape * w_stride - w_dimension_input_shape) / w_dimension_shard_num_ +
         w_kernel_size - w_stride - left_pad;
}

void MaxPool3DInfo::InferOverlapSizeForHDim() {
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

void MaxPool3DInfo::InferOverlapSizeForWDim() {
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

void MaxPool3DInfo::CheckHDimensionOverlapSizeNonNegative() {
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

void MaxPool3DInfo::CheckWDimensionOverlapSizeNonNegative() {
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

void MaxPool3DInfo::CheckOverlapSizeNonNegative() {
  CheckHDimensionOverlapSizeNonNegative();
  CheckWDimensionOverlapSizeNonNegative();
}

void MaxPool3DInfo::InferOverlapSize() {
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

Status MaxPool3DInfo::InferTensorMap() {
  TensorMap input_tensor_map = {4, 3, 2, 1, 0};
  TensorMap output_tensor_map = {4, 3, 2, 1, 0};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status MaxPool3DInfo::InferForwardCommunication() {
  forward_op_.clear();
  return SUCCESS;
}

void MaxPool3DInfo::InferNewPadList() {
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

void MaxPool3DInfo::InferSendRankIds() {
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

void MaxPool3DInfo::InferRecvRankIds() {
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

void MaxPool3DInfo::InferCommunicationAttrs() {
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

void MaxPool3DInfo::InferNewOperatorAttrs() {
  InferNewPadList();

  InferCommunicationAttrs();
}

OperatorAttrs MaxPool3DInfo::CreateNeighborExchangeV2Attrs() {
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

OperatorAttrs MaxPool3DInfo::CreateNewOpAttrs() {
  std::vector<int64_t> strides_v = {stride_[2], stride_[3], stride_[4]};
  bool ceil_mode_v = GetValue<int64_t>(attrs_[CEIL_MODE]) == 1;

  Attr kernel_size = {KERNEL_SIZE, MakeValue(kernel_size_use_dilation_)};
  Attr strides = {STRIDES, MakeValue(strides_v)};
  Attr pad_mode = {PAD_MODE, MakeValue(PAD)};
  Attr pad_list = {PAD_LIST, MakeValue(new_pad_list_)};
  Attr ceil_mode = {CEIL_MODE, MakeValue(ceil_mode_v)};
  Attr data_format = {DATA_FORMAT, MakeValue(format_)};

  OperatorAttrs attrs;
  attrs = {kernel_size, strides, pad_mode, pad_list, ceil_mode, data_format};
  return attrs;
}

OperatorAttrs AvgPool3DInfo::CreateNewOpAttrs() {
  std::vector<int64_t> strides_v = {stride_[2], stride_[3], stride_[4]};
  bool ceil_mode_v = GetValue<bool>(attrs_[CEIL_MODE]);

  Attr kernel_size = {KERNEL_SIZE, MakeValue(kernel_size_use_dilation_)};
  Attr strides = {STRIDES, MakeValue(strides_v)};
  Attr pad_mode = {PAD_MODE, MakeValue(PAD)};
  Attr pad = {PAD, MakeValue(new_pad_list_)};
  Attr ceil_mode = {CEIL_MODE, MakeValue(ceil_mode_v)};
  Attr count_include_pad = {COUNT_INCLUDE_PAD, attrs_[COUNT_INCLUDE_PAD]};
  Attr divisor_override = {DIVISOR_OVERRIDE, attrs_[DIVISOR_OVERRIDE]};
  Attr data_format = {DATA_FORMAT, MakeValue(format_)};

  OperatorAttrs attrs;
  attrs = {kernel_size, strides, pad_mode, pad, ceil_mode, count_include_pad, divisor_override, data_format};
  return attrs;
}

std::string MaxPool3DInfo::ReplaceNodeName() const { return MAXPOOL_3D; }

std::string AvgPool3DInfo::ReplaceNodeName() const { return AVGPOOL_3D; }

AnfNodePtr MaxPool3DInfo::GenerateNewOpNode(const AnfNodePtr &new_input, const CNodePtr &cnode) {
  auto op_attrs = CreateNewOpAttrs();
  auto node_name = ReplaceNodeName();

  return gen_g_.PushBack({gen_g_.NewOpInst(node_name, op_attrs), new_input});
}

void MaxPool3DInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  // Because the NeighborExchangeV2 only support the 4-dim input, and it only exchange the last 2-dim of input, but the
  // input of 3d-op is 5-dim, and need to exchange 3/4th-dim of input, so here use some operators to build the graph:
  // slice input (ncdhw) -> transpose(in, (4, 0, 1, 2, 3)) -> reshape(in, (w*n, c, d, h)) -> neighborexchangev2(in)
  // -> reshape(in, (w, n, c, d', h')) -> transpose(in, (1, 2, 3, 4, 0)) -> conv3d
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  if (gen_g_.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateGraph Init failed";
  }

  // transpose-1
  std::vector<int64_t> t1 = {4, 0, 1, 2, 3};
  auto transpose_1 = gen_g_.PushBack({gen_g_.NewOpInst(TRANSPOSE), gen_g_.virtual_input_node(), CreateTuple(t1)});

  // reshape-1
  auto s = input_slice_shape_;
  if (s.size() != 5) {
    MS_LOG(EXCEPTION) << name_ << ": The size of input slice shape must be 5, but got " << s.size();
  }
  Shape s1 = {s[4] * s[0], s[1], s[2], s[3]};
  auto reshape_1 = gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), transpose_1, CreateTuple(s1)});

  // neighborexchangev2
  auto neighbor_exchange_v2_attrs = CreateNeighborExchangeV2Attrs();
  auto neighbor_exchange_v2 =
    gen_g_.PushBack({gen_g_.NewOpInst(NEIGHBOREXCHANGEV2, neighbor_exchange_v2_attrs), reshape_1});

  // reshape-2
  Shape s2 = {s[4], s[0], s[1], s[2] + recv_lens_[0] + recv_lens_[1], s[3] + recv_lens_[2] + recv_lens_[3]};
  auto reshape_2 = gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), neighbor_exchange_v2, CreateTuple(s2)});

  // transopse-2
  std::vector<int64_t> t2 = {1, 2, 3, 4, 0};
  auto transpose_2 = gen_g_.PushBack({gen_g_.NewOpInst(TRANSPOSE), reshape_2, CreateTuple(t2)});

  // 3d-op
  auto op_3d = GenerateNewOpNode(transpose_2, cnode);

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(transpose_1, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, op_3d));
}

ReplaceGraphPtr MaxPool3DInfo::replace_graph(const CNodePtr &cnode) {
  if (!w_dim_need_exchange_overlap_ && !h_dim_need_exchange_overlap_) {
    return nullptr;
  }

  InferAdjacentRankInfo();

  InferOverlapSize();

  InferNewOperatorAttrs();

  int64_t all_send_lens = std::accumulate(send_lens_.begin(), send_lens_.end(), 0);
  int64_t all_recv_lens = std::accumulate(recv_lens_.begin(), recv_lens_.end(), 0);
  if (all_send_lens + all_recv_lens == 0) {
    std::string pad_mode = PAD;
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    prim->set_attr(PAD_MODE, MakeValue(pad_mode));

    if (ReplaceNodeName() == MAXPOOL_3D) {
      prim->set_attr(PAD_LIST, MakeValue(new_pad_list_));
    } else {
      prim->set_attr(PAD, MakeValue(new_pad_list_));
    }
    MS_LOG(INFO) << name_ << ": the send lens and recv lens is 0, no need exchange data";
    return nullptr;
  }

  ComputeReplaceGraph(cnode);
  return replace_graph_;
}

Status MaxPool3DInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> MaxPool3DInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  auto parallel_context = ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  auto search_mode = parallel_context->strategy_search_mode();
  // generate data parallel strategy when the search mode is not sharding propagation
  if (parallel_mode == parallel::kAutoParallel && search_mode != parallel::kShardingPropagation) {
    Shape input_strategy(inputs_shape_[0].size(), 1);
    input_strategy[0] = stage_device_size_;
    Strategies strategy = {input_strategy};
    StrategyPtr data_parallel_sp = std::make_shared<Strategy>(stage_id, strategy);
    sp_vector.push_back(data_parallel_sp);
    return sp_vector;
  }

  Shapes splittable_input = {{1, 1, 1, 1, 1}};
  Shapes tmp_inputs_shape = inputs_shape_;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  return sp_vector;
}

REGISTER(MaxPool3DInfo);
REGISTER(AvgPool3DInfo);
}  // namespace parallel
}  // namespace mindspore
