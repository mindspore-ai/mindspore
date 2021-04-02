/*
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

#include "wrapper/int8/conv1x1_run_int8_wrapper.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/int8/pack_int8.h"
#include "nnacl/int8/conv1x1_int8.h"
#include "nnacl/errorcode.h"

void Pre1x1Trans(Conv1x1Args *args, int8_t *src_input, int8_t *src_output) {
  args->output_ptr_ = src_output;
  if (args->pre_trans_input_) {
    Conv1x1InputPack(src_input, args->input_ptr_, args->conv_param_, sizeof(int8_t));
  } else {
    args->input_ptr_ = src_input;
  }
}

int OcOptPre(void *cdata, int task_id) {
  Conv1x1Args *args = (Conv1x1Args *)(cdata);
  int cur_stride = args->thread_stride_hw_ * C4NUM;
  int res_stride = args->matmul_param_->row_ - task_id * args->thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return NNACL_OK;
  }
  int8_t *hw_in = args->input_ptr_ + task_id * args->thread_stride_hw_ * C4NUM * args->conv_param_->input_channel_;
  int8_t *hw_packed_in = args->packed_input_ + task_id * args->thread_stride_hw_ * C4NUM * args->matmul_param_->deep_4_;
  int32_t *hw_input_sum = args->input_sum_ + task_id * args->thread_stride_hw_ * C4NUM;

  if (args->filter_peroc_) {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, args->matmul_param_->deep_, cur_hw, 1);
  } else {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, args->matmul_param_->deep_, cur_hw,
                                args->conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_);
  }
  return NNACL_OK;
}

int RunArm64OptOc(void *cdata, int task_id) {
  Conv1x1Args *args = (Conv1x1Args *)(cdata);
  int stride = args->thread_stride_oc_ * C16NUM;
  int cur_stride = task_id * stride;
  int res_stride = args->matmul_param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return NNACL_OK;
  }

  bool filter_peroc = args->filter_peroc_;
  int32_t *cur_left_shift =
    filter_peroc ? args->left_shift_ + cur_stride : args->conv_param_->conv_quant_arg_.left_shift_;
  int32_t *cur_right_shift =
    filter_peroc ? args->right_shift_ + cur_stride : args->conv_param_->conv_quant_arg_.right_shift_;
  int32_t *cur_multiplier =
    filter_peroc ? args->multiplier_ + cur_stride : args->conv_param_->conv_quant_arg_.quant_multiplier_;
  int32_t *cur_zp = filter_peroc ? args->filter_zp_ptr_ + cur_stride : args->filter_zp_ptr_;

  Conv1x1Int8Opt(args->packed_input_, args->packed_weight_ + cur_stride * args->matmul_param_->deep_4_,
                 args->output_ptr_ + cur_stride, args->input_sum_, args->bias_data_ + cur_stride,
                 args->matmul_param_->row_, cur_oc, args->matmul_param_->deep_4_, cur_left_shift, cur_right_shift,
                 cur_multiplier, args->conv_param_, args->matmul_func_, cur_zp);
  return NNACL_OK;
}

int RunArmOc(void *cdata, int task_id) {
  Conv1x1Args *args = (Conv1x1Args *)(cdata);
#ifdef ENABLE_ARM32
  int col_tile = C2NUM;
#else
  int col_tile = C4NUM;
#endif
  int stride = args->thread_stride_oc_ * col_tile;
  int cur_stride = task_id * stride;
  int res_stride = args->matmul_param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return NNACL_OK;
  }

  bool filter_peroc = args->filter_peroc_;
  int32_t *cur_left_shift =
    filter_peroc ? args->left_shift_ + cur_stride : args->conv_param_->conv_quant_arg_.left_shift_;
  int32_t *cur_right_shift =
    filter_peroc ? args->right_shift_ + cur_stride : args->conv_param_->conv_quant_arg_.right_shift_;
  int32_t *cur_multiplier =
    filter_peroc ? args->multiplier_ + cur_stride : args->conv_param_->conv_quant_arg_.quant_multiplier_;
  int32_t *cur_zp = filter_peroc ? args->filter_zp_ptr_ + cur_stride : args->filter_zp_ptr_;

  Conv1x1Int8(args->packed_input_, args->packed_weight_ + cur_stride * args->matmul_param_->deep_16_,
              args->output_ptr_ + cur_stride, args->input_sum_, args->bias_data_ + cur_stride,
              args->matmul_param_->row_, cur_oc, args->matmul_param_->deep_16_, cur_left_shift, cur_right_shift,
              cur_multiplier, args->conv_param_, cur_zp);
  return NNACL_OK;
}

int RunArm64OptHw(void *cdata, int task_id) {
  Conv1x1Args *args = (Conv1x1Args *)(cdata);
  int cur_stride = args->thread_stride_hw_ * C4NUM;
  int res_stride = args->matmul_param_->row_ - task_id * args->thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return NNACL_OK;
  }
  int8_t *hw_in = args->input_ptr_ + task_id * args->thread_stride_hw_ * C4NUM * args->conv_param_->input_channel_;
  int8_t *hw_out = args->output_ptr_ + task_id * args->thread_stride_hw_ * C4NUM * args->conv_param_->output_channel_;
  int8_t *hw_packed_in = args->packed_input_ + task_id * args->thread_stride_hw_ * C4NUM * args->matmul_param_->deep_4_;
  int32_t *hw_input_sum = args->input_sum_ + task_id * args->thread_stride_hw_ * C4NUM;

  if (args->filter_peroc_) {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, args->matmul_param_->deep_, cur_hw, 1);
  } else {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, args->matmul_param_->deep_, cur_hw,
                                args->conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_);
  }

  Conv1x1Int8Opt(hw_packed_in, args->packed_weight_, hw_out, hw_input_sum, args->bias_data_, cur_hw,
                 args->matmul_param_->col_, args->matmul_param_->deep_4_, args->left_shift_, args->right_shift_,
                 args->multiplier_, args->conv_param_, args->matmul_func_, args->filter_zp_ptr_);
  return NNACL_OK;
}

int RunArmHw(void *cdata, int task_id) {
  Conv1x1Args *args = (Conv1x1Args *)(cdata);
  int cur_stride = args->thread_stride_hw_ * C4NUM;
  int res_stride = args->matmul_param_->row_ - task_id * args->thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return NNACL_OK;
  }

  int8_t *hw_in = args->input_ptr_ + task_id * args->thread_stride_hw_ * C4NUM * args->conv_param_->input_channel_;
  int8_t *hw_out = args->output_ptr_ + task_id * args->thread_stride_hw_ * C4NUM * args->conv_param_->output_channel_;
  int8_t *hw_packed_in =
    args->packed_input_ + task_id * args->thread_stride_hw_ * C4NUM * args->matmul_param_->deep_16_;
  int32_t *hw_input_sum = args->input_sum_ + task_id * args->thread_stride_hw_ * C4NUM;

  RowMajor2Row16x4MajorInt8(hw_in, hw_packed_in, cur_hw, args->matmul_param_->deep_);

  if (args->filter_peroc_) {
    PackInputSum16x4PerLayer(hw_packed_in, hw_input_sum, 1, UP_ROUND(cur_hw, C4NUM), args->matmul_param_->deep_16_);
  } else {
    PackInputSum16x4PerLayer(hw_packed_in, hw_input_sum, args->conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_,
                             UP_ROUND(cur_hw, C4NUM), args->matmul_param_->deep_16_);
  }

  Conv1x1Int8(hw_packed_in, args->packed_weight_, hw_out, hw_input_sum, args->bias_data_, cur_hw,
              args->matmul_param_->col_, args->matmul_param_->deep_16_, args->left_shift_, args->right_shift_,
              args->multiplier_, args->conv_param_, args->filter_zp_ptr_);
  return NNACL_OK;
}

void Conv1x1PreRun(Conv1x1Args *args, int thread_num) {
  int row_pack_count = C4NUM;
  int col_pack_count;

#ifdef ENABLE_ARM32
  col_pack_count = C2NUM;
#else
  if (args->support_optimize_) {
    col_pack_count = C16NUM;
  } else {
    col_pack_count = C4NUM;
  }
#endif
  int hw_thread_count = UP_DIV(args->matmul_param_->row_, row_pack_count);
  int oc_thread_count = UP_DIV(args->matmul_param_->col_, col_pack_count);
  args->thread_count_hw = MSMIN(thread_num, hw_thread_count);
  args->thread_stride_hw_ = UP_DIV(hw_thread_count, args->thread_count_hw);
  args->thread_count_oc = MSMIN(thread_num, oc_thread_count);
  args->thread_stride_oc_ = UP_DIV(oc_thread_count, args->thread_count_oc);
  args->parallel_by_oc_ = oc_thread_count > thread_num;
  if (!args->filter_peroc_) {
    args->right_shift_ = args->conv_param_->conv_quant_arg_.right_shift_;
    args->left_shift_ = args->conv_param_->conv_quant_arg_.left_shift_;
    args->multiplier_ = args->conv_param_->conv_quant_arg_.quant_multiplier_;
  }
}
