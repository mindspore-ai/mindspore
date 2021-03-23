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

#ifndef MINDSPORE_LITE_NNACL_CONV_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_CONV_PARAMETER_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"

typedef struct ConvParameter {
  OpParameter op_parameter_;
  ConvQuantArg conv_quant_arg_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int dilation_h_;
  int dilation_w_;
  int pad_u_;
  int pad_d_;
  int pad_l_;
  int pad_r_;
  int group_;
  int tile_num_;
  int input_batch_;
  int input_h_;
  int input_w_;
  int input_channel_;
  int output_batch_;
  int output_h_;
  int output_w_;
  int output_channel_;
  int thread_num_;
  int input_unit_;
  int output_unit_;
  PadMode pad_mode_;
  ActType act_type_;
  int channel_multiplie_;
  int output_padding_w_;
  int output_padding_h_;
} ConvParameter;

typedef struct SlidingWindowParam {
  int left_;
  int right_;
  int top_;
  int bottom_;
  int c_block_;
  int block_channel_;
  int ic4_channel_;
  int out_step_;
  int out_h_step_;
  int in_step_;
  int in_h_step_;
  int in_sh_step_;  // stride H
  int in_sw_step_;  // stride W
  int in_kh_step_;  // kernel H
  int in_kw_step_;  // kernel W
  int kernel_step_;
} SlidingWindowParam;

#define OUPUT_UNIT 2
#define DECONV_WINOGRAD_DEFAULT_UNIT 3
#define DECONV_WINOGRAD_DEFAULT_TILE 8
#define DECONV_WINOGRAD_BUFFER_COUNT 8
typedef struct DeConvWg {
  void *b_buffer_;
  void *AT_;
  void *BT_;

  int kh_;
  int kw_;

  int k_;
  int i_;
  int o_;
} DeConvWg;

typedef struct DeConvWgABuffer {
  bool buf_init_;
  void *middle_buffer_;
  void *dest_buffer_;
} DeConvWgABuffer;

typedef struct DeConvComputeUnit {
  void *weight_;
  void *tmp_buffer_;
  int w_start_;
  int h_start_;
  int w_size_;
  int h_size_;
  bool use_winograd_;
  DeConvWg winograd_;
} DeConvComputeUnit;

typedef struct DeConvParam {
  DeConvComputeUnit *compute_units_;
  int compute_size_;
  DeConvWgABuffer a_buffer_[DECONV_WINOGRAD_BUFFER_COUNT];
  int input_plane_;
  int output_plane_;
  int kernel_plane_;
  int ic_div4_;
  int oc_div4_;
  int ic_up4_;
  int oc_up4_;
  int thread_num_;
  int in_tile_count_;
  int in_tile_h_count_;
  int in_tile_w_count_;
  int out_tile_h_;
  int out_tile_w_;
} DeConvParam;

#endif  // MINDSPORE_LITE_NNACL_CONV_PARAMETER_H_
