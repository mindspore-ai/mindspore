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

#ifndef NNACL_CONV_PARAMETER_H_
#define NNACL_CONV_PARAMETER_H_

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
  int tile_num_;    /* # */
  int input_batch_; /* # */
  int input_h_;     /* # */
  int input_w_;     /* # */
  int input_channel_;
  int output_batch_; /* # */
  int output_h_;     /* # */
  int output_w_;     /* # */
  int output_channel_;
  int thread_num_;  /* # */
  int input_unit_;  /* # */
  int output_unit_; /* # */
  PadType pad_mode_;
  ActType act_type_;
  int channel_multiplie_; /* # */
  int output_padding_w_;  /* # */
  int output_padding_h_;  /* # */
  int out_format_;

  bool dynamic_shape_;
} ConvParameter;

typedef struct ConvComputeParam {
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

  int in_n_;
  int in_h_;
  int in_w_;
  int in_c_;
  int out_n_;
  int out_h_;
  int out_w_;
  int out_c_;

  int in_hw_;
  int out_hw_;
  int kernel_hw_;
  int tile_num_;
} ConvComputeParam;

typedef struct SlidingWindowParam {
  int left_;
  int right_;
  int top_;
  int bottom_;
  int c_block_;
  int block_channel_;
  int ic_align_;
  int out_step_;
  int out_h_step_;
  int out_c_step_;
  int out_w_step_;
  int out_block_step_;
  int in_step_;
  int in_h_step_;
  int in_sh_step_;  // stride H
  int in_sw_step_;  // stride W
  int in_kh_step_;  // kernel H
  int in_kw_step_;  // kernel W
  int kernel_step_;
} SlidingWindowParam;

typedef struct ConvDwCalcParam {
  void *num_pixels_;
  void *out_w_start_;
  void *out_w_end_;
  int first_calc_kw_;
} ConvDwCalcParam;

#define OUPUT_UNIT 2
#define DECONV_WINOGRAD_DEFAULT_UNIT 3 /* # */
#define DECONV_WINOGRAD_DEFAULT_TILE 8 /* # */
#define DECONV_WINOGRAD_BUFFER_COUNT 8 /* # */
typedef struct DeConvWg {              /* # */
  void *b_buffer_;
  void *AT_;
  void *BT_;

  int kh_;
  int kw_;

  int k_;
  int i_;
  int o_;
} DeConvWg;

typedef struct DeConvWgABuffer { /* # */
  bool buf_init_;
  void *middle_buffer_;
  void *dest_buffer_;
} DeConvWgABuffer;

typedef struct DeConvComputeUnit { /* # */
  void *weight_;
  void *tmp_buffer_;
  int w_start_;
  int h_start_;
  int w_size_;
  int h_size_;
  bool use_winograd_;
  DeConvWg winograd_;
} DeConvComputeUnit;

typedef struct DeConvParam { /* # */
  DeConvComputeUnit *compute_units_;
  int compute_size_;
  DeConvWgABuffer a_buffer_[DECONV_WINOGRAD_BUFFER_COUNT];
  int input_plane_;
  int output_plane_;
  int kernel_plane_;
  int ic_div_;
  int oc_div_;
  int ic_up_;
  int oc_up_;
  int thread_num_;
  int in_tile_count_;
  int in_tile_h_count_;
  int in_tile_w_count_;
  int out_tile_h_;
  int out_tile_w_;
} DeConvParam;

#endif  // NNACL_CONV_PARAMETER_H_
