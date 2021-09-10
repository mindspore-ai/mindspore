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
#include "nnacl/fp16/conv_fp16.h"
#include <string.h>
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/winograd_transform_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"

// fp16 convolution common (im2col+gemm)
void ConvFp16(const float16_t *input_data, float16_t *packed_input, const float16_t *packed_weight,
              const float16_t *bias_data, float16_t *col_major_input, float16_t *output_data, int task_id,
              const ConvParameter *conv_param) {
#ifdef ENABLE_ARM64
  const int tile_n = 16;
#else
  const int tile_n = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(conv_param->thread_num_);
  NNACL_CHECK_ZERO_RETURN(tile_n);
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int block_per_thread = UP_DIV(UP_DIV(output_hw, tile_n), conv_param->thread_num_);
  int start_block = block_per_thread * task_id;
  int start_hw = start_block * tile_n;
  int end_hw = MSMIN(output_hw, (start_block + block_per_thread) * tile_n);
  if (start_hw >= end_hw) {
    return;
  }
  int out_stride = conv_param->output_channel_ * tile_n;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * tile_n;
  col_major_input += task_id * deep * tile_n;
  size_t input_size = deep * tile_n * sizeof(float16_t);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * conv_param->output_channel_ * output_hw + start_hw * conv_param->output_channel_;
    for (int i = start_hw; i < end_hw; i += tile_n, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, tile_n);
      memset(packed_input, 0, input_size);
      Im2ColPackUnitFp16(input_data + in_offset, conv_param, packed_input, real_cal_row, i);
#ifdef ENABLE_ARM64
      RowMajor2Col16MajorFp16Opt(packed_input, col_major_input, tile_n, deep);
#else
      RowMajor2Col12MajorFp16Opt(packed_input, col_major_input, tile_n, deep);
#endif
      MatMulFp16(col_major_input, packed_weight, output_data + out_offset, bias_data, conv_param->act_type_, deep,
                 real_cal_row, conv_param->output_channel_, conv_param->output_channel_, OutType_Nhwc);
    }
  }
}

void ConvOutNc8hw8Fp16(const float16_t *input_data, float16_t *packed_input, const float16_t *packed_weight,
                       const float16_t *bias_data, float16_t *col_major_input, float16_t *output_data, int task_id,
                       const ConvParameter *conv_param) {
#ifdef ENABLE_ARM64
  const int tile_n = 16;
#else
  const int tile_n = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(conv_param->op_parameter_.thread_num_);
  NNACL_CHECK_ZERO_RETURN(tile_n);
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int input_block = UP_DIV(output_hw, tile_n);
  int block_per_thread = UP_DIV(input_block, conv_param->thread_num_);
  int start_block = block_per_thread * task_id;
  int end_block = MSMIN(start_block + block_per_thread, input_block);
  if (start_block >= end_block) {
    return;
  }
  int weight_block = UP_DIV(conv_param->output_channel_, C8NUM);
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += deep * tile_n * task_id;
  col_major_input += deep * tile_n * task_id;
  size_t input_size = deep * tile_n * sizeof(float16_t);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    for (int i = start_block; i < end_block; i++) {
      int real_in_row = (i != input_block - 1) ? tile_n : output_hw - i * tile_n;
      memset(packed_input, 0, input_size);
      Im2ColPackUnitFp16(input_data + in_offset, conv_param, packed_input, real_in_row, i * tile_n);
#ifdef ENABLE_ARM64
      RowMajor2Col16MajorFp16Opt(packed_input, col_major_input, tile_n, deep);
#else
      RowMajor2Col12MajorFp16Opt(packed_input, col_major_input, tile_n, deep);
#endif
      const float16_t *cur_weight = packed_weight;
      const float16_t *cur_bias = bias_data;
      for (int j = 0; j < weight_block; j++, cur_weight += C8NUM * deep, cur_bias += C8NUM) {
        int real_weight_row = (j != weight_block - 1) ? C8NUM : conv_param->output_channel_ - j * C8NUM;
        int out_offset = j * output_hw * C8NUM + i * tile_n * real_weight_row;
        MatMulFp16(col_major_input, cur_weight, output_data + out_offset, cur_bias, conv_param->act_type_, deep,
                   real_in_row, real_weight_row, real_weight_row, OutType_Nhwc);
      }
    }
  }
}

void Conv1x1OutNc8hw8MultiThreadByInputFp16(const float16_t *input, float16_t *pack_input, const float16_t *weight,
                                            const float16_t *bias, float16_t *output, int task_id,
                                            const MatMulParameter *param) {
#ifdef ENABLE_ARM64
  const int tile_n = 16;
#else
  const int tile_n = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(tile_n);
  NNACL_CHECK_ZERO_RETURN(param->op_parameter_.thread_num_);
  int input_block = UP_DIV(param->row_, tile_n);
  int weight_block = UP_DIV(param->col_, C8NUM);

  int block_per_thread = UP_DIV(input_block, param->op_parameter_.thread_num_);
  int input_start_block = block_per_thread * task_id;
  int input_end_block = MSMIN(input_start_block + block_per_thread, input_block);
  if (input_start_block >= input_end_block) {
    return;
  }
  input += input_start_block * tile_n * param->deep_;
  pack_input += input_start_block * tile_n * param->deep_;

  int cur_row_cnt = MSMIN(block_per_thread * tile_n, param->row_ - input_start_block * tile_n);
#ifdef ENABLE_ARM64
  RowMajor2Col16MajorFp16Opt(input, pack_input, cur_row_cnt, param->deep_);
#else
  RowMajor2Col12MajorFp16Opt(input, pack_input, cur_row_cnt, param->deep_);
#endif
  for (int i = input_start_block; i < input_end_block; i++) {
    int real_in_row = (i != input_block - 1) ? tile_n : param->row_ - i * tile_n;
    const float16_t *cur_weight = weight;
    const float16_t *cur_bias = bias;
    for (int j = 0; j < weight_block; j++, cur_weight += C8NUM * param->deep_, cur_bias += C8NUM) {
      int real_weight_row = (j != weight_block - 1) ? C8NUM : param->col_ - j * C8NUM;
      int out_offset = j * param->row_ * C8NUM + i * tile_n * real_weight_row;
      MatMulFp16(pack_input, cur_weight, output + out_offset, cur_bias, param->act_type_, param->deep_, real_in_row,
                 real_weight_row, real_weight_row, OutType_Nhwc);
    }
    pack_input += real_in_row * param->deep_;
  }
}

void Conv1x1OutNc8hw8MultiThreadByWeightFp16(const float16_t *input, float16_t *pack_input, const float16_t *weight,
                                             const float16_t *bias, float16_t *output, int task_id,
                                             const MatMulParameter *param) {
#ifdef ENABLE_ARM64
  const int tile_n = 16;
#else
  const int tile_n = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(tile_n);
  NNACL_CHECK_ZERO_RETURN(param->op_parameter_.thread_num_);
  int input_block = UP_DIV(param->row_, tile_n);
  int weight_block = UP_DIV(param->col_, C8NUM);

  int block_per_thread = UP_DIV(weight_block, param->op_parameter_.thread_num_);
  int weight_start_block = block_per_thread * task_id;
  int weight_end_block = MSMIN(weight_start_block + block_per_thread, weight_block);
  if (weight_start_block >= weight_end_block) {
    return;
  }
  for (int i = 0; i < input_block; i++) {
    int real_in_row = (i != input_block - 1) ? tile_n : param->row_ - i * tile_n;
    const float16_t *cur_weight = weight + weight_start_block * C8NUM * param->deep_;
    const float16_t *cur_bias = bias + weight_start_block * C8NUM;
    for (int j = weight_start_block; j < weight_end_block; j++, cur_weight += C8NUM * param->deep_, cur_bias += C8NUM) {
      int real_weight_row = (j != weight_block - 1) ? C8NUM : param->col_ - j * C8NUM;
      int out_offset = j * param->row_ * C8NUM + i * tile_n * real_weight_row;
      MatMulFp16(pack_input, cur_weight, output + out_offset, cur_bias, param->act_type_, param->deep_, real_in_row,
                 real_weight_row, real_weight_row, OutType_Nhwc);
    }
    pack_input += real_in_row * param->deep_;
  }
}

// fp16 convolution winograd
void ConvWinogardFp16(const float16_t *input_data, const float16_t *trans_weight, const float16_t *bias_data,
                      float16_t *output_data, TmpBufferAddressFp16 *buffer_list, int task_id,
                      const ConvParameter *conv_param, InputTransFp16Func in_func, OutputTransFp16Func out_func) {
#ifdef ENABLE_ARM64
  const int tile_num = 16;
#else
  const int tile_num = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(conv_param->output_unit_);
  NNACL_CHECK_ZERO_RETURN(conv_param->thread_num_);
  int in_channel = conv_param->input_channel_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  int per_thread_num = UP_DIV(output_count, conv_param->thread_num_);
  int real_tile = per_thread_num < tile_num ? per_thread_num : tile_num;
  NNACL_CHECK_ZERO_RETURN(real_tile);
  int output_tile_count = UP_DIV(output_count, real_tile);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);
  int input_unit_square = conv_param->input_unit_ * conv_param->input_unit_;

  float16_t *trans_input = buffer_list[0];
  float16_t *gemm_out = buffer_list[1];
  float16_t *tmp_data = buffer_list[2];
  float16_t *col_buffer = buffer_list[3];
  int trans_input_offset = tile_num * input_unit_square * in_channel;
  int gemm_out_offset = tile_num * input_unit_square * oc8 * C8NUM;
  int tmp_data_offset = input_unit_square * C8NUM;
  int col_buffer_offset = tile_num * in_channel;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)
  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * conv_param->output_channel_ * conv_param->output_h_ * conv_param->output_w_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int out_tile_index = thread_id * real_tile;
      int cal_num = output_count - thread_id * real_tile;
      cal_num = cal_num > real_tile ? real_tile : cal_num;
      if (cal_num <= 0) {
        return;
      }
      WinogradInputTransformFp16(input_data + in_batch_offset, trans_input + task_id * trans_input_offset,
                                 tmp_data + task_id * tmp_data_offset, cal_num, out_tile_index, out_w_block, conv_param,
                                 in_func);
      // step 3 : gemm
      float16_t *src_ptr = trans_input + task_id * trans_input_offset;
      float16_t *dst_ptr = gemm_out + task_id * gemm_out_offset;
      float16_t *tmp_col_ptr = col_buffer + task_id * col_buffer_offset;
      for (int i = 0; i < input_unit_square; ++i) {
#ifdef ENABLE_ARM64
        RowMajor2Col16MajorFp16Opt(src_ptr + i * tile_num * in_channel, tmp_col_ptr, cal_num, in_channel);
#else
        RowMajor2Col12MajorFp16Opt(src_ptr + i * tile_num * in_channel, tmp_col_ptr, cal_num, in_channel);
#endif
        MatMulFp16(tmp_col_ptr, trans_weight + i * in_channel * oc8 * C8NUM, dst_ptr + i * C8NUM, NULL, 0, in_channel,
                   cal_num, oc8 * C8NUM, input_unit_square, OutType_TileC8);
      }

      // step 4 : output transform
      if (conv_param->out_format_ != NNACL_NC4HW4) {  // nc4hw4
        WinogradOutputNHWCTransformFp16(gemm_out + task_id * gemm_out_offset, output_data + out_batch_offset, bias_data,
                                        cal_num, out_tile_index, out_w_block, conv_param, out_func);
      } else {
        WinogradOutputNC8HW8TransformFp16(gemm_out + task_id * gemm_out_offset, output_data + out_batch_offset,
                                          bias_data, cal_num, out_tile_index, out_w_block, conv_param, out_func);
      }
    }
  }
}
