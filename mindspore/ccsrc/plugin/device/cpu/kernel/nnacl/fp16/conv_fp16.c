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

void Im2ColPackUnitFp16(const float16_t *input_data, const ConvParameter *conv_param, float16_t *packed_input,
                        int real_cal_num, int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int kernel_plane = kernel_h * kernel_w;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * stride_h - pad_h;
    int input_w = block_start % out_w * stride_w - pad_w;
    int input_stride = (input_h * in_w + input_w) * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(in_h - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    if (dilation_h == 1 && dilation_w == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        memcpy(packed_input + input_plane_offset, input_data + input_x_stride,
               (kw_e - kw_s) * in_channel * sizeof(float16_t));
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int n = kw_s; n < kw_e; n++) {
          int input_x_stride = input_y_stride + n * dilation_w * in_channel;
          int input_plane_offset = (j * kernel_w + n) * in_channel + i * in_channel * kernel_plane;
          memcpy(packed_input + input_plane_offset, input_data + input_x_stride, in_channel * sizeof(float16_t));
        }  // kernel_w loop
      }    // kernel_h loop
    }
  }  // tile num loop
}

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
                      const ConvParameter *conv_param, TransFp16FuncList trans_func) {
#ifdef ENABLE_ARM64
  const int tile_num = 16;
#else
  const int tile_num = 12;
#endif
  NNACL_CHECK_ZERO_RETURN(conv_param->output_unit_);
  NNACL_CHECK_ZERO_RETURN(conv_param->thread_num_);
  int in_channel = conv_param->input_channel_;
  int input_unit = conv_param->input_unit_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  int per_thread_num = UP_DIV(output_count, conv_param->thread_num_);
  int real_tile = per_thread_num < tile_num ? per_thread_num : tile_num;
  NNACL_CHECK_ZERO_RETURN(real_tile);
  int output_tile_count = UP_DIV(output_count, real_tile);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);
  int input_unit_square = input_unit * input_unit;

  float16_t *trans_input = buffer_list[0] + task_id * tile_num * input_unit_square * in_channel;
  float16_t *gemm_out = buffer_list[1] + task_id * tile_num * input_unit_square * oc8 * C8NUM;
  float16_t *tmp_data = buffer_list[2] + task_id * input_unit_square * C8NUM;
  float16_t *col_buffer = buffer_list[3] + task_id * tile_num * in_channel;
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

#ifdef ENABLE_ARM64
      // Optimize input transform. Only valid for arm64, the tile num is 16.
      // For arm32, the tile_num is 12. The function(InputTransform4x4Pack12Fp16) needs to be rewritten.
      bool fused_pack =
        (cal_num == tile_num) && (trans_func.in_step_func_ != NULL) && (trans_func.in_pack_func_ != NULL);
      if (fused_pack) {
        float16_t *opt_trans_input =
          buffer_list[4] + task_id * tile_num * input_unit_square * UP_ROUND(in_channel, C8NUM);
        WinogradInputTransformOptStepFp16(input_data + in_batch_offset, opt_trans_input, tmp_data, cal_num,
                                          out_tile_index, out_w_block, conv_param, trans_func.in_step_func_);

        for (int w_index = 0; w_index < input_unit; w_index++) {
          float16_t *src_w = opt_trans_input + w_index * input_unit * tile_num * C8NUM;
          for (int c = 0; c < UP_DIV(in_channel, C8NUM); c++) {
            int real_c = in_channel - c * C8NUM;
            real_c = real_c > C8NUM ? C8NUM : real_c;
            float16_t *src_c = src_w + c * input_unit_square * tile_num * C8NUM;
            float16_t *dst_c = trans_input + c * tile_num * C8NUM;
            trans_func.in_pack_func_(src_c, dst_c, C8NUM, in_channel * tile_num, real_c);
          }

          for (int h_index = 0; h_index < input_unit; h_index++) {
            const float16_t *gemm_input = trans_input + h_index * tile_num * in_channel;
            int point_index = h_index * input_unit + w_index;
            const float16_t *gemm_weight = trans_weight + point_index * in_channel * oc8 * C8NUM;
            MatMulFp16(gemm_input, gemm_weight, gemm_out + point_index * C8NUM, NULL, 0, in_channel, cal_num,
                       oc8 * C8NUM, input_unit_square, OutType_TileC8);
          }
        }
      } else {
#endif
        WinogradInputTransformFp16(input_data + in_batch_offset, trans_input, tmp_data, cal_num, out_tile_index,
                                   out_w_block, conv_param, trans_func.in_func_);
        // step 3 : gemm
        float16_t *src_ptr = trans_input;
        float16_t *dst_ptr = gemm_out;
        float16_t *tmp_col_ptr = col_buffer;
        for (int i = 0; i < input_unit_square; ++i) {
#ifdef ENABLE_ARM64
          RowMajor2Col16MajorFp16Opt(src_ptr + i * tile_num * in_channel, tmp_col_ptr, cal_num, in_channel);
#else
        RowMajor2Col12MajorFp16Opt(src_ptr + i * tile_num * in_channel, tmp_col_ptr, cal_num, in_channel);
#endif
          MatMulFp16(tmp_col_ptr, trans_weight + i * in_channel * oc8 * C8NUM, dst_ptr + i * C8NUM, NULL, 0, in_channel,
                     cal_num, oc8 * C8NUM, input_unit_square, OutType_TileC8);
        }
#ifdef ENABLE_ARM64
      }
#endif

      // step 4 : output transform
      if (conv_param->out_format_ != Format_NC4HW4) {  // nc4hw4
        WinogradOutputNHWCTransformFp16(gemm_out, output_data + out_batch_offset, bias_data, cal_num, out_tile_index,
                                        out_w_block, conv_param, trans_func.out_func_);
      } else {
        WinogradOutputNC8HW8TransformFp16(gemm_out, output_data + out_batch_offset, bias_data, cal_num, out_tile_index,
                                          out_w_block, conv_param, trans_func.out_func_);
      }
    }
  }
}
