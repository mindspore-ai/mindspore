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

#ifdef __cplusplus
extern "C" {
#endif
#ifdef ENABLE_ARM64
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu,
                           size_t relu6);
#endif

#ifdef __cplusplus
}
#endif
#ifndef ENABLE_NEON
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t out_channel, size_t offset, size_t mode, size_t writeC8, size_t relu,
                           size_t relu6) {
  if (!(mode && writeC8)) {
    IndirectGemmFp16_16x8_common(output, input, weight, bias, step, ic4, out_channel, offset, relu, relu6);
  } else {
    IndirectGemmFp16_16x8_c8(output, input, weight, bias, step, ic4, out_channel, offset, mode, writeC8, relu, relu6);
  }
}

void IndirectGemmFp16_16x8_common(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                                  size_t ic4, size_t out_channel, size_t offset, size_t relu, size_t relu6) {
  const int tile_n = 16;
  for (int i = 0; i < out_channel; i++) {
    int oc8_block = i / C8NUM;
    int oc8_res = i % C8NUM;
    int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res;
    for (int k = 0; k < tile_n; k++) {
      int input_tile_offset = k * C4NUM;
      int out_tile_offset = i + k * out_channel;

      float16_t tmp_out = 0;
      for (int n = 0; n < step; n++) {
        int input_kw_offset = input_tile_offset + n * tile_n * ic4 * C4NUM;
        int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;
        for (int j = 0; j < ic4; j++) {
          int input_ic4_offset = input_kw_offset + j * tile_n * C4NUM;
          int weight_ic4_offset = weight_kw_offset + j * C4NUM * C8NUM;
          for (int m = 0; m < C4NUM; m++) {
            int input_c4_offset = input_ic4_offset + m;
            int weight_c4_offset = weight_ic4_offset + m * C8NUM;
            tmp_out += (input + input_c4_offset)[0] * (weight + weight_c4_offset)[0];
          }
        }
      }

      float16_t *tmp = output + out_tile_offset;
      if (bias != NULL) {
        tmp[0] = tmp_out + bias[i];
      }
      if (relu) {
        tmp[0] = tmp[0] < 0 ? 0 : tmp[0];
      } else if (relu6) {
        tmp[0] = tmp[0] < 0 ? 0 : tmp[0];
        tmp[0] = tmp[0] > 6 ? 6 : tmp[0];
      }
    }
  }
}

void IndirectGemmFp16_16x8_c8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                              size_t ic4, size_t output_channel, size_t offset, size_t mode, size_t writeC8,
                              size_t relu, size_t relu6) {
  const int tile_num = 16;
  if (mode && writeC8) {
    for (int i = 0; i < tile_num; i++) {
      int input_tile_offset = i * C4NUM;
      int output_tile_offset = i * output_channel * step;
      for (int j = 0; j < output_channel; j++) {
        int oc8_block = j / C8NUM;
        int oc8_res = j % C8NUM;
        int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res;
        int out_oc_offset = output_tile_offset + oc8_block * step * C8NUM + oc8_res;

        for (int n = 0; n < step; n++) {
          int input_kw_offset = input_tile_offset + n * ic4 * C4NUM * tile_num;
          int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;
          int output_kw_offset = out_oc_offset + n * C8NUM;
          float16_t acc = 0;

          for (int k = 0; k < ic4; k++) {
            int input_ic4_offset = input_kw_offset + k * tile_num * C4NUM;
            int weight_ic4_offset = weight_kw_offset + k * C4NUM * C8NUM;
            for (int m = 0; m < C4NUM; m++) {
              int input_ic_offset = input_ic4_offset + m;
              int weight_ic_offset = weight_ic4_offset + m * C8NUM;
              acc += (weight + weight_ic_offset)[0] * (input + input_ic_offset)[0];
            }
          }

          (output + output_kw_offset)[0] = acc;
        }
      }
    }
  } else {
  }
}
#endif

// fp16 convolution common (im2col+gemm)
void ConvFp16(float16_t *input_data, float16_t *packed_input, float16_t *packed_weight, float16_t *bias_data,
              float16_t *col_major_input, float16_t *output_data, int task_id, ConvParameter *conv_param) {
  const int tile_n = 16;
  int out_channel = conv_param->output_channel_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_n);
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * out_channel * output_count;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      float16_t *gemm_input = packed_input + task_id * deep * tile_n;
      float16_t *col_major_gemm_input = col_major_input + task_id * deep * tile_n;
      size_t packed_input_size = deep * tile_n * sizeof(float16_t);
      memset(gemm_input, 0, packed_input_size);
      memset(col_major_gemm_input, 0, packed_input_size);
      Im2ColPackUnitFp16(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;
      RowMajor2Col16MajorFp16Opt(gemm_input, col_major_gemm_input, tile_n, deep);
      MatMulFp16(col_major_gemm_input, packed_weight, output_data + out_offset, bias_data, conv_param->act_type_, deep,
                 real_cal_num, out_channel, out_channel, OutType_Nhwc);
    }
  }
}

// fp16 convolution winograd
void ConvWinogardFp16(float16_t *input_data, float16_t *trans_weight, const float16_t *bias_data,
                      float16_t *output_data, TmpBufferAddressFp16 *buffer_list, int task_id, ConvParameter *conv_param,
                      InputTransFp16Func in_func, OutputTransFp16Func out_func) {
  const int tile_num = 16;
  int in_channel = conv_param->input_channel_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  int per_thread_num = UP_DIV(output_count, conv_param->thread_num_);
  int real_tile = per_thread_num < tile_num ? per_thread_num : tile_num;
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
        RowMajor2Col16MajorFp16Opt(src_ptr + i * tile_num * in_channel, tmp_col_ptr, cal_num, in_channel);
        MatMulFp16(tmp_col_ptr, trans_weight + i * in_channel * oc8 * C8NUM, dst_ptr + i * C8NUM, NULL, 0, in_channel,
                   cal_num, oc8 * C8NUM, input_unit_square, OutType_TileC8);
      }

      // step 4 : output transform
      WinogradOutputTransformFp16(gemm_out + task_id * gemm_out_offset, output_data + out_batch_offset, bias_data,
                                  cal_num, out_tile_index, out_w_block, conv_param, out_func);
    }
  }
}
