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

#include "nnacl/fp32/adder_fp32.h"
#include <string.h>
#include <math.h>
#include "nnacl/fp32/matmul_fp32.h"

void Adder12x4(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
               int col, int stride) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r12div = r / 12, r12mod = r % 12;
      int c4div = c / 4, c4mod = c % 4;
      size_t ci = r * stride + c;
      float value = 0;
      for (int d = 0; d < deep; d++) {
        size_t ai = r12div * deep * 12 + d * 12 + r12mod;
        size_t bi = c4div * deep * 4 + d * 4 + c4mod;
        value += fabsf(a[ai] - b[bi]);
      }
      value = -value;
      if (bias != NULL) value += bias[c];
      if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
      if (act_type != ActType_No) value = MSMAX(0.0f, value);
      dst[ci] = value;
    }
  }
}

void AdderOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row, int col,
              size_t stride) {
#ifdef ENABLE_ARM64
  AdderFloatNeon64(a, b, c, bias, (int)act_type, deep, row, col, stride);
#else
  Adder12x4(a, b, c, bias, act_type, deep, row, col, stride);
#endif
}

void AdderFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
               float *col_major_input, float *output_data, int task_id, ConvParameter *conv_param) {
  int out_channel = conv_param->output_channel_;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
  if (conv_param->thread_num_ == 0) {
    return;
  }
#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
  const int cal_num = C4NUM;
#else
  const int cal_num = C12NUM;
#endif
  int output_tile_count = UP_DIV(output_count, cal_num);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * out_channel * output_count;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * cal_num;
      int real_cal_num = (output_count - start_index) < cal_num ? (output_count - start_index) : cal_num;
      float *gemm_input = packed_input + task_id * deep * cal_num;
      float *col_major_gemm_input = col_major_input + task_id * deep * cal_num;
      size_t packed_input_size = deep * cal_num * sizeof(float);
      memset(gemm_input, 0, packed_input_size);
      memset(col_major_gemm_input, 0, packed_input_size);
      Im2ColPackUnitFp32(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * cal_num * out_channel + out_batch_offset;
      float *gemm_output = output_data + out_offset;
#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
      RowMajor2Col4Major(gemm_input, col_major_gemm_input, cal_num, deep);
#else
      RowMajor2Col12Major(gemm_input, col_major_gemm_input, cal_num, deep);
#endif
      AdderOpt(col_major_gemm_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_num,
               out_channel, out_channel);
    }
  }
}
