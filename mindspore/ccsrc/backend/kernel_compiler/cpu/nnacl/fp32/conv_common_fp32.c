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

#include "nnacl/fp32/conv_common_fp32.h"
#include <string.h>
#include "nnacl/fp32/matmul_fp32.h"

// fp32 conv common
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param) {
  int out_channel = conv_param->output_channel_;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
#ifdef ENABLE_AVX
  const int cal_num = C6NUM;
#elif defined(ENABLE_SSE)
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
      if (real_cal_num <= 0) {
        return;
      }
      float *gemm_input = packed_input + task_id * deep * cal_num;
      float *col_major_gemm_input = col_major_input + task_id * deep * cal_num;
      size_t packed_input_size = deep * cal_num * sizeof(float);
      memset(gemm_input, 0, packed_input_size);
      memset(col_major_gemm_input, 0, packed_input_size);
      Im2ColPackUnitFp32(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * cal_num * out_channel + out_batch_offset;
      float *gemm_output = output_data + out_offset;
#ifdef ENABLE_AVX
      RowMajor2Col6Major(gemm_input, col_major_gemm_input, cal_num, deep);
#elif defined(ENABLE_SSE)
      RowMajor2Col4Major(gemm_input, col_major_gemm_input, cal_num, deep);
#else
      RowMajor2Col12Major(gemm_input, col_major_gemm_input, cal_num, deep);
#endif
      MatMulOpt(col_major_gemm_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_num,
                out_channel, out_channel, OutType_Nhwc);
    }
  }
}
