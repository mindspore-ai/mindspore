/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/conv_im2col_avx512_fp32.h"
#include "nnacl/fp32/conv_im2col_fp32.h"
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/intrinsics/ms_simd_avx512_instructions.h"

// fp32 conv common
void ConvIm2ColAVX512Fp32(const float *input_data, float *packed_input, const float *packed_weight,
                          const float *bias_data, float *output_data, int task_id, const ConvParameter *conv_param,
                          int cal_num) {
  if (conv_param->thread_num_ == 0) {
    return;
  }
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int out_channel_align = UP_ROUND(conv_param->output_channel_, C16NUM);

  int block_per_thread = UP_DIV(UP_DIV(output_hw, cal_num), conv_param->thread_num_);
  int start_block = block_per_thread * task_id;
  int start_hw = start_block * cal_num;
  int end_hw = MSMIN(output_hw, (start_block + block_per_thread) * cal_num);
  if (start_hw >= end_hw) {
    return;
  }
  int out_stride = out_channel_align * cal_num;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * cal_num;
  size_t input_size = deep * cal_num * sizeof(float);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * out_channel_align * output_hw + start_hw * out_channel_align;
    for (int i = start_hw; i < end_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(packed_input, 0, input_size);
      Im2ColDataPackUnitFp32(input_data + in_offset, conv_param, packed_input, real_cal_row, i);

      float *gemm_output = output_data + out_offset;
      MatMulAvx512Fp32(packed_input, packed_weight, gemm_output, bias_data, (size_t)conv_param->act_type_, deep,
                       out_channel_align, out_channel_align, real_cal_row);
    }
  }
}

// fp32 conv common
void ConvIm2ColAVX512Fp32CutByBatch(const float *input_data, float *packed_input, const float *packed_weight,
                                    const float *bias_data, float *output_data, int task_id,
                                    const ConvParameter *conv_param, int cal_num) {
  if (conv_param->thread_num_ == 0) {
    return;
  }
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int out_channel_align = UP_ROUND(conv_param->output_channel_, C16NUM);

  int block_batch_per_thread = UP_DIV(conv_param->input_batch_, conv_param->thread_num_);
  int start_batch = block_batch_per_thread * task_id;
  int end_batch = MSMIN(conv_param->input_batch_, (start_batch + block_batch_per_thread));

  int out_stride = out_channel_align * cal_num;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  packed_input += task_id * deep * cal_num;

  size_t input_size = deep * cal_num * sizeof(float);

  for (int b = start_batch; b < end_batch; b++) {
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * out_channel_align * output_hw;
    for (int i = 0; i < output_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(packed_input, 0, input_size);
      Im2ColDataPackUnitFp32(input_data + in_offset, conv_param, packed_input, real_cal_row, i);

      float *gemm_output = output_data + out_offset;
      MatMulAvx512Fp32(packed_input, packed_weight, gemm_output, bias_data, (size_t)conv_param->act_type_, deep,
                       out_channel_align, out_channel_align, real_cal_row);
    }
  }
}
