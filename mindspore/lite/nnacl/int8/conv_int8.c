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

#include "nnacl/int8/conv_int8.h"
#include <string.h>
#include "nnacl/winograd_transform.h"
#include "nnacl/int8/common_func.h"

void IndirectGemmInt8(int8_t *dst, int32_t *tmp_dst, const int8_t *src, const int8_t *weight, const int32_t *bias,
                      int ic4, size_t kernel_plane, size_t output_channel, const int32_t *input_sum,
                      ConvParameter *conv_param) {
  int32_t *shift_before = conv_param->conv_quant_arg_.left_shift_;
  int32_t *shift_after = conv_param->conv_quant_arg_.right_shift_;
  int32_t *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int32_t out_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int32_t act_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int32_t act_max = conv_param->conv_quant_arg_.out_act_max_[0];

#ifdef ENABLE_ARM64
  size_t asymmetric = conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC;
  size_t per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  IndirectGemmInt8_4x4(dst, src, weight, bias, UP_DIV(kernel_plane, C4NUM), ic4, output_channel,
                       output_channel * sizeof(int8_t), input_sum, act_min, act_max, out_zp, out_multiplier,
                       shift_before, shift_after, asymmetric, per_channel);
#else
  int oc4 = UP_DIV(output_channel, C4NUM);
  int tile_num = conv_param->tile_num_;
  int plane_c4 = UP_DIV(kernel_plane, C4NUM);
  for (int oc = 0; oc < output_channel; oc++) {
    int oc4_block = oc / C4NUM;
    int oc4_res = oc % C4NUM;
    int weight_oc4_offset = oc4_block * C4NUM * plane_c4 * C4NUM * ic4 * C4NUM + oc4_res * C4NUM * C4NUM;
    int dst_oc_offset = oc;
    for (int n = 0; n < tile_num; n++) {
      int src_tile_offset = n * C4NUM * C4NUM;
      int dst_tile_offset = dst_oc_offset + n * output_channel;

      for (int b = 0; b < kernel_plane; b++) {
        int plane_c4_block = b / C4NUM;
        int plane_c4_res = b % C4NUM;
        int src_plane_offset = src_tile_offset + plane_c4_block * tile_num * C4NUM * ic4 * C4NUM + plane_c4_res * C4NUM;
        int weight_plane_offset =
          weight_oc4_offset + plane_c4_block * C4NUM * C4NUM * ic4 * C4NUM + plane_c4_res * C4NUM;
        for (int i = 0; i < ic4; i++) {
          int src_ic4_offset = src_plane_offset + i * tile_num * C4NUM * C4NUM;
          int weight_ic4_offset = weight_plane_offset + i * C4NUM * C4NUM * C4NUM;
          for (int j = 0; j < C4NUM; j++) {
            int weight_ic_offset = weight_ic4_offset + j;
            tmp_dst[dst_tile_offset] += weight[weight_ic_offset] * src[src_ic4_offset + j];
          }  // in c4num loop
        }    // ic4 loop
      }      // kernel_plane loop
      if (!(conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
          (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
        int result = tmp_dst[dst_tile_offset] + bias[oc];
        result = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[oc]), out_multiplier[oc]),
          -shift_after[oc]);
        result += out_zp;
        result = result > act_min ? result : act_min;
        result = result < act_max ? result : act_max;
        dst[dst_tile_offset] = (int8_t)result;
      } else if (!(conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                 !(conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
        int result = tmp_dst[dst_tile_offset] + bias[oc];
        result = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[0]), out_multiplier[0]),
          -shift_after[0]);
        result += out_zp;
        result = result > act_min ? result : act_min;
        result = result < act_max ? result : act_max;
        dst[dst_tile_offset] = (int8_t)result;
      } else if ((conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                 !(conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
        tmp_dst[dst_tile_offset] -= input_sum[n];
        int result = tmp_dst[dst_tile_offset] + bias[oc];
        result = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[0]), out_multiplier[0]),
          -shift_after[0]);
        result += out_zp;
        result = result > act_min ? result : act_min;
        result = result < act_max ? result : act_max;
        dst[dst_tile_offset] = (int8_t)result;
      } else if ((conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                 (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
        tmp_dst[dst_tile_offset] -= input_sum[n * oc4 * C4NUM + oc];
        int result = tmp_dst[dst_tile_offset] + bias[oc];
        result = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[oc]), out_multiplier[oc]),
          -shift_after[oc]);
        result += out_zp;
        result = result > act_min ? result : act_min;
        result = result < act_max ? result : act_max;
        dst[dst_tile_offset] = (int8_t)result;
      }
    }  // tile_num loop
  }    // output_channel loop
#endif
}

void IndirectGemmInt8Opt(int8_t *dst, int32_t *tmp_dst, const int8_t *src, const int8_t *weight, const int32_t *bias,
                         int ic4, size_t kernel_plane, size_t output_channel, const int32_t *input_sum,
                         ConvParameter *conv_param, GEMM_FUNC gemm_func) {
  int32_t *shift_before = conv_param->conv_quant_arg_.left_shift_;
  int32_t *shift_after = conv_param->conv_quant_arg_.right_shift_;
  int32_t *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int32_t out_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int32_t act_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int32_t act_max = conv_param->conv_quant_arg_.out_act_max_[0];
  int oc4 = UP_DIV(output_channel, C4NUM);
  if (gemm_func != NULL) {
#ifdef __aarch64__
    size_t asymmetric = conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC;
    size_t per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
    gemm_func(dst, src, weight, bias, kernel_plane, ic4, output_channel, output_channel * sizeof(int8_t), input_sum,
              act_min, act_max, out_zp, out_multiplier, shift_before, shift_after, asymmetric, per_channel);
#endif
  } else {
    int tile_num = conv_param->tile_num_;
    for (int oc = 0; oc < output_channel; oc++) {
      int oc4_block = oc / C4NUM;
      int oc4_res = oc % C4NUM;
      int weight_oc4_offset = oc4_block * C4NUM * kernel_plane * ic4 * C4NUM + oc4_res * C4NUM;
      int dst_oc_offset = oc;
      for (int n = 0; n < tile_num; n++) {
        int src_tile_offset = n * C4NUM;
        int dst_tile_offset = dst_oc_offset + n * output_channel;

        for (int b = 0; b < kernel_plane; b++) {
          int src_plane_offset = src_tile_offset + b * tile_num * ic4 * C4NUM;
          int weight_plane_offset = weight_oc4_offset + b * C4NUM * ic4 * C4NUM;
          for (int i = 0; i < ic4; i++) {
            int src_ic4_offset = src_plane_offset + i * tile_num * C4NUM;
            int weight_ic4_offset = weight_plane_offset + i * C4NUM * C4NUM;
            for (int j = 0; j < C4NUM; j++) {
              int weight_ic_offset = weight_ic4_offset + j;
              tmp_dst[dst_tile_offset] += weight[weight_ic_offset] * src[src_ic4_offset + j];
            }  // in c4num loop
          }    // ic4 loop
        }      // kernel_plane loop
        if (!(conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
            (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
          int result = tmp_dst[dst_tile_offset] + bias[oc];
          result = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[oc]), out_multiplier[oc]),
            -shift_after[oc]);
          result += out_zp;
          result = result > act_min ? result : act_min;
          result = result < act_max ? result : act_max;
          dst[dst_tile_offset] = (int8_t)result;
        } else if (!(conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                   !(conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
          int result = tmp_dst[dst_tile_offset] + bias[oc];
          result = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[0]), out_multiplier[0]),
            -shift_after[0]);
          result += out_zp;
          result = result > act_min ? result : act_min;
          result = result < act_max ? result : act_max;
          dst[dst_tile_offset] = (int8_t)result;
        } else if ((conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                   !(conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
          tmp_dst[dst_tile_offset] -= input_sum[n];
          int result = tmp_dst[dst_tile_offset] + bias[oc];
          result = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[0]), out_multiplier[0]),
            -shift_after[0]);
          result += out_zp;
          result = result > act_min ? result : act_min;
          result = result < act_max ? result : act_max;
          dst[dst_tile_offset] = (int8_t)result;
        } else if ((conv_param->conv_quant_arg_.asymmetric_ & FILTER_ASYMMETRIC) &&
                   (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
          tmp_dst[dst_tile_offset] -= input_sum[n * oc4 * C4NUM + oc];
          int result = tmp_dst[dst_tile_offset] + bias[oc];
          result = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(result * (1 << (unsigned int)shift_before[oc]), out_multiplier[oc]),
            -shift_after[oc]);
          result += out_zp;
          result = result > act_min ? result : act_min;
          result = result < act_max ? result : act_max;
          dst[dst_tile_offset] = (int8_t)result;
        }
      }  // tile_num loop
    }    // output_channel loop
  }
}

void Conv3x3Int8Gemm(int32_t *dst, const int16_t *src, const int16_t *weight, int oc, int ic8, size_t real_cal_num) {
  int oc4 = UP_DIV(oc, C4NUM);
#ifdef ENABLE_ARM64
  IndirectGemmInt16to32_8x4(dst, src, weight, 16, ic8, oc4, oc4 * 4 * 16 * sizeof(int32_t));
#else
  const int input_unit_square = 16;
  for (int c = 0; c < oc4; c++) {
    int filter_oc_offset = c * input_unit_square * ic8 * C8NUM * C4NUM;
    int dst_oc_offset = c * input_unit_square * C4NUM;
    for (int n = 0; n < real_cal_num; n++) {
      int src_tile_offset = n * C8NUM;
      int dst_tile_offset = dst_oc_offset + n * oc4 * C4NUM * input_unit_square;
      for (int i = 0; i < 4; i++) {
        int filter_h_offset = filter_oc_offset + i * 4 * ic8 * C8NUM * C4NUM;
        int src_h_offset = src_tile_offset + i * C8NUM * ic8 * C8NUM * C4NUM;
        int dst_h_offset = dst_tile_offset + i * 4 * 4;
        for (int m = 0; m < 4; m++) {
          int filter_w_offset = filter_h_offset + m * 4 * C8NUM * ic8;
          int src_w_offset = src_h_offset + m * 8 * ic8 * C8NUM;
          int dst_w_offset = dst_h_offset + m * C4NUM;

          int32_t acc[4] = {0};
          for (int z = 0; z < 4; z++) {
            int filter_offset = filter_w_offset + z;
            for (int j = 0; j < ic8; j++) {
              int filter_c8_offset = filter_offset + j * 4 * 8;
              int src_c8_offset = src_w_offset + j * 8 * 8;

              for (int k = 0; k < 8; k++) {
                const int16_t *w_ptr = weight + filter_c8_offset + k * 4;
                const int16_t *input_ptr = src + src_c8_offset + k;
                acc[z] += w_ptr[0] * input_ptr[0];
              }
            }
            (dst + dst_w_offset + z)[0] = acc[z];
          }
        }
      }
    }
  }
#endif
}

// int8 conv common
void ConvInt8(int8_t *input_data, int8_t *packed_input, int8_t *packed_weight, const int32_t *bias_data,
              int32_t *tmp_dst, int8_t *tmp_out, int8_t *output_data, int32_t *input_sum, int task_id,
              ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int32_t input_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;

  int tile_n = conv_param->tile_num_;
  int thread_count = conv_param->thread_num_;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, tile_n);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int plane_block = UP_DIV(kernel_plane, C4NUM);
  int unit_size = plane_block * C4NUM * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_n * unit_size;
  int input_sum_offset;
  if (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_offset = tile_n * oc4 * C4NUM;
  } else {
    input_sum_offset = tile_n;
  }

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * ic4 * C4NUM * in_h * in_w;
    int out_batch_offset = b * out_channel * out_h * out_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      int32_t *tmp_input_sum = input_sum + task_id * input_sum_offset;
      int8_t *gemm_input = packed_input + thread_id * unit_size * tile_n + gemm_in_batch_offset;
      // clear tmp buffer before compute
      memset(gemm_input, (int8_t)input_zp, unit_size * tile_n);
      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;

      size_t tmp_dst_size = tile_n * conv_param->output_channel_ * sizeof(int32_t);
      int tmp_dst_offset = task_id * tile_n * conv_param->output_channel_;
      memset(tmp_dst + tmp_dst_offset, 0, tmp_dst_size);

      Im2ColPackUnitInt8(input_data + in_batch_offset, gemm_input, real_cal_num, start_index, tmp_input_sum,
                         conv_param);
      if (real_cal_num == tile_n) {
        int8_t *gemm_output = output_data + out_offset;
        IndirectGemmInt8(gemm_output, tmp_dst + tmp_dst_offset, gemm_input, packed_weight, bias_data, ic4, kernel_plane,
                         out_channel, tmp_input_sum, conv_param);
      } else {
        // res part
        int8_t *tmp_out_ptr = tmp_out + task_id * tile_n * out_channel;
        IndirectGemmInt8(tmp_out_ptr, tmp_dst + tmp_dst_offset, gemm_input, packed_weight, bias_data, ic4, kernel_plane,
                         out_channel, tmp_input_sum, conv_param);
        memcpy(output_data + out_offset, tmp_out_ptr, real_cal_num * out_channel);
      }
    }
  }
}

void ConvInt8Opt(int8_t *input_data, int8_t *packed_input, int8_t *packed_weight, const int32_t *bias_data,
                 int32_t *tmp_dst, int8_t *tmp_out, int8_t *output_data, int32_t *input_sum, int task_id,
                 ConvParameter *conv_param, GEMM_FUNC gemm_func) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int32_t input_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;
  int tile_n = conv_param->tile_num_;
  int thread_count = conv_param->thread_num_;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, tile_n);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_n * unit_size;
  int input_sum_offset;
  if (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_offset = tile_n * oc4 * C4NUM;
  } else {
    input_sum_offset = tile_n;
  }

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * ic4 * C4NUM * in_h * in_w;
    int out_batch_offset = b * out_channel * out_h * out_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      int32_t *tmp_input_sum = input_sum + task_id * input_sum_offset;
      int8_t *gemm_input = packed_input + thread_id * unit_size * tile_n + gemm_in_batch_offset;
      // clear tmp buffer before compute
      memset(gemm_input, (int8_t)input_zp, unit_size * tile_n);
      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;

      size_t tmp_dst_size = tile_n * conv_param->output_channel_ * sizeof(int32_t);
      int tmp_dst_offset = task_id * tile_n * conv_param->output_channel_;
      memset(tmp_dst + tmp_dst_offset, 0, tmp_dst_size);

      Im2ColPackUnitInt8Opt(input_data + in_batch_offset, gemm_input, real_cal_num, start_index, tmp_input_sum,
                            conv_param);
      if (real_cal_num == tile_n) {
        int8_t *gemm_output = output_data + out_offset;
        IndirectGemmInt8Opt(gemm_output, tmp_dst + tmp_dst_offset, gemm_input, packed_weight, bias_data, ic4,
                            kernel_plane, out_channel, tmp_input_sum, conv_param, gemm_func);
      } else {
        // res part
        int8_t *tmp_out_ptr = tmp_out + task_id * tile_n * out_channel;
        IndirectGemmInt8Opt(tmp_out_ptr, tmp_dst + tmp_dst_offset, gemm_input, packed_weight, bias_data, ic4,
                            kernel_plane, out_channel, tmp_input_sum, conv_param, gemm_func);
        memcpy(output_data + out_offset, tmp_out_ptr, real_cal_num * out_channel);
      }
    }
  }
}

void Conv1x1PreOpt(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                   size_t output_channel, size_t plane_size, ConvParameter *conv_param) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  size_t hw_8div = plane_size / C8NUM * C8NUM;
  size_t hw_8res = plane_size - hw_8div;
  size_t ic_4div = input_channel / C4NUM * C4NUM;
  int32_t filter_zp = conv_param->conv_quant_arg_.filter_quant_args_[0].zp_;

  if (conv_param->conv_quant_arg_.filter_arg_num_ == 1) {
    const int8_t *src_r = src_input;
    int8_t *pack_r = packed_input;
    /* per layer */
    for (int hwi = 0; hwi < hw_8div; hwi += C8NUM) {
      const int8_t *src_ic = src_r;
      int8_t *pack_ic = pack_r;
      int32_t *input_sum_r = input_sum + hwi;
#ifdef ENABLE_ARM64
      size_t src_stride = input_channel;
      size_t ic_4res = input_channel - ic_4div;
      asm volatile(
        "dup v10.4s, wzr \n"
        "dup v11.4s, wzr \n"
        "mov x20, %[input_sum_r] \n"
        "dup v20.4s, %w[filter_zp]  \n"

        "mov x10, %[src_ic] \n"
        "mov x11, %[pack_ic] \n"

        "mov x0, #0 \n"
        "1: \n"
        "cmp x0, %[ic_4div] \n"
        "add x0, x0, #4\n"
        "mov x12, x10 \n"
        "add x10, x10, #4\n"
        "blt 2f \n"
        "cmp %[ic_4res], #0\n"
        "beq 6f \n"
        "cmp %[ic_4res], #1\n"
        "beq 3f \n"
        "cmp %[ic_4res], #2\n"
        "beq 4f \n"
        "cmp %[ic_4res], #3\n"
        "beq 5f \n"

        "2: \n"
        "ld1 {v0.s}[0], [x12], %[src_stride]\n"
        "ld1 {v0.s}[1], [x12], %[src_stride]\n"
        "ld1 {v0.s}[2], [x12], %[src_stride]\n"
        "ld1 {v0.s}[3], [x12], %[src_stride]\n"
        "ld1 {v1.s}[0], [x12], %[src_stride]\n"
        "ld1 {v1.s}[1], [x12], %[src_stride]\n"
        "ld1 {v1.s}[2], [x12], %[src_stride]\n"
        "ld1 {v1.s}[3], [x12], %[src_stride]\n"

        "st1 {v0.16b}, [x11], #16\n"
        "st1 {v1.16b}, [x11], #16\n"

        "saddlp v4.8h, v0.16b \n"
        "saddlp v5.8h, v1.16b \n"

        "saddlp v0.4s, v4.8h \n"
        "saddlp v1.4s, v5.8h \n"

        "add v10.4s, v10.4s, v0.4s \n"
        "add v11.4s, v11.4s, v1.4s \n"
        "b 1b \n"

        "3: \n" /* col res 1 */
        "dup v0.4s, wzr \n"
        "dup v1.4s, wzr \n"

        "ld1 {v0.b}[0],  [x12], %[src_stride]\n"
        "ld1 {v0.b}[4],  [x12], %[src_stride]\n"
        "ld1 {v0.b}[8],  [x12], %[src_stride]\n"
        "ld1 {v0.b}[12], [x12], %[src_stride]\n"
        "ld1 {v1.b}[0],  [x12], %[src_stride]\n"
        "ld1 {v1.b}[4],  [x12], %[src_stride]\n"
        "ld1 {v1.b}[8],  [x12], %[src_stride]\n"
        "ld1 {v1.b}[12], [x12], %[src_stride]\n"

        "st1 {v0.16b}, [x11], #16\n"
        "st1 {v1.16b}, [x11], #16\n"
        "saddlp v4.8h, v0.16b \n"
        "saddlp v5.8h, v1.16b \n"
        "saddlp v0.4s, v4.8h \n"
        "saddlp v1.4s, v5.8h \n"
        "add v10.4s, v10.4s, v0.4s \n"
        "add v11.4s, v11.4s, v1.4s \n"
        "b 6f \n"

        "4: \n" /* col res 2 */
        "dup v0.4s, wzr \n"
        "dup v1.4s, wzr \n"

        "ld1 {v0.h}[0], [x12], %[src_stride]\n"
        "ld1 {v0.h}[2], [x12], %[src_stride]\n"
        "ld1 {v0.h}[4], [x12], %[src_stride]\n"
        "ld1 {v0.h}[6], [x12], %[src_stride]\n"
        "ld1 {v1.h}[0], [x12], %[src_stride]\n"
        "ld1 {v1.h}[2], [x12], %[src_stride]\n"
        "ld1 {v1.h}[4], [x12], %[src_stride]\n"
        "ld1 {v1.h}[6], [x12], %[src_stride]\n"

        "st1 {v0.16b}, [x11], #16\n"
        "st1 {v1.16b}, [x11], #16\n"
        "saddlp v4.8h, v0.16b \n"
        "saddlp v5.8h, v1.16b \n"
        "saddlp v0.4s, v4.8h \n"
        "saddlp v1.4s, v5.8h \n"
        "add v10.4s, v10.4s, v0.4s \n"
        "add v11.4s, v11.4s, v1.4s \n"
        "b 6f \n"

        "5: \n" /* col res 3 */
        "dup v0.4s, wzr \n"
        "dup v1.4s, wzr \n"
        "add x13, x12, #2 \n"

        "ld1 {v0.h}[0], [x12], %[src_stride]\n"
        "ld1 {v0.b}[2], [x13], %[src_stride]\n"
        "ld1 {v0.h}[2], [x12], %[src_stride]\n"
        "ld1 {v0.b}[6], [x13], %[src_stride]\n"
        "ld1 {v0.h}[4], [x12], %[src_stride]\n"
        "ld1 {v0.b}[10], [x13], %[src_stride]\n"
        "ld1 {v0.h}[6], [x12], %[src_stride]\n"
        "ld1 {v0.b}[14], [x13], %[src_stride]\n"
        "ld1 {v1.h}[0], [x12], %[src_stride]\n"
        "ld1 {v1.b}[2], [x13], %[src_stride]\n"
        "ld1 {v1.h}[2], [x12], %[src_stride]\n"
        "ld1 {v1.b}[6], [x13], %[src_stride]\n"
        "ld1 {v1.h}[4], [x12], %[src_stride]\n"
        "ld1 {v1.b}[10], [x13], %[src_stride]\n"
        "ld1 {v1.h}[6], [x12], %[src_stride]\n"
        "ld1 {v1.b}[14], [x13], %[src_stride]\n"

        "st1 {v0.16b}, [x11], #16\n"
        "st1 {v1.16b}, [x11], #16\n"
        "saddlp v4.8h, v0.16b \n"
        "saddlp v5.8h, v1.16b \n"
        "saddlp v0.4s, v4.8h \n"
        "saddlp v1.4s, v5.8h \n"
        "add v10.4s, v10.4s, v0.4s \n"
        "add v11.4s, v11.4s, v1.4s \n"
        "b 6f \n"

        "6: \n"
        "mul v10.4s, v10.4s, v20.4s \n"
        "mul v11.4s, v11.4s, v20.4s \n"

        "st1 {v10.4s}, [x20], #16 \n"
        "st1 {v11.4s}, [x20], #16 \n"

        :
        : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ input_sum_r ] "r"(input_sum_r),
          [ src_stride ] "r"(src_stride), [ ic_4div ] "r"(ic_4div), [ ic_4res ] "r"(ic_4res),
          [ filter_zp ] "r"(filter_zp)
        : "x0", "x1", "x10", "x11", "x12", "x13", "x20", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11",
          "v20");
#else
      int32_t tmp_sum_value[8] = {0};
      for (int ici = 0; ici < ic_4div; ici += C4NUM) {
        for (int i = 0; i < C8NUM; i++) {
          tmp_sum_value[i] += src_ic[0 + i * input_channel];
          tmp_sum_value[i] += src_ic[1 + i * input_channel];
          tmp_sum_value[i] += src_ic[2 + i * input_channel];
          tmp_sum_value[i] += src_ic[3 + i * input_channel];
          pack_ic[0 + i * C4NUM] = src_ic[0 + i * input_channel];
          pack_ic[1 + i * C4NUM] = src_ic[1 + i * input_channel];
          pack_ic[2 + i * C4NUM] = src_ic[2 + i * input_channel];
          pack_ic[3 + i * C4NUM] = src_ic[3 + i * input_channel];
        }
        src_ic += C4NUM;
        pack_ic += C4NUM * C8NUM;
      }
      for (int ici = ic_4div; ici < input_channel; ici += 1) {
        for (int i = 0; i < C8NUM; i++) {
          tmp_sum_value[i] += src_ic[i * input_channel];
          pack_ic[i * C4NUM] = src_ic[i * input_channel];
        }
        src_ic += 1;
        pack_ic += 1;
      }

      for (int i = 0; i < C8NUM; i++) {
        input_sum_r[i] = tmp_sum_value[i] * filter_zp;
      }
#endif
      src_r += input_channel * C8NUM;
      pack_r += ic4 * C8NUM;
    }

    if (hw_8div != plane_size) {
      memset(pack_r, 0, C8NUM * ic4);
      for (int hwi = hw_8div; hwi < plane_size; hwi += 1) {
        int32_t tmp_sum_value = 0;
        const int8_t *src_ic = src_r;
        int8_t *pack_ic = pack_r;
        for (int ici = 0; ici < ic_4div; ici += C4NUM) {
          tmp_sum_value += src_ic[0];
          tmp_sum_value += src_ic[1];
          tmp_sum_value += src_ic[2];
          tmp_sum_value += src_ic[3];
          pack_ic[0] = src_ic[0];
          pack_ic[1] = src_ic[1];
          pack_ic[2] = src_ic[2];
          pack_ic[3] = src_ic[3];
          src_ic += C4NUM;
          pack_ic += C4NUM * C8NUM;
        }
        for (int ici = ic_4div; ici < input_channel; ici += 1) {
          tmp_sum_value += src_ic[0];
          pack_ic[0] = src_ic[0];
          src_ic += 1;
          pack_ic += 1;
        }
        input_sum[hwi] = tmp_sum_value * filter_zp;
        src_r += input_channel;
        pack_r += C4NUM;
      }
      for (int hwi = plane_size; hwi < plane_size + hw_8res; hwi++) {
        input_sum[hwi] = 0;
      }
    }
  } else {
    /* per channel */
    RowMajor2Row4x8MajorInt8(src_input, packed_input, plane_size, input_channel);
    PackInputSum8x4Int8(packed_input, input_sum, input_channel, output_channel, plane_size, conv_param);
  }
  return;
}

void Conv1x1Int8Opt(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                    const int32_t *bias, int row, int col, int deep4, ConvParameter *conv_param,
                    MATMUL_OPT_R_FUNC matmul_func) {
  matmul_func(packed_input, packed_weight, dst, row, col, deep4, conv_param->output_channel_, input_sum, bias,
              conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_,
              conv_param->conv_quant_arg_.quant_multiplier_, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
              conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0], false);
  return;
}

void Conv1x1Int8(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                 const int32_t *bias, int row, int col, int deep16, ConvParameter *conv_param) {
#ifdef ENABLE_ARM64
  MatmulInt8Neon64(packed_input, packed_weight, dst, UP_ROUND(row, C4NUM), UP_ROUND(col, C4NUM), deep16, input_sum,
                   bias, conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
                   conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                   conv_param->conv_quant_arg_.quant_multiplier_[0], conv_param->conv_quant_arg_.left_shift_[0],
                   conv_param->conv_quant_arg_.right_shift_[0], row, col, conv_param->output_channel_);
#else
  MatMulInt8_16x4_r(packed_input, packed_weight, dst, row, col, deep16, conv_param->output_channel_, input_sum, bias,
                    conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_,
                    conv_param->conv_quant_arg_.quant_multiplier_,
                    conv_param->conv_quant_arg_.output_quant_args_[0].zp_, conv_param->conv_quant_arg_.out_act_min_[0],
                    conv_param->conv_quant_arg_.out_act_max_[0], false);
#endif
  return;
}

// int8 convolution 3x3
void Conv3x3Int8(int16_t *input_data, int16_t *transed_weight, const int32_t *bias_data, int8_t *output_data,
                 int16_t *tile_buffer, int16_t *block_unit_buffer, int32_t *tmp_dst_buffer, int8_t *tmp_out,
                 int task_id, ConvParameter *conv_param) {
  int thread_count = conv_param->thread_num_;
  int ic8 = UP_DIV(conv_param->input_channel_, C8NUM);
  int output_channel = conv_param->output_channel_;
  int out_w_block = UP_DIV(conv_param->output_w_, OUPUT_UNIT);
  int out_h_block = UP_DIV(conv_param->output_h_, OUPUT_UNIT);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int oc4 = UP_DIV(output_channel, C4NUM);
  int tile_buffer_offset = TILE_NUM * 16 * ic8 * C8NUM;
  const int block_unit_buffer_offset = 16 * C8NUM;
  int tmp_dst_buffer_offset = TILE_NUM * 16 * oc4 * C4NUM;

  int input_batch = conv_param->input_batch_;
  for (int batch = 0; batch < input_batch; batch++) {
    int in_batch_offset = batch * ic8 * C8NUM * conv_param->input_h_ * conv_param->input_w_;
    int tmp_out_batch_offset = batch * oc4 * C4NUM * conv_param->output_w_ * conv_param->output_h_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * TILE_NUM;
      int real_cal_num = (output_count - start_index) < TILE_NUM ? (output_count - start_index) : TILE_NUM;

      Conv3x3Int8InputTransform(input_data + in_batch_offset, tile_buffer + task_id * tile_buffer_offset,
                                block_unit_buffer + task_id * block_unit_buffer_offset, start_index, real_cal_num,
                                out_w_block, conv_param);

      Conv3x3Int8Gemm(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tile_buffer + task_id * tile_buffer_offset,
                      transed_weight, output_channel, ic8, real_cal_num);

      Conv3x3Int8OutputTransform(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tmp_out + tmp_out_batch_offset,
                                 bias_data, start_index, real_cal_num, out_w_block, conv_param);
    }
  }
}
