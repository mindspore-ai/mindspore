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
#include "nnacl/int8/common_func_int8.h"

void Conv3x3Int8Gemm(int32_t *dst, const int16_t *src, const int16_t *weight, int oc, int ic8, size_t real_cal_num) {
  int oc4 = UP_DIV(oc, C4NUM);
#ifdef ENABLE_ARM
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

void ConvInt8(int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int8_t *packed_weight,
              const int32_t *bias_data, int8_t *output_data, int32_t *filter_zp, int32_t *input_sum, int task_id,
              ConvParameter *conv_param, MATMUL_OPT_R_FUNC matmul_func, bool is_optimize) {
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int tile_n = conv_param->tile_num_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_n);
  int kernel_plane = conv_param->kernel_h_ * conv_param->kernel_w_;
  int unit_size;
  int input_sum_offset;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(out_channel, C2NUM);
  unit_size = UP_ROUND(kernel_plane * in_channel, C16NUM);
#else
  if (is_optimize) {
    up_round_oc = UP_ROUND(out_channel, C8NUM);
    unit_size = UP_ROUND(kernel_plane * in_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(out_channel, C4NUM);
    unit_size = UP_ROUND(kernel_plane * in_channel, C16NUM);
  }
#endif
  bool per_channel = false;
  if (conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_offset = tile_n * up_round_oc;
    per_channel = true;
  } else {
    input_sum_offset = tile_n;
    per_channel = false;
  }

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * out_channel * conv_param->output_h_ * conv_param->output_w_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - start_index) : tile_n;
      int32_t *tmp_input_sum = input_sum + task_id * input_sum_offset;
      int8_t *gemm_input = packed_input + task_id * unit_size * tile_n;
      int8_t *matmul = matmul_input + task_id * kernel_plane * in_channel * tile_n;
      memset(matmul, conv_param->conv_quant_arg_.input_quant_args_[0].zp_, kernel_plane * in_channel * tile_n);
      Im2ColPackUnitInt8Opt(input_data + in_batch_offset, gemm_input, matmul, real_cal_num, start_index, filter_zp,
                            tmp_input_sum, conv_param, per_channel, is_optimize);

      int out_offset = thread_id * tile_n * out_channel + out_batch_offset;
      int8_t *gemm_output = output_data + out_offset;
#ifdef ENABLE_ARM32
      MatmulInt8Neon32(
        gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, tmp_input_sum, bias_data,
        conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
        conv_param->conv_quant_arg_.output_quant_args_[0].zp_, conv_param->conv_quant_arg_.quant_multiplier_,
        conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_, out_channel, per_channel);
#elif ENABLE_ARM64
      if (is_optimize) {
        matmul_func(gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, out_channel,
                    tmp_input_sum, bias_data, conv_param->conv_quant_arg_.left_shift_,
                    conv_param->conv_quant_arg_.right_shift_, conv_param->conv_quant_arg_.quant_multiplier_,
                    conv_param->conv_quant_arg_.output_quant_args_[0].zp_, conv_param->conv_quant_arg_.out_act_min_[0],
                    conv_param->conv_quant_arg_.out_act_max_[0], per_channel);
      } else {
        MatmulInt8Neon64(gemm_input, packed_weight, gemm_output, UP_ROUND(real_cal_num, C4NUM),
                         UP_ROUND(out_channel, C4NUM), unit_size, tmp_input_sum, bias_data,
                         conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
                         conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                         conv_param->conv_quant_arg_.quant_multiplier_, conv_param->conv_quant_arg_.left_shift_,
                         conv_param->conv_quant_arg_.right_shift_, real_cal_num, out_channel, out_channel, per_channel);
      }
#else
      MatMulInt8_8x8_r(
        gemm_input, packed_weight, gemm_output, real_cal_num, out_channel, unit_size, out_channel, tmp_input_sum,
        bias_data, conv_param->conv_quant_arg_.left_shift_, conv_param->conv_quant_arg_.right_shift_,
        conv_param->conv_quant_arg_.quant_multiplier_, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
        conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0], per_channel);
#endif
    }
  }
}

void Conv1x1PreOptPeroc(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                        size_t output_channel, size_t plane_size, int32_t *filter_zp, size_t inputsum_stride) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  int oc8 = UP_ROUND(output_channel, C8NUM);
  int hw8 = UP_ROUND(plane_size, C8NUM);
  size_t hw_8div = plane_size / C8NUM * C8NUM;
  size_t oc_8div = output_channel / C8NUM * C8NUM;
  size_t oc_8res = output_channel - oc_8div;
  size_t ic_4div = input_channel / C4NUM * C4NUM;

  const int8_t *src_r = src_input;
  int8_t *pack_r = packed_input;
  int32_t *input_sum_r = input_sum;

  for (int hwi = 0; hwi < hw_8div; hwi += C8NUM) {
    const int8_t *src_ic = src_r;
    int8_t *pack_ic = pack_r;
    int32_t *input_sum_oc = input_sum_r;
#ifdef ENABLE_ARM64
    size_t src_stride = input_channel;
    size_t ic_4res = input_channel - ic_4div;
    size_t input_sum_stride = inputsum_stride * 4 - C8NUM * C8NUM * 4;
    asm volatile(
      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"

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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "6: \n"
      "dup v0.4s, v16.s[0]  \n"
      "dup v1.4s, v16.s[1]  \n"
      "dup v2.4s, v16.s[2]  \n"
      "dup v3.4s, v16.s[3]  \n"
      "dup v4.4s, v17.s[0]  \n"
      "dup v5.4s, v17.s[1]  \n"
      "dup v6.4s, v17.s[2]  \n"
      "dup v7.4s, v17.s[3]  \n"
      "mov x4, #0 \n"
      "mov x10, %[filter_zp] \n"
      "mov x11, %[input_sum_oc] \n"

      "7: \n"
      "cmp x4, %[oc_8div] \n"
      "beq 8f \n"
      "add x4, x4, #8\n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.4s}, [x10], #16\n"

      "mul v18.4s, v16.4s, v0.4s \n"
      "mul v19.4s, v17.4s, v0.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"

      "mul v20.4s, v16.4s, v1.4s \n"
      "mul v21.4s, v17.4s, v1.4s \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"

      "mul v22.4s, v16.4s, v2.4s \n"
      "mul v23.4s, v17.4s, v2.4s \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"

      "mul v24.4s, v16.4s, v3.4s \n"
      "mul v25.4s, v17.4s, v3.4s \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "mul v18.4s, v16.4s, v4.4s \n"
      "mul v19.4s, v17.4s, v4.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"

      "mul v20.4s, v16.4s, v5.4s \n"
      "mul v21.4s, v17.4s, v5.4s \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"

      "mul v22.4s, v16.4s, v6.4s \n"
      "mul v23.4s, v17.4s, v6.4s \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"

      "mul v24.4s, v16.4s, v7.4s \n"
      "mul v25.4s, v17.4s, v7.4s \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "add x11, x11, %[input_sum_stride] \n"
      "b 7b \n"

      "8: \n"
      "cmp %[oc_8res], #0\n"
      "beq 17f \n"

      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"
      "cmp %[oc_8res], #1\n"
      "beq 9f \n"
      "cmp %[oc_8res], #2\n"
      "beq 10f \n"
      "cmp %[oc_8res], #3\n"
      "beq 11f \n"
      "cmp %[oc_8res], #4\n"
      "beq 12f \n"
      "cmp %[oc_8res], #5\n"
      "beq 13f \n"
      "cmp %[oc_8res], #6\n"
      "beq 14f \n"
      "cmp %[oc_8res], #7\n"
      "beq 15f \n"

      "9: \n"
      "ld1 {v16.s}[0], [x10] \n"
      "b 16f \n"

      "10: \n"
      "ld1 {v16.d}[0], [x10] \n"
      "b 16f \n"

      "11: \n"
      "ld1 {v16.d}[0], [x10] \n"
      "add x10, x10, #8 \n"
      "ld1 {v16.s}[2], [x10] \n"
      "b 16f \n"

      "12: \n"
      "ld1 {v16.4s}, [x10] \n"
      "b 16f \n"

      "13: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.s}[0], [x10] \n"
      "b 16f \n"

      "14: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.d}[0], [x10] \n"
      "b 16f \n"

      "15: \n"
      "ld1 {v16.4s}, [x10], #16\n"
      "ld1 {v17.d}[0], [x10] \n"
      "add x10, x10, #8 \n"
      "ld1 {v17.s}[2], [x10] \n"
      "b 16f \n"

      "16: \n"
      "mul v18.4s, v16.4s, v0.4s \n"
      "mul v19.4s, v17.4s, v0.4s \n"
      "mul v20.4s, v16.4s, v1.4s \n"
      "mul v21.4s, v17.4s, v1.4s \n"
      "mul v22.4s, v16.4s, v2.4s \n"
      "mul v23.4s, v17.4s, v2.4s \n"
      "mul v24.4s, v16.4s, v3.4s \n"
      "mul v25.4s, v17.4s, v3.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "mul v18.4s, v16.4s, v4.4s \n"
      "mul v19.4s, v17.4s, v4.4s \n"
      "mul v20.4s, v16.4s, v5.4s \n"
      "mul v21.4s, v17.4s, v5.4s \n"
      "mul v22.4s, v16.4s, v6.4s \n"
      "mul v23.4s, v17.4s, v6.4s \n"
      "mul v24.4s, v16.4s, v7.4s \n"
      "mul v25.4s, v17.4s, v7.4s \n"
      "st1 {v18.4s}, [x11], #16 \n"
      "st1 {v19.4s}, [x11], #16 \n"
      "st1 {v20.4s}, [x11], #16 \n"
      "st1 {v21.4s}, [x11], #16 \n"
      "st1 {v22.4s}, [x11], #16 \n"
      "st1 {v23.4s}, [x11], #16 \n"
      "st1 {v24.4s}, [x11], #16 \n"
      "st1 {v25.4s}, [x11], #16 \n"

      "17: \n"

      :
      : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ filter_zp ] "r"(filter_zp),
        [ input_sum_oc ] "r"(input_sum_oc), [ input_sum_stride ] "r"(input_sum_stride), [ src_stride ] "r"(src_stride),
        [ ic_4div ] "r"(ic_4div), [ ic_4res ] "r"(ic_4res), [ oc_8div ] "r"(oc_8div), [ oc_8res ] "r"(oc_8res)
      : "x0", "x1", "x4", "x9", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
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

    for (int ici = input_channel; ici < ic4; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        pack_ic[i * C4NUM] = 0;
      }
      pack_ic += 1;
    }

    for (int oci = 0; oci < oc_8div; oci += C8NUM) {
      for (int ri = 0; ri < C8NUM; ri++) {
        input_sum_oc[ri * C8NUM + 0] = tmp_sum_value[ri] * filter_zp[oci + 0];
        input_sum_oc[ri * C8NUM + 1] = tmp_sum_value[ri] * filter_zp[oci + 1];
        input_sum_oc[ri * C8NUM + 2] = tmp_sum_value[ri] * filter_zp[oci + 2];
        input_sum_oc[ri * C8NUM + 3] = tmp_sum_value[ri] * filter_zp[oci + 3];
        input_sum_oc[ri * C8NUM + 4] = tmp_sum_value[ri] * filter_zp[oci + 4];
        input_sum_oc[ri * C8NUM + 5] = tmp_sum_value[ri] * filter_zp[oci + 5];
        input_sum_oc[ri * C8NUM + 6] = tmp_sum_value[ri] * filter_zp[oci + 6];
        input_sum_oc[ri * C8NUM + 7] = tmp_sum_value[ri] * filter_zp[oci + 7];
      }
      input_sum_oc += inputsum_stride;
    }
    if (oc_8div != output_channel) {
      for (int oci = 0; oci < oc_8res; oci += 1) {
        for (int ri = 0; ri < C8NUM; ri++) {
          input_sum_oc[ri * C8NUM + oci] = tmp_sum_value[ri] * filter_zp[oc_8div + oci];
        }
      }
      for (int oci = oc_8res; oci < C8NUM; oci += 1) {
        for (int ri = 0; ri < C8NUM; ri++) {
          input_sum_oc[ri * C8NUM + oci] = 0;
        }
      }
    } /* oc8 res done */
#endif
    src_r += input_channel * C8NUM;
    pack_r += ic4 * C8NUM;
    input_sum_r += C8NUM * C8NUM;
  }

  if (hw_8div != plane_size) {
    memset(pack_r, 0, C8NUM * ic4);
    for (int hwi = hw_8div; hwi < plane_size; hwi += 1) {
      int32_t *input_sum_oc = input_sum_r;
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

      for (int oci = 0; oci < oc_8div; oci += C8NUM) {
        for (int curoi = 0; curoi < C8NUM; curoi++) {
          input_sum_oc[curoi] = tmp_sum_value * filter_zp[oci + curoi];
        }
        input_sum_oc += inputsum_stride;
      }
      if (oc_8div != output_channel) {
        for (int oci = 0; oci < oc_8res; oci += 1) {
          input_sum_oc[oci] = tmp_sum_value * filter_zp[oc_8div + oci];
        }
        for (int oci = oc_8res; oci < C8NUM; oci += 1) {
          input_sum_oc[oci] = 0;
        }
      } /* oc8 res done */

      src_r += input_channel;
      pack_r += C4NUM;
      input_sum_r += C8NUM;
    }

    for (int hwi = plane_size; hwi < hw8; hwi++) {
      for (int oc = 0; oc < oc8; oc++) {
        int oc8div = oc / C8NUM, oc8res = oc % C8NUM;
        input_sum[oc8div * inputsum_stride + hwi * C8NUM + oc8res] = 0;
      }
    }
  }
  return;
}

void Conv1x1PreOptPert(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                       size_t plane_size, ConvParameter *conv_param) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  size_t hw_8div = plane_size / C8NUM * C8NUM;
  size_t ic_4div = input_channel / C4NUM * C4NUM;
  int32_t filter_zp = conv_param->conv_quant_arg_.filter_quant_args_[0].zp_;

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
      "dup v16.4s, wzr \n"
      "dup v17.4s, wzr \n"
      "mov x14, %[input_sum_r] \n"
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

      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
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
      "add v16.4s, v16.4s, v0.4s \n"
      "add v17.4s, v17.4s, v1.4s \n"
      "b 6f \n"

      "6: \n"
      "mul v16.4s, v16.4s, v20.4s \n"
      "mul v17.4s, v17.4s, v20.4s \n"

      "st1 {v16.4s}, [x14], #16 \n"
      "st1 {v17.4s}, [x14], #16 \n"

      :
      : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ input_sum_r ] "r"(input_sum_r),
        [ src_stride ] "r"(src_stride), [ ic_4div ] "r"(ic_4div), [ ic_4res ] "r"(ic_4res), [ filter_zp ] "r"(filter_zp)
      : "x0", "x1", "x10", "x11", "x12", "x13", "x14", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
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

    for (int ici = input_channel; ici < ic4; ici += 1) {
      for (int i = 0; i < C8NUM; i++) {
        pack_ic[i * C4NUM] = 0;
      }
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
    for (int hwi = plane_size; hwi < UP_ROUND(plane_size, C8NUM); hwi++) {
      input_sum[hwi] = 0;
    }
  }
  return;
}

void Conv1x1Int8Opt(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                    const int32_t *bias, int row, int col, int deep4, int32_t *left_shift, int32_t *right_shift,
                    int32_t *multiplier, ConvParameter *conv_param, MATMUL_OPT_DP_FUNC matmul_func, int *filter_zp) {
  int is_per_oc = (int)conv_param->conv_quant_arg_.filter_arg_num_ != 1;
  matmul_func(packed_input, packed_weight, dst, row, col, deep4, conv_param->output_channel_, input_sum, bias,
              left_shift, right_shift, multiplier, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
              conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0], is_per_oc,
              filter_zp);
  return;
}

void Conv1x1Int8(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                 const int32_t *bias, int row, int col, int deep16, int32_t *left_shift, int32_t *right_shift,
                 int32_t *multiplier, ConvParameter *conv_param, int32_t *filter_zp) {
  int is_per_oc = (int)conv_param->conv_quant_arg_.filter_arg_num_ != 1;
  MatmulInt8Opt(packed_input, packed_weight, dst, row, col, deep16, input_sum, bias,
                conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
                conv_param->conv_quant_arg_.output_quant_args_[0].zp_, multiplier, left_shift, right_shift,
                conv_param->output_channel_, is_per_oc, filter_zp);
  return;
}

// int8 convolution 3x3
void Conv3x3Int8(int16_t *input_data, int16_t *transed_weight, const int32_t *bias_data, int8_t *output_data,
                 int16_t *tile_buffer, int16_t *block_unit_buffer, int32_t *tmp_dst_buffer, int8_t *tmp_out,
                 int task_id, ConvParameter *conv_param) {
  int ic8 = UP_DIV(conv_param->input_channel_, C8NUM);
  int out_w_block = UP_DIV(conv_param->output_w_, OUPUT_UNIT);
  int out_h_block = UP_DIV(conv_param->output_h_, OUPUT_UNIT);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int oc4 = UP_DIV(conv_param->output_channel_, C4NUM);
  int tile_buffer_offset = TILE_NUM * 16 * ic8 * C8NUM;
  const int block_unit_buffer_offset = 16 * C8NUM;
  int tmp_dst_buffer_offset = TILE_NUM * 16 * oc4 * C4NUM;

  for (int batch = 0; batch < conv_param->input_batch_; batch++) {
    int in_batch_offset = batch * ic8 * C8NUM * conv_param->input_h_ * conv_param->input_w_;
    int tmp_out_batch_offset = batch * oc4 * C4NUM * conv_param->output_w_ * conv_param->output_h_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * TILE_NUM;
      int real_cal_num = (output_count - start_index) < TILE_NUM ? (output_count - start_index) : TILE_NUM;

      Conv3x3Int8InputTransform(input_data + in_batch_offset, tile_buffer + task_id * tile_buffer_offset,
                                block_unit_buffer + task_id * block_unit_buffer_offset, start_index, real_cal_num,
                                out_w_block, conv_param);

      Conv3x3Int8Gemm(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tile_buffer + task_id * tile_buffer_offset,
                      transed_weight, conv_param->output_channel_, ic8, real_cal_num);

      Conv3x3Int8OutputTransform(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tmp_out + tmp_out_batch_offset,
                                 bias_data, start_index, real_cal_num, out_w_block, conv_param);
    }
  }
}
