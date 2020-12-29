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
