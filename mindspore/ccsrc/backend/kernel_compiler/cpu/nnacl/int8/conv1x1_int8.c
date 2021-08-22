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

#include "nnacl/int8/conv1x1_int8.h"

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
