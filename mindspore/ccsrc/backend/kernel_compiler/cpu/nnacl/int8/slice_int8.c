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

#include "nnacl/int8/slice_int8.h"
#include "nnacl/errorcode.h"

int SliceInt8(const int8_t *input, int8_t *output, const SliceParameter *param, int thread_id) {
  double input_scale = param->quant_arg_.in_args_.scale_;
  int input_zp = param->quant_arg_.in_args_.zp_;
  double output_scale = param->quant_arg_.out_args_.scale_;
  int output_zp = param->quant_arg_.out_args_.zp_;
  const int base_offset = 20;
  int act_min = param->quant_arg_.output_activation_min_;
  int act_max = param->quant_arg_.output_activation_max_;

  size_t out_stride[8];
  out_stride[7] = 1;
  for (int i = 6; i >= 0; --i) {
    out_stride[i] = out_stride[i + 1] * param->size_[i + 1];
  }
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
  int count_per_thread = UP_DIV(param->size_[5], param->op_parameter_.thread_num_);
  size_t thread_begin = thread_id * count_per_thread;
  size_t thread_end = MSMIN(param->size_[5], thread_begin + count_per_thread);
  int unit_size = param->size_[7] * sizeof(int8_t);
  size_t in_stride[8];
  in_stride[7] = 1;
  for (int i = 6; i >= 0; --i) {
    in_stride[i] = param->shape_[i + 1] * in_stride[i + 1];
  }
  int i, j, k, l, n, h, w, c;

  int equal_quant = 0;
  if (input_scale == output_scale && input_zp == output_zp) {
    equal_quant = 1;
  }

  for (i = 0; i < param->size_[0]; ++i) {
    size_t out_offset0 = i * out_stride[0];
    size_t in_offset0 = (i + param->begin_[0]) * in_stride[0] + param->begin_[7];
    for (j = 0; j < param->size_[1]; ++j) {
      size_t out_offset1 = j * out_stride[1] + out_offset0;
      size_t in_offset1 = (j + param->begin_[1]) * in_stride[1] + in_offset0;
      for (k = 0; k < param->size_[2]; ++k) {
        size_t out_offset2 = k * out_stride[2] + out_offset1;
        size_t in_offset2 = (k + param->begin_[2]) * in_stride[2] + in_offset1;
        for (l = 0; l < param->size_[3]; ++l) {
          size_t out_offset3 = l * out_stride[3] + out_offset2;
          size_t in_offset3 = (l + param->begin_[3]) * in_stride[3] + in_offset2;
          for (n = 0; n < param->size_[4]; ++n) {
            size_t out_offset4 = n * out_stride[4] + out_offset3;
            size_t in_offset4 = (n + param->begin_[4]) * in_stride[4] + in_offset3;
            for (h = thread_begin; h < thread_end; ++h) {
              size_t out_offset5 = h * out_stride[5] + out_offset4;
              size_t in_offset5 = (h + param->begin_[5]) * in_stride[5] + in_offset4;
              for (w = 0; w < param->size_[6]; ++w) {
                size_t out_offset = w * out_stride[6] + out_offset5;
                size_t in_offset = (w + param->begin_[6]) * in_stride[6] + in_offset5;
                if (equal_quant == 1) {
                  memcpy(output + out_offset, input + in_offset, unit_size);
                } else {
                  for (c = 0; c < param->size_[7]; ++c) {
                    int32_t output_val = MultiplyByQuantizedMultiplier(
                                           input[in_offset + c] - input_zp, param->quant_arg_.multiplier_.multiplier_,
                                           param->quant_arg_.multiplier_.left_shift_ + base_offset,
                                           param->quant_arg_.multiplier_.right_shift_ - base_offset) +
                                         output_zp;
                    output_val = MSMAX(INT8_MIN, MSMIN(output_val, INT8_MAX));
                    output[c + out_offset] = (int8_t)MSMAX(act_min, MSMIN(output_val, act_max));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return 0;
}
