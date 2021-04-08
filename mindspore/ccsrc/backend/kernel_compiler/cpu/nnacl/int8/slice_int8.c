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

int SliceInt8NoParallel(const int8_t *input, int8_t *output, SliceParameter *param) {
  double input_scale = param->quant_arg_.in_args_.scale_;
  int input_zp = param->quant_arg_.in_args_.zp_;
  double output_scale = param->quant_arg_.out_args_.scale_;
  int output_zp = param->quant_arg_.out_args_.zp_;
  int act_min = param->quant_arg_.output_activation_min_;
  int act_max = param->quant_arg_.output_activation_max_;

  int equal_quant = 0;
  double multiplier = input_scale / output_scale;
  if (input_scale == output_scale && input_zp == output_zp) {
    equal_quant = 1;
  }

  int32_t end_n = param->begin_[0] + param->size_[0];
  int32_t end_h = param->begin_[1] + param->size_[1];
  int32_t end_w = param->begin_[2] + param->size_[2];

  int unit_count = param->size_[3];
  int unit_size = unit_count * sizeof(int8_t);
  int in_stride2 = param->shape_[3];
  int in_stride1 = param->shape_[2] * in_stride2;
  int in_stride0 = param->shape_[1] * in_stride1;
  int out_offset = 0;
  int n, h, w, c;

  for (n = param->begin_[0]; n < end_n; ++n) {
    size_t in_offset0 = n * in_stride0;
    for (h = param->begin_[1]; h < end_h; ++h) {
      size_t in_offset1 = h * in_stride1 + in_offset0;
      for (w = param->begin_[2]; w < end_w; ++w) {
        size_t in_offset = in_offset1 + w * in_stride2;
        if (equal_quant == 1) {
          memcpy(output + out_offset, input + in_offset, unit_size);
        } else {
          for (c = 0; c < unit_count; ++c) {
            int32_t output_val = round(multiplier * (input[in_offset + c] - input_zp)) + output_zp;
            output[c + out_offset] = (int8_t)MSMAX(act_min, MSMIN(output_val, act_max));
          }
        }
        out_offset += unit_count;
      }
    }
  }
  return 0;
}

int SliceInt8(const int8_t *input, int8_t *output, SliceParameter *param, int thread_id) {
  double input_scale = param->quant_arg_.in_args_.scale_;
  int input_zp = param->quant_arg_.in_args_.zp_;
  double output_scale = param->quant_arg_.out_args_.scale_;
  int output_zp = param->quant_arg_.out_args_.zp_;
  int act_min = param->quant_arg_.output_activation_min_;
  int act_max = param->quant_arg_.output_activation_max_;

  int32_t out_dim1 = param->size_[1];
  int32_t out_dim2 = param->size_[2];
  int32_t out_dim3 = param->size_[3];
  int out_stride2 = out_dim3;
  int out_stride1 = out_stride2 * out_dim2;
  int out_stride0 = out_stride1 * out_dim1;
  int count_per_thread = UP_DIV(out_dim1, param->op_parameter_.thread_num_);
  int thread_stride = thread_id * count_per_thread;
  int unit_size = param->size_[3] * sizeof(int8_t);
  int in_stride2 = param->shape_[3];
  int in_stride1 = param->shape_[2] * in_stride2;
  int in_stride0 = param->shape_[1] * in_stride1;
  int n, h, w, c;

  int equal_quant = 0;
  double multiplier = input_scale / output_scale;
  if (input_scale == output_scale && input_zp == output_zp) {
    equal_quant = 1;
  }

  for (n = 0; n < param->size_[0]; ++n) {
    size_t out_offset0 = n * out_stride0;
    size_t in_offset0 = (n + param->begin_[0]) * in_stride0 + param->begin_[3];
    for (h = 0; h < count_per_thread; ++h) {
      size_t k = h + thread_stride;
      if (k >= out_dim1) {
        break;
      }
      size_t out_offset1 = k * out_stride1 + out_offset0;
      size_t in_offset1 = (k + param->begin_[1]) * in_stride1 + in_offset0;
      for (w = 0; w < out_dim2; ++w) {
        size_t out_offset = out_offset1 + w * out_stride2;
        size_t in_offset = in_offset1 + (w + param->begin_[2]) * in_stride2;
        if (equal_quant == 1) {
          memcpy(output + out_offset, input + in_offset, unit_size);
        } else {
          for (c = 0; c < out_dim3; ++c) {
            int32_t output_val = round(multiplier * (input[in_offset + c] - input_zp)) + output_zp;
            output[c + out_offset] = (int8_t)MSMAX(act_min, MSMIN(output_val, act_max));
          }
        }
      }
    }
  }
  return 0;
}
