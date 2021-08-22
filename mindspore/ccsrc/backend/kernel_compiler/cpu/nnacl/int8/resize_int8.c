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
#include <math.h>
#include "nnacl/int8/resize_int8.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/fixed_point.h"
#include "nnacl/errorcode.h"

int ResizeBilinearInt8(const int8_t *input_ptr, int8_t *output_ptr, int batch, int in_h, int in_w, int out_h, int out_w,
                       int channel, int index, int count, ResizeQuantArg quant_arg) {
  if (out_w == 0) {
    return NNACL_ERRCODE_DIVISOR_ZERO;
  }
  int in_plane = in_h * in_w;
  int out_plane = out_h * out_w;
  for (int n = 0; n < batch; n++) {
    const int8_t *in_b_ptr = input_ptr + n * in_plane * channel;
    int8_t *out_b_ptr = output_ptr + n * out_plane * channel;
    for (int t = 0; t < count; t++) {
      int ori_out_h = (index + t) / out_w;
      int ori_out_w = (index + t) % out_w;
      int32_t x_lower_value = quant_arg.x_axis_lower_[ori_out_w];
      int32_t x_upper_value = quant_arg.x_axis_upper_[ori_out_w];
      int32_t y_lower_value = quant_arg.y_axis_lower_[ori_out_h];
      int32_t y_upper_value = quant_arg.y_axis_upper_[ori_out_h];
      int32_t weight_x = quant_arg.x_axis_index_[ori_out_w] - (1 << 10) * x_lower_value;
      int32_t one_minus_weight_x = (1 << 10) - weight_x;
      int32_t weight_y = quant_arg.y_axis_index_[ori_out_h] - (1 << 10) * y_lower_value;
      int32_t one_minus_weight_y = (1 << 10) - weight_y;
      int64_t left_bottom_coef = (int64_t)(one_minus_weight_x * one_minus_weight_y);
      int64_t left_top_coef = (int64_t)(weight_y * one_minus_weight_x);
      int64_t right_bottom_coef = (int64_t)(weight_x * one_minus_weight_y);
      int64_t right_top_coef = (int64_t)(weight_x * weight_y);
      int input_lb_index = (y_lower_value * in_w + x_lower_value) * channel;
      int input_lt_index = (y_upper_value * in_w + x_lower_value) * channel;
      int input_rb_index = (y_lower_value * in_w + x_upper_value) * channel;
      int input_rt_index = (y_upper_value * in_w + x_upper_value) * channel;
      int c = 0;
      for (; c < channel; c++) {
        int64_t out_left_bottom = left_bottom_coef * in_b_ptr[input_lb_index];
        int64_t out_left_top = left_top_coef * in_b_ptr[input_lt_index];
        int64_t out_right_bottom = right_bottom_coef * in_b_ptr[input_rb_index];
        int64_t out_right_top = right_top_coef * in_b_ptr[input_rt_index];
        int64_t out_value = out_left_bottom + out_left_top + out_right_bottom + out_right_top;
        out_b_ptr[0] = (int8_t)((out_value + (1 << 19)) / (1 << 20));
        input_lb_index++;
        input_lt_index++;
        input_rb_index++;
        input_rt_index++;
        out_b_ptr++;
      }
    }
  }
  return NNACL_OK;
}

int ResizeBilinearWithFloatScaleInt8(const int8_t *input_ptr, int8_t *output_ptr, int batch, int in_h, int in_w,
                                     int out_h, int out_w, int channel, int index, int count,
                                     ResizeFloatScaleQuantArg quant_arg) {
  if (out_w == 0) {
    return NNACL_ERRCODE_DIVISOR_ZERO;
  }
  int in_plane = in_h * in_w;
  int out_plane = out_h * out_w;
  for (int n = 0; n < batch; n++) {
    const int8_t *in_b_ptr = input_ptr + n * in_plane * channel;
    int8_t *out_b_ptr = output_ptr + n * out_plane * channel;
    for (int t = 0; t < count; t++) {
      int ori_out_h = (index + t) / out_w;
      int ori_out_w = (index + t) % out_w;
      int32_t x_lower_value = quant_arg.x_axis_lower_[ori_out_w];
      int32_t x_upper_value = quant_arg.x_axis_upper_[ori_out_w];
      int32_t y_lower_value = quant_arg.y_axis_lower_[ori_out_h];
      int32_t y_upper_value = quant_arg.y_axis_upper_[ori_out_h];
      float weight_x = quant_arg.x_axis_index_[ori_out_w] - x_lower_value;
      const float one_minus_weight_x = 1 - weight_x;
      float weight_y = quant_arg.y_axis_index_[ori_out_h] - y_lower_value;
      const float one_minus_weight_y = 1 - weight_y;
      float left_bottom_coef = one_minus_weight_x * one_minus_weight_y;
      float left_top_coef = weight_y * one_minus_weight_x;
      float right_bottom_coef = weight_x * one_minus_weight_y;
      float right_top_coef = weight_x * weight_y;
      int input_lb_index = (y_lower_value * in_w + x_lower_value) * channel;
      int input_lt_index = (y_upper_value * in_w + x_lower_value) * channel;
      int input_rb_index = (y_lower_value * in_w + x_upper_value) * channel;
      int input_rt_index = (y_upper_value * in_w + x_upper_value) * channel;
      int c = 0;
#ifdef ENABLE_ARM
      for (; c <= channel - 4; c += 4) {
        float32x4_t in_lb;
        in_lb[0] = (float)in_b_ptr[input_lb_index];
        in_lb[1] = (float)in_b_ptr[input_lb_index + 1];
        in_lb[2] = (float)in_b_ptr[input_lb_index + 2];
        in_lb[3] = (float)in_b_ptr[input_lb_index + 3];
        float32x4_t out_left_bottom = vmulq_n_f32(in_lb, left_bottom_coef);
        float32x4_t in_lt;
        in_lt[0] = (float)in_b_ptr[input_lt_index];
        in_lt[1] = (float)in_b_ptr[input_lt_index + 1];
        in_lt[2] = (float)in_b_ptr[input_lt_index + 2];
        in_lt[3] = (float)in_b_ptr[input_lt_index + 3];
        float32x4_t out_left_top = vmulq_n_f32(in_lt, left_top_coef);
        float32x4_t in_rb;
        in_rb[0] = (float)in_b_ptr[input_rb_index];
        in_rb[1] = (float)in_b_ptr[input_rb_index + 1];
        in_rb[2] = (float)in_b_ptr[input_rb_index + 2];
        in_rb[3] = (float)in_b_ptr[input_rb_index + 3];
        float32x4_t out_right_bottom = vmulq_n_f32(in_rb, right_bottom_coef);
        float32x4_t in_rt;
        in_rt[0] = (float)in_b_ptr[input_rt_index];
        in_rt[1] = (float)in_b_ptr[input_rt_index + 1];
        in_rt[2] = (float)in_b_ptr[input_rt_index + 2];
        in_rt[3] = (float)in_b_ptr[input_rt_index + 3];
        float32x4_t out_right_top = vmulq_n_f32(in_rt, right_top_coef);
        float32x4_t out_value1 = vaddq_f32(out_left_bottom, out_left_top);
        float32x4_t out_value2 = vaddq_f32(out_right_top, out_right_bottom);
        float32x4_t out_value = vaddq_f32(out_value1, out_value2);
        out_b_ptr[0] = (int8_t)(out_value[0]);
        out_b_ptr[1] = (int8_t)(out_value[1]);
        out_b_ptr[2] = (int8_t)(out_value[2]);
        out_b_ptr[3] = (int8_t)(out_value[3]);
        input_lb_index += 4;
        input_lt_index += 4;
        input_rb_index += 4;
        input_rt_index += 4;
        out_b_ptr += 4;
      }
#endif
      for (; c < channel; c++) {
        float out_left_bottom = left_bottom_coef * in_b_ptr[input_lb_index];
        float out_left_top = left_top_coef * in_b_ptr[input_lt_index];
        float out_right_bottom = right_bottom_coef * in_b_ptr[input_rb_index];
        float out_right_top = right_top_coef * in_b_ptr[input_rt_index];
        float out_value = out_left_bottom + out_left_top + out_right_bottom + out_right_top;
        out_b_ptr[0] = (int8_t)(out_value);
        input_lb_index++;
        input_lt_index++;
        input_rb_index++;
        input_rt_index++;
        out_b_ptr++;
      }
    }
  }
  return NNACL_OK;
}

int ResizeNearestNeighborInt8Simple(const int8_t *input_data, int8_t *output_data, const int *input_shape,
                                    const int *output_shape, const bool align_corners, int tid, int thread_num) {
  int batch, y, x, c;
  c = output_shape[3];
  int in_h, in_w, new_height, new_width;
  in_h = input_shape[1];
  in_w = input_shape[2];
  new_height = output_shape[1];
  new_width = output_shape[2];

  for (batch = 0; batch < output_shape[0]; batch++) {
    for (y = tid; y < output_shape[1]; y += thread_num) {
      int input_y = 0;
      ComputeNearestNeighborInt(y, in_h, new_height, align_corners, &input_y);
      for (x = 0; x < output_shape[2]; x++) {
        int input_x = 0;
        ComputeNearestNeighborInt(x, in_w, new_width, align_corners, &input_x);
        int in_offset = offset(input_shape, batch, input_y, input_x, 0);
        int out_offset = offset(output_shape, batch, y, x, 0);
        memcpy(output_data + out_offset, input_data + in_offset, c * sizeof(int8_t));
      }
    }
  }

  return NNACL_OK;
}

void ComputeNearestNeighborInt(const int32_t pos, const int in_size, const int32_t new_size, const bool align_corners,
                               int32_t *nearest) {
  if (new_size == 0) {
    return;
  }
  *nearest = (in_size * pos) / new_size;
  if (align_corners && new_size != 1) {
    *nearest = ((in_size - 1) * pos + (new_size - 1) / 2) / (new_size - 1);
  }
  *nearest = *nearest < in_size ? *nearest : in_size - 1;
}

int ResizeNearestNeighborInt8(const int8_t *input_data, int8_t *output_data, const int *input_shape,
                              const int *output_shape, const bool align_corners, const QuantMulArg *multiplier,
                              QuantArg *quant_in, QuantArg *quant_out, int tid, int thread_num) {
  const int base_offset = 20;
  int32_t batch, y, x, c;
  int32_t in_h, in_w, new_height, new_width;
  in_h = input_shape[1];
  in_w = input_shape[2];
  new_height = output_shape[1];
  new_width = output_shape[2];

  for (batch = 0; batch < output_shape[0]; batch++) {
    for (y = tid; y < output_shape[1]; y += thread_num) {
      int input_y = 0;
      ComputeNearestNeighborInt(y, in_h, new_height, align_corners, &input_y);
      for (x = 0; x < output_shape[2]; x++) {
        int input_x = 0;
        ComputeNearestNeighborInt(x, in_w, new_width, align_corners, &input_x);
        for (c = 0; c < output_shape[3]; c++) {
          int in_offset = offset(input_shape, batch, input_y, input_x, c);
          int out_offset = offset(output_shape, batch, y, x, c);

          int32_t out_value = MultiplyByQuantizedMultiplier(
                                input_data[in_offset] - quant_in->zp_, multiplier->multiplier_,
                                multiplier->left_shift_ + base_offset, multiplier->right_shift_ - base_offset) +
                              quant_out->zp_;
          out_value = out_value > INT8_MAX ? INT8_MAX : out_value;
          out_value = out_value < INT8_MIN ? INT8_MIN : out_value;
          output_data[out_offset] = (int8_t)out_value;
        }
      }
    }
  }

  return NNACL_OK;
}
