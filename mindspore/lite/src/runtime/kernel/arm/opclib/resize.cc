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
#include "src/runtime/kernel/arm/opclib/resize.h"
#include "src/runtime/kernel/arm/opclib/common_func.h"

int ResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                   bool align_corners, int tid, int thread_num) {
  if (input_data == nullptr || output_data == nullptr || input_shape == nullptr || output_shape == nullptr) {
    return OPCLIB_NULL_PTR;
  }
  // nhwc (memory layout is nc4hw4)
  int n = input_shape[0];
  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int channel = input_shape[3];
  int c4 = UP_DIV(channel, C4NUM);

  int new_height = output_shape[1];
  int new_width = output_shape[2];
  float height_scale = (float)(in_h) / new_height;
  float width_scale = (float)(in_w) / new_width;
  if (align_corners && new_height > 1) {
    height_scale = (float)(in_h - 1) / (new_height - 1);
  }
  if (align_corners && new_width > 1) {
    width_scale = (float)(in_w - 1) / (new_width - 1);
  }

  int o[5];  // n c4 h w 4
  for (o[0] = 0; o[0] < n; o[0]++) {
    for (o[1] = tid; o[1] < c4; o[1] += thread_num) {
      for (o[2] = 0; o[2] < new_height; o[2]++) {
        float actual_y = (float)(o[2]) * height_scale;
        int y_left = (int)(floor(actual_y));
        int y_right = y_left + 1 < in_h ? (y_left + 1) : (in_h - 1);
        float y_right_weight = actual_y - (float)(y_left);
        float y_left_weight = 1.0 - y_right_weight;
        for (o[3] = 0; o[3] < new_width; o[3]++) {
          float actual_x = (float)(o[3]) * width_scale;
          int x_left = (int)(floor(actual_x));
          int x_right = x_left + 1 < in_w ? (x_left + 1) : (in_w - 1);
          float x_right_weight = actual_x - (float)(x_left);
          float x_left_weight = 1.0 - x_right_weight;

          auto input_base_offset = (((o[0] * c4 + o[1]) * in_h + y_left) * in_w + x_left) * C4NUM;
          auto output_base_offset = (((o[0] * c4 + o[1]) * new_height + o[2]) * new_width + o[3]) * C4NUM;
          int in_offset_1_0 = (y_right - y_left) * in_w * C4NUM;
          int in_offset_0_1 = (x_right - x_left) * C4NUM;
#ifdef ENABLE_NEON
          float32x4_t x_l_weight = vdupq_n_f32(x_left_weight);
          float32x4_t x_r_weight = vdupq_n_f32(x_right_weight);
          float32x4_t y_l_weight = vdupq_n_f32(y_left_weight);
          float32x4_t y_r_weight = vdupq_n_f32(y_right_weight);

          float32x4_t input_yl_xl = vld1q_f32(input_data + input_base_offset);
          float32x4_t input_yr_xl = vld1q_f32(input_data + input_base_offset + in_offset_1_0);
          float32x4_t input_yl_xr = vld1q_f32(input_data + input_base_offset + in_offset_0_1);
          float32x4_t input_yr_xr = vld1q_f32(input_data + input_base_offset + in_offset_0_1 + in_offset_1_0);

          float32x4_t interp_value = vdupq_n_f32(0.0);
          float32x4_t interp_value_tmp = vmulq_f32(input_yl_xl, y_l_weight);
          interp_value_tmp = vmulq_f32(interp_value_tmp, x_l_weight);
          interp_value = vaddq_f32(interp_value, interp_value_tmp);

          interp_value_tmp = vmulq_f32(input_yr_xl, y_r_weight);
          interp_value_tmp = vmulq_f32(interp_value_tmp, x_l_weight);
          interp_value = vaddq_f32(interp_value, interp_value_tmp);

          interp_value_tmp = vmulq_f32(input_yl_xr, y_l_weight);
          interp_value_tmp = vmulq_f32(interp_value_tmp, x_r_weight);
          interp_value = vaddq_f32(interp_value, interp_value_tmp);

          interp_value_tmp = vmulq_f32(input_yr_xr, y_r_weight);
          interp_value_tmp = vmulq_f32(interp_value_tmp, x_r_weight);
          interp_value = vaddq_f32(interp_value, interp_value_tmp);
          vst1q_f32(output_base_offset + output_data, interp_value);
#else
          // 4 continuous data in a group;
          for (o[4] = 0; o[4] < C4NUM; o[4]++) {
            auto in_offset = input_base_offset + o[4];
            auto output_offset = output_base_offset + o[4];
            float interp_value =
              input_data[in_offset] * y_left_weight * x_left_weight +
              input_data[in_offset + in_offset_1_0] * y_right_weight * x_left_weight +
              input_data[in_offset + in_offset_0_1] * y_left_weight * x_right_weight +
              input_data[in_offset + in_offset_0_1 + in_offset_1_0] * y_right_weight * x_right_weight;
            output_data[output_offset] = interp_value;
          }
#endif
        }
      }
    }
  }
  return OPCLIB_OK;
}

int ResizeNearestNeighbor(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                          int tid, int thread_num) {
  int batch, y, x, c;
  c = input_shape[3];

  float height_scale = (float)(input_shape[1]) / (float)(output_shape[1]);
  float width_scale = (float)(input_shape[2]) / (float)(output_shape[2]);

  for (batch = 0; batch < output_shape[0]; batch++) {
    for (y = tid; y < output_shape[1]; y += thread_num) {
      int actual_y = (int)(floor((float)(y) * height_scale));
      int input_y = actual_y < input_shape[1] ? actual_y : input_shape[1] - 1;
      for (x = 0; x < output_shape[2]; x++) {
        int actual_x = (int)(floor((float)(x) * width_scale));
        int input_x = actual_x < input_shape[2] ? actual_x : input_shape[2] - 1;
        int in_offset = offset(input_shape, batch, input_y, input_x, 0);
        int out_offset = offset(output_shape, batch, y, x, 0);
        memcpy(output_data + out_offset, input_data + in_offset, c * sizeof(float));
      }
    }
  }

  return OPCLIB_OK;
}

