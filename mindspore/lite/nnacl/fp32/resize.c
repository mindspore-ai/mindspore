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
#include "nnacl/fp32/resize.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

int ResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                   bool align_corners, int tid, int thread_num) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_n = input_shape[0];
  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int in_c = input_shape[3];

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

  int n, h, w, c;
  for (n = 0; n < in_n; n++) {
    for (h = tid; h < new_height; h += thread_num) {
      float actual_y = (float)h * height_scale;
      int y_bottom = (int)(floor(actual_y));
      int y_top = y_bottom + 1 < in_h ? (y_bottom + 1) : (in_h - 1);
      float y_top_weight = actual_y - (float)(y_bottom);
      const float y_bottom_weight = 1.0f - y_top_weight;
      for (w = 0; w < new_width; w++) {
        float actual_x = (float)(w)*width_scale;
        int x_left = (int)(floor(actual_x));
        int x_right = x_left + 1 < in_w ? (x_left + 1) : (in_w - 1);
        float x_right_weight = actual_x - (float)(x_left);
        const float x_left_weight = 1.0f - x_right_weight;
        c = 0;
#ifdef ENABLE_NEON
        for (; c <= in_c - 4; c += 4) {
          float32x4_t bottom_left = vld1q_f32(input_data + offset(input_shape, n, y_bottom, x_left, c));
          float32x4_t bottom_right = vld1q_f32(input_data + offset(input_shape, n, y_bottom, x_right, c));
          float32x4_t top_left = vld1q_f32(input_data + offset(input_shape, n, y_top, x_left, c));
          float32x4_t top_right = vld1q_f32(input_data + offset(input_shape, n, y_top, x_right, c));

          float32x4_t y_top_w = vdupq_n_f32(y_top_weight);
          float32x4_t y_bottom_w = vdupq_n_f32(y_bottom_weight);
          float32x4_t x_left_w = vdupq_n_f32(x_left_weight);
          float32x4_t x_right_w = vdupq_n_f32(x_right_weight);

          float32x4_t interp_value = vdupq_n_f32(0.0);
          float32x4_t tmp = vmulq_f32(bottom_left, y_bottom_w);
          tmp = vmulq_f32(tmp, x_left_w);
          interp_value = vaddq_f32(interp_value, tmp);

          tmp = vmulq_f32(bottom_right, y_bottom_w);
          tmp = vmulq_f32(tmp, x_right_w);
          interp_value = vaddq_f32(interp_value, tmp);

          tmp = vmulq_f32(top_left, y_top_w);
          tmp = vmulq_f32(tmp, x_left_w);
          interp_value = vaddq_f32(interp_value, tmp);

          tmp = vmulq_f32(top_right, y_top_w);
          tmp = vmulq_f32(tmp, x_right_w);
          interp_value = vaddq_f32(interp_value, tmp);
          vst1q_f32(output_data + offset(output_shape, n, h, w, c), interp_value);
        }
#endif
        for (; c < in_c; c++) {
          float bottom_left = input_data[offset(input_shape, n, y_bottom, x_left, c)];
          float bottom_right = input_data[offset(input_shape, n, y_bottom, x_right, c)];
          float top_left = input_data[offset(input_shape, n, y_top, x_left, c)];
          float top_right = input_data[offset(input_shape, n, y_top, x_right, c)];
          float interp_value = bottom_left * y_bottom_weight * x_left_weight +
                               bottom_right * y_bottom_weight * x_right_weight +
                               top_left * y_top_weight * x_left_weight + top_right * y_top_weight * x_right_weight;
          output_data[offset(output_shape, n, h, w, c)] = interp_value;
        }
      }
    }
  }
  return NNACL_OK;
}

int ResizeNearestNeighbor(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                          int tid, int thread_num) {
  int batch, y, x, c;
  c = input_shape[3];

  float height_scale = (float)(input_shape[1]) / (float)(output_shape[1]);
  float width_scale = (float)(input_shape[2]) / (float)(output_shape[2]);

  for (batch = 0; batch < output_shape[0]; batch++) {
    for (y = tid; y < output_shape[1]; y += thread_num) {
      int actual_y = (int)(floor((float)(y)*height_scale));
      int input_y = actual_y < input_shape[1] ? actual_y : input_shape[1] - 1;
      for (x = 0; x < output_shape[2]; x++) {
        int actual_x = (int)(floor((float)(x)*width_scale));
        int input_x = actual_x < input_shape[2] ? actual_x : input_shape[2] - 1;
        int in_offset = offset(input_shape, batch, input_y, input_x, 0);
        int out_offset = offset(output_shape, batch, y, x, 0);
        memcpy(output_data + out_offset, input_data + in_offset, c * sizeof(float));
      }
    }
  }

  return NNACL_OK;
}
