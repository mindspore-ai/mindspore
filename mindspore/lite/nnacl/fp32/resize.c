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
int PrepareResizeBilinear(const int *input_shape, const int *output_shape, bool align_corners, int *y_bottoms,
                          int *y_tops, int *x_lefts, int *x_rights, float *y_bottom_weights, float *x_left_weights) {
  if (input_shape == NULL || output_shape == NULL || y_bottoms == NULL || y_tops == NULL || x_lefts == NULL ||
      x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];

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

  int h, w;
  for (h = 0; h < new_height; h++) {
    float actual_y = (float)h * height_scale;
    int y_bottom = (int)(floor(actual_y));
    int y_top = y_bottom + 1 < in_h ? (y_bottom + 1) : (in_h - 1);
    float y_top_weight = actual_y - (float)(y_bottom);
    const float y_bottom_weight = 1.0f - y_top_weight;

    y_bottoms[h] = y_bottom;
    y_tops[h] = y_top;
    y_bottom_weights[h] = y_bottom_weight;
  }
  for (w = 0; w < new_width; w++) {
    float actual_x = (float)(w)*width_scale;
    int x_left = (int)(floor(actual_x));
    int x_right = x_left + 1 < in_w ? (x_left + 1) : (in_w - 1);
    float x_right_weight = actual_x - (float)(x_left);
    const float x_left_weight = 1.0f - x_right_weight;

    x_lefts[w] = x_left;
    x_rights[w] = x_right;
    x_left_weights[w] = x_left_weight;
  }
  return NNACL_OK;
}

int ResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                   int *y_bottoms, int *y_tops, int *x_lefts, int *x_rights, float *y_bottom_weights,
                   float *x_left_weights, int n_h_begin, int n_h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_bottoms == NULL ||
      y_tops == NULL || x_lefts == NULL || x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_w = input_shape[2];
  int in_c = input_shape[3];

  int new_height = output_shape[1];
  int new_width = output_shape[2];

  int n_h, n, h, w, c;
  n = n_h_begin / new_height;
  h = n_h_begin % new_height;
  int n_h_stride = new_width * in_c;
  int out_offset = n_h_begin * n_h_stride;
  for (n_h = n_h_begin; n_h < n_h_end; n_h++, h++) {
    if (h == new_height) {
      h = 0;
      n++;
    }
    int y_bottom = y_bottoms[h];
    int y_top = y_tops[h];
    float y_bottom_weight = y_bottom_weights[h];
    float y_top_weight = 1.0f - y_bottom_weight;

    for (w = 0; w < new_width; w++) {
      int x_left = x_lefts[w];
      int x_right = x_rights[w];
      float x_left_weight = x_left_weights[w];
      float x_right_weight = 1.0f - x_left_weight;
      float top_left_weight = y_top_weight * x_left_weight;
      float top_right_weight = y_top_weight * x_right_weight;
      float bottom_left_weight = y_bottom_weight * x_left_weight;
      float bottom_right_weight = y_bottom_weight * x_right_weight;

      c = 0;
      int in_bottom_left_offset = offset(input_shape, n, y_bottom, x_left, c);
      int in_bottom_right_offset = in_bottom_left_offset + (x_right - x_left) * in_c;
      int in_top_left_offset = in_bottom_left_offset + (y_top - y_bottom) * in_w * in_c;
      int in_top_right_offset = in_bottom_right_offset + (y_top - y_bottom) * in_w * in_c;

#ifdef ENABLE_NEON
      float32x4_t top_left_w = vdupq_n_f32(top_left_weight);
      float32x4_t top_right_w = vdupq_n_f32(top_right_weight);
      float32x4_t bottom_left_w = vdupq_n_f32(bottom_left_weight);
      float32x4_t bottom_right_w = vdupq_n_f32(bottom_right_weight);

      for (; c <= in_c - 4; c += 4) {
        float32x4_t bottom_left = vld1q_f32(input_data + in_bottom_left_offset + c);
        float32x4_t bottom_right = vld1q_f32(input_data + in_bottom_right_offset + c);
        float32x4_t top_left = vld1q_f32(input_data + in_top_left_offset + c);
        float32x4_t top_right = vld1q_f32(input_data + in_top_right_offset + c);

        float32x4_t interp_value = vdupq_n_f32(0.0);

        float32x4_t tmp = vmulq_f32(bottom_left, bottom_left_w);
        interp_value = vaddq_f32(interp_value, tmp);

        tmp = vmulq_f32(bottom_right, bottom_right_w);
        interp_value = vaddq_f32(interp_value, tmp);

        tmp = vmulq_f32(top_left, top_left_w);
        interp_value = vaddq_f32(interp_value, tmp);

        tmp = vmulq_f32(top_right, top_right_w);
        interp_value = vaddq_f32(interp_value, tmp);
        vst1q_f32(output_data + out_offset, interp_value);
        out_offset += 4;
      }
#endif
      for (; c < in_c; c++) {
        float bottom_left = input_data[in_bottom_left_offset + c];
        float bottom_right = input_data[in_bottom_right_offset + c];
        float top_left = input_data[in_top_left_offset + c];
        float top_right = input_data[in_top_right_offset + c];
        float interp_value = bottom_left * bottom_left_weight + bottom_right * bottom_right_weight +
                             top_left * top_left_weight + top_right * top_right_weight;
        output_data[out_offset] = interp_value;
        out_offset++;
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
