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
#include "nnacl/fp32/resize_fp32.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

int PrepareResizeBilinear(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                          int *y_bottoms, int *y_tops, int *x_lefts, int *x_rights, float *y_bottom_weights,
                          float *x_left_weights) {
  if (input_shape == NULL || output_shape == NULL || y_bottoms == NULL || y_tops == NULL || x_lefts == NULL ||
      x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];

  int new_height = output_shape[1];
  int new_width = output_shape[2];

  int h, w;
  for (h = 0; h < new_height; h++) {
    float actual_y = calculate(h, in_h, new_height);
    int y_bottom = (int)(floor(actual_y));
    int y_top = y_bottom + 1 < in_h ? (y_bottom + 1) : (in_h - 1);
    float y_top_weight = actual_y - (float)(y_bottom);
    const float y_bottom_weight = 1.0f - y_top_weight;

    y_bottoms[h] = y_bottom;
    y_tops[h] = y_top;
    y_bottom_weights[h] = y_bottom_weight;
  }
  for (w = 0; w < new_width; w++) {
    float actual_x = calculate(w, in_w, new_width);
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

int InterpRow(const float *src_line, float *linear_output, int new_width, const float *x_left_weights,
              const int *x_lefts, const int *x_rights, int in_c) {
  int w;
  for (w = 0; w < new_width; w++) {
    int c = 0;
#ifdef ENABLE_NEON
    float32x4_t left_w = vdupq_n_f32(x_left_weights[w]);
    float32x4_t right_w = vdupq_n_f32(1.0f - x_left_weights[w]);

    for (; c <= in_c - 4; c += 4) {
      float32x4_t left = vld1q_f32(src_line + x_lefts[w] * in_c + c);
      float32x4_t right = vld1q_f32(src_line + x_rights[w] * in_c + c);

      float32x4_t interp_value = left * left_w + right * right_w;
      vst1q_f32(linear_output + w * in_c + c, interp_value);
    }
#endif
    int left_w_offset = x_lefts[w] * in_c;
    int right_w_offset = x_rights[w] * in_c;
    for (; c < in_c; c++) {
      float left = src_line[left_w_offset + c];
      float right = src_line[right_w_offset + c];
      linear_output[w * in_c + c] = left * x_left_weights[w] + right * (1.0f - x_left_weights[w]);
    }
  }
  return 0;
}

int InterpCol(const float *bottom_line, const float *top_line, float *output, int new_width, float y_bottom_weight,
              int in_c) {
  int w;
  for (w = 0; w < new_width; w++) {
    int c = 0;
#ifdef ENABLE_NEON
    float32x4_t bottom_w = vdupq_n_f32(y_bottom_weight);
    float32x4_t top_w = vdupq_n_f32(1.0f - y_bottom_weight);

    for (; c <= in_c - 4; c += 4) {
      float32x4_t bottom = vld1q_f32(bottom_line + w * in_c + c);
      float32x4_t top = vld1q_f32(top_line + w * in_c + c);
      float32x4_t interp_value = bottom * bottom_w + top * top_w;
      vst1q_f32(output + w * in_c + c, interp_value);
    }
#endif
    for (; c < in_c; c++) {
      float bottom = bottom_line[w * in_c + c];
      float top = top_line[w * in_c + c];
      output[w * in_c + c] = bottom * y_bottom_weight + top * (1.0f - y_bottom_weight);
    }
  }
  return 0;
}

int ResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                   const int *y_bottoms, const int *y_tops, const int *x_lefts, const int *x_rights,
                   const float *y_bottom_weights, const float *x_left_weights, float *line0, float *line1,
                   const int n_h_begin, const int n_h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_bottoms == NULL ||
      y_tops == NULL || x_lefts == NULL || x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int in_c = input_shape[3];

  int new_height = output_shape[1];
  int new_width = output_shape[2];

  int n_h;
  int n_h_stride = new_width * in_c;

  bool cache_line_used[2] = {false, false};
  int cache_line_num[2] = {-1, -1};
  float *const cache_line_ptr[2] = {line0, line1};
  float *current_line_ptr[2] = {line0, line1};
  int current_line_num[2] = {-1, -1};

  for (n_h = n_h_begin; n_h < n_h_end; n_h++) {
    int n, h;
    n = n_h / new_height;
    h = n_h % new_height;

    current_line_num[0] = n * in_h + y_bottoms[h];
    current_line_num[1] = n * in_h + y_tops[h];
    int i;
    for (i = 0; i < 2; i++) {
      cache_line_used[i] = false;
    }
    // search if we cached
    int j, k;
    for (j = 0; j < 2; j++) {
      bool find = false;
      for (k = 0; k < 2; k++) {
        if (current_line_num[j] == cache_line_num[k]) {
          cache_line_used[k] = true;
          current_line_ptr[j] = cache_line_ptr[k];
          find = true;
          break;
        }
      }

      if (!find) {
        const float *line = input_data + current_line_num[j] * in_w * in_c;
        for (k = 0; k < 2; k++) {
          if (!cache_line_used[k]) {
            cache_line_num[k] = current_line_num[j];
            cache_line_used[k] = true;
            current_line_ptr[j] = cache_line_ptr[k];
            InterpRow(line, current_line_ptr[j], new_width, x_left_weights, x_lefts, x_rights, in_c);
            break;
          }
        }
      }
    }

    // do col interp
    InterpCol(current_line_ptr[0], current_line_ptr[1], output_data + n_h * n_h_stride, new_width, y_bottom_weights[h],
              in_c);
  }

  return NNACL_OK;
}

int ResizeNearestNeighbor(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                          CalculateOriginalCoordinate calculate, int coordinate_transform_mode, int tid,
                          int thread_num) {
  int batch, y, x, c;
  c = input_shape[3];
  bool align_corners = coordinate_transform_mode == 1;
  for (batch = 0; batch < output_shape[0]; batch++) {
    for (y = tid; y < output_shape[1]; y += thread_num) {
      float actual_y = calculate(y, input_shape[1], output_shape[1]);
      int input_y;
      if (align_corners) {
        input_y = (int)(round(actual_y));
      } else {
        input_y = (int)(floor(actual_y));
      }
      for (x = 0; x < output_shape[2]; x++) {
        float actual_x = calculate(x, input_shape[2], output_shape[2]);
        int input_x;
        if (align_corners) {
          input_x = (int)(round(actual_x));
        } else {
          input_x = (int)(floor(actual_x));
        }
        int in_offset = offset(input_shape, batch, input_y, input_x, 0);
        int out_offset = offset(output_shape, batch, y, x, 0);
        memcpy(output_data + out_offset, input_data + in_offset, c * sizeof(float));
      }
    }
  }

  return NNACL_OK;
}

float CalculateAsymmetric(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / length_original;
  return (float)(x_resized) / scale;
}

float CalculateAlignCorners(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized - 1) / (length_original - 1);
  return (float)(x_resized) / scale;
}

float CalculateHalfPixel(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / length_original;
  float actual = (float)(x_resized + 0.5) / scale - 0.5;
  return actual > 0 ? actual : 0;
}
