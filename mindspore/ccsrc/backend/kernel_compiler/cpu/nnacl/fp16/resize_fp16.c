/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp16/resize_fp16.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

void CalculateCoordinateFp16(float16_t out, int in, int *bottom, int *top, float16_t *bottom_weight) {
  *bottom = (int)(floorf(out));
  *bottom = *bottom >= 0 ? *bottom : 0;  // extrapolate may generate neg value
  *top = *bottom + 1 < in ? (*bottom + 1) : (in - 1);
  float16_t top_weight = (float16_t)out - (float16_t)(*bottom);
  *bottom_weight = 1.0f - top_weight;
}

static void BicubicBaseFuncFp16(float16_t a, float16_t x, float16_t *weight) {
  float16_t abs_x = fabsf(x);
  if (abs_x >= 0 && abs_x <= 1) {
    *weight = ((a + 2) * abs_x - (a + 3)) * abs_x * abs_x + 1;
  } else if (abs_x > 1 && abs_x <= 2) {
    *weight = a * abs_x * abs_x * abs_x - 5 * a * abs_x * abs_x + 8 * a * abs_x - 4 * a;
  } else {
    *weight = 0;
  }
}

// a is a coefficient
// W(x) = { (a + 2) * |x| * |x| * |x| - (a + 3) * |x| * |x| + 1,           for |x| <= 1
//        { a * |x| * |x| * |x| - 5 * a * |x| * |x| + 8 * a *|x| - 4 * a,  for 1 < |x| < 2
//        { 0,                                                             otherwise
// the value of 'a' depends on if is half_pixel_center(the scheme is the same as tf).
// If is half pixel mode, a equals to -0.5, otherwise -0.75.
void CalculateWeightForBicubicFp16(float16_t out, int in, int *index, float16_t *weights, float16_t a) {
  int floor_index = (int)(floorf(out));
  index[0] = (floor_index - 1) < 0 ? 0 : (floor_index - 1);
  index[1] = floor_index;
  index[2] = (floor_index + 1) < in ? (floor_index + 1) : (in - 1);
  index[3] = (floor_index + 2) < in ? (floor_index + 2) : (in - 1);

  // get positive value
  float16_t distance[4] = {-1, 0, 1, 2};
  float16_t tmp_dis = out - (float16_t)floor_index;
  distance[0] -= tmp_dis;
  distance[1] -= tmp_dis;
  distance[2] -= tmp_dis;
  distance[3] -= tmp_dis;

  for (int i = 0; i < 4; ++i) {
    BicubicBaseFuncFp16(a, distance[i], &weights[i]);
  }
}

int PrepareResizeBilinearFp16(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                              int *y_bottoms, int *y_tops, int *x_lefts, int *x_rights, float16_t *y_bottom_weights,
                              float16_t *x_left_weights) {
  if (input_shape == NULL || output_shape == NULL || y_bottoms == NULL || y_tops == NULL || x_lefts == NULL ||
      x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];

  int new_height = output_shape[1];
  int new_width = output_shape[2];

  for (int h = 0; h < new_height; h++) {
    float16_t actual_y = calculate(h, in_h, new_height);
    CalculateCoordinateFp16(actual_y, in_h, y_bottoms + h, y_tops + h, y_bottom_weights + h);
  }
  for (int w = 0; w < new_width; w++) {
    float16_t actual_x = calculate(w, in_w, new_width);
    CalculateCoordinateFp16(actual_x, in_w, x_lefts + w, x_rights + w, x_left_weights + w);
  }
  return NNACL_OK;
}

int PrepareResizeBicubicFp16(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                             int *y_tops, int *x_lefts, float16_t *y_weights, float16_t *x_weights,
                             float16_t cubic_coeff) {
  if (input_shape == NULL || output_shape == NULL || y_tops == NULL || x_lefts == NULL || y_weights == NULL ||
      x_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int new_height = output_shape[1];
  int new_width = output_shape[2];

  for (int h = 0; h < new_height; h++) {
    float16_t actual_y = calculate(h, in_h, new_height);
    CalculateWeightForBicubicFp16(actual_y, in_h, y_tops + 4 * h, y_weights + 4 * h, cubic_coeff);
  }
  for (int w = 0; w < new_width; w++) {
    float16_t actual_x = calculate(w, in_w, new_width);
    CalculateWeightForBicubicFp16(actual_x, in_w, x_lefts + 4 * w, x_weights + 4 * w, cubic_coeff);
  }
  return NNACL_OK;
}

int InterpRowFp16(const float16_t *src_line, float16_t *linear_output, int new_width, const float16_t *x_left_weights,
                  const int *x_lefts, const int *x_rights, int in_c) {
  int w;
  for (w = 0; w < new_width; w++) {
    int c = 0;
#if defined(ENABLE_NEON)
    float16x8_t left_w_8 = vdupq_n_f16(x_left_weights[w]);
    float16x8_t right_w_8 = vdupq_n_f16(1.0f - x_left_weights[w]);
    for (; c <= in_c - C8NUM; c += C8NUM) {
      float16x8_t left = vld1q_f16(src_line + x_lefts[w] * in_c + c);
      float16x8_t right = vld1q_f16(src_line + x_rights[w] * in_c + c);
      float16x8_t interp_value = vaddq_f16(vmulq_f16(left, left_w_8), vmulq_f16(right, right_w_8));
      vst1q_f16(linear_output + w * in_c + c, interp_value);
    }
#endif
    int left_w_offset = x_lefts[w] * in_c;
    int right_w_offset = x_rights[w] * in_c;
    for (; c < in_c; c++) {
      float16_t left = src_line[left_w_offset + c];
      float16_t right = src_line[right_w_offset + c];
      linear_output[w * in_c + c] = left * x_left_weights[w] + right * (1.0f - x_left_weights[w]);
    }
  }
  return 0;
}

int InterpColFp16(const float16_t *bottom_line, const float16_t *top_line, float16_t *output, int new_width,
                  float16_t y_bottom_weight, int in_c) {
  int w;
  for (w = 0; w < new_width; w++) {
    int c = 0;
#if defined(ENABLE_NEON)
    float16x8_t bottom_w_8 = vdupq_n_f16(y_bottom_weight);
    float16x8_t top_w_8 = vdupq_n_f16(1.0f - y_bottom_weight);
    for (; c <= in_c - C8NUM; c += C8NUM) {
      float16x8_t bottom = vld1q_f16(bottom_line + w * in_c + c);
      float16x8_t top = vld1q_f16(top_line + w * in_c + c);
      float16x8_t interp_value = vaddq_f16(vmulq_f16(bottom, bottom_w_8), vmulq_f16(top, top_w_8));
      vst1q_f16(output + w * in_c + c, interp_value);
    }
#endif
    for (; c < in_c; c++) {
      float16_t bottom = bottom_line[w * in_c + c];
      float16_t top = top_line[w * in_c + c];
      output[w * in_c + c] = bottom * y_bottom_weight + top * (1.0f - y_bottom_weight);
    }
  }
  return 0;
}

void BilinearFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape, const int *output_shape,
                  const int *y_bottom, const int *y_top, const int *x_left, const int *x_right,
                  const float16_t *y_bottom_weight, const float16_t *x_left_weight, float16_t *line0, float16_t *line1,
                  const int h_begin, const int h_end) {
  int in_w = input_shape[2];
  int in_c = input_shape[3];
  int new_width = output_shape[2];
  int h_stride = new_width * in_c;

  bool cache_line_used[2] = {false, false};
  int cache_line_num[2] = {-1, -1};
  float16_t *const cache_line_ptr[2] = {line0, line1};
  float16_t *current_line_ptr[2] = {line0, line1};
  int current_line_num[2] = {-1, -1};

  for (int h = h_begin; h < h_end; h++) {
    current_line_num[0] = y_bottom[h];
    current_line_num[1] = y_top[h];

    for (int i = 0; i < 2; i++) {
      cache_line_used[i] = false;
    }
    // search if we cached
    for (int j = 0; j < 2; j++) {
      bool find = false;
      for (int k = 0; k < 2; k++) {
        if (current_line_num[j] == cache_line_num[k]) {
          cache_line_used[k] = true;
          current_line_ptr[j] = cache_line_ptr[k];
          find = true;
          break;
        }
      }

      if (!find) {
        const float16_t *line = input_data + current_line_num[j] * in_w * in_c;
        for (int k = 0; k < 2; k++) {
          if (!cache_line_used[k]) {
            cache_line_num[k] = current_line_num[j];
            cache_line_used[k] = true;
            current_line_ptr[j] = cache_line_ptr[k];
            InterpRowFp16(line, current_line_ptr[j], new_width, x_left_weight, x_left, x_right, in_c);
            break;
          }
        }
      }
    }
    // do col interp
    InterpColFp16(current_line_ptr[0], current_line_ptr[1], output_data + h * h_stride, new_width, y_bottom_weight[h],
                  in_c);
  }
}

int ResizeBilinearFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                       const int *output_shape, const int *y_bottoms, const int *y_tops, const int *x_lefts,
                       const int *x_rights, const float16_t *y_bottom_weights, const float16_t *x_left_weights,
                       float16_t *line0, float16_t *line1, const int h_begin, const int h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_bottoms == NULL ||
      y_tops == NULL || x_lefts == NULL || x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_b = input_shape[0];
  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int in_c = input_shape[3];
  int new_height = output_shape[1];
  int new_width = output_shape[2];

  for (int b = 0; b < in_b; b++) {
    const float16_t *input = input_data + b * in_h * in_w * in_c;
    float16_t *output = output_data + b * new_height * new_width * in_c;
    BilinearFp16(input, output, input_shape, output_shape, y_bottoms, y_tops, x_lefts, x_rights, y_bottom_weights,
                 x_left_weights, line0, line1, h_begin, h_end);
  }
  return NNACL_OK;
}

void BicubicInterpRowFp16(const float16_t *src, float16_t *dst, const float16_t *weights, const int *lefts, int width,
                          int channel) {
  for (int w = 0; w < width; w++) {
    const float16_t *weight = weights + 4 * w;
    float16_t *dst_w = dst + w * channel;
    const float16_t *src0_w = src + lefts[4 * w] * channel;
    const float16_t *src1_w = src + lefts[4 * w + 1] * channel;
    const float16_t *src2_w = src + lefts[4 * w + 2] * channel;
    const float16_t *src3_w = src + lefts[4 * w + 3] * channel;
    int c = 0;
#if defined(ENABLE_NEON)
    float16x8_t weight0_vec_8 = vdupq_n_f16(weight[0]);
    float16x8_t weight1_vec_8 = vdupq_n_f16(weight[1]);
    float16x8_t weight2_vec_8 = vdupq_n_f16(weight[2]);
    float16x8_t weight3_vec_8 = vdupq_n_f16(weight[3]);
    for (; c <= channel - C8NUM; c += C8NUM) {
      float16x8_t src0_vec = vld1q_f16(src0_w + c);
      float16x8_t src1_vec = vld1q_f16(src1_w + c);
      float16x8_t src2_vec = vld1q_f16(src2_w + c);
      float16x8_t src3_vec = vld1q_f16(src3_w + c);
      float16x8_t dst0 = vmulq_f16(src0_vec, weight0_vec_8);
      float16x8_t dst1 = vmulq_f16(src1_vec, weight1_vec_8);
      float16x8_t dst2 = vmulq_f16(src2_vec, weight2_vec_8);
      float16x8_t dst3 = vmulq_f16(src3_vec, weight3_vec_8);
      float16x8_t interp_value = vaddq_f16(dst3, vaddq_f16(dst2, vaddq_f16(dst1, dst0)));
      vst1q_f16(dst_w + c, interp_value);
    }
#endif
    for (; c < channel; c++) {
      dst_w[c] = src0_w[c] * weight[0] + src1_w[c] * weight[1] + src2_w[c] * weight[2] + src3_w[c] * weight[3];
    }
  }
}

void BicubicInterpColFp16(const float16_t *src, float16_t *dst, const float16_t *weights, int width, int channel) {
  const float16_t *src0 = src;
  const float16_t *src1 = src + width * channel;
  const float16_t *src2 = src + 2 * width * channel;
  const float16_t *src3 = src + 3 * width * channel;
  for (int w = 0; w < width; w++) {
    float16_t *dst_w = dst + w * channel;
    const float16_t *src0_w = src0 + w * channel;
    const float16_t *src1_w = src1 + w * channel;
    const float16_t *src2_w = src2 + w * channel;
    const float16_t *src3_w = src3 + w * channel;
    int c = 0;
#ifdef ENABLE_NEON
    float16x8_t weight0_vec_8 = vdupq_n_f16(weights[0]);
    float16x8_t weight1_vec_8 = vdupq_n_f16(weights[1]);
    float16x8_t weight2_vec_8 = vdupq_n_f16(weights[2]);
    float16x8_t weight3_vec_8 = vdupq_n_f16(weights[3]);
    for (; c <= channel - C8NUM; c += C8NUM) {
      float16x8_t src0_vec = vld1q_f16(src0_w + c);
      float16x8_t src1_vec = vld1q_f16(src1_w + c);
      float16x8_t src2_vec = vld1q_f16(src2_w + c);
      float16x8_t src3_vec = vld1q_f16(src3_w + c);
      float16x8_t dst1 = vmulq_f16(src0_vec, weight0_vec_8);
      float16x8_t dst2 = vmulq_f16(src1_vec, weight1_vec_8);
      float16x8_t dst3 = vmulq_f16(src2_vec, weight2_vec_8);
      float16x8_t dst4 = vmulq_f16(src3_vec, weight3_vec_8);
      float16x8_t interp_value = vaddq_f16(dst4, vaddq_f16(dst3, vaddq_f16(dst1, dst2)));
      vst1q_f16(dst_w + c, interp_value);
    }
#endif
    for (; c < channel; c++) {
      dst_w[c] = src0_w[c] * weights[0] + src1_w[c] * weights[1] + src2_w[c] * weights[2] + src3_w[c] * weights[3];
    }
  }
}

void BicubicFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape, const int *output_shape,
                 const int *y_tops, const int *x_lefts, const float16_t *y_weights, const float16_t *x_weights,
                 float16_t *line_buffer, const int h_begin, const int h_end) {
  int in_w = input_shape[2];
  int in_c = input_shape[3];
  int new_width = output_shape[2];
  int h_stride = new_width * in_c;

  for (int h = h_begin; h < h_end; h++) {
    for (int i = 0; i < 4; ++i) {
      BicubicInterpRowFp16(input_data + y_tops[4 * h + i] * in_w * in_c, line_buffer + i * h_stride, x_weights, x_lefts,
                           new_width, in_c);
    }
    BicubicInterpColFp16(line_buffer, output_data + h * h_stride, y_weights + 4 * h, new_width, in_c);
  }
}

int ResizeBicubicFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                      const int *output_shape, const int *y_tops, const int *x_lefts, const float16_t *y_weights,
                      const float16_t *x_weights, float16_t *line_buffer, const int h_begin, const int h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_tops == NULL ||
      x_lefts == NULL || y_weights == NULL || x_weights == NULL) {
    return NNACL_NULL_PTR;
  }
  int input_cube_per_batch = input_shape[1] * input_shape[2] * input_shape[3];
  int output_cube_per_batch = output_shape[1] * output_shape[2] * input_shape[3];
  for (int b = 0; b < input_shape[0]; b++) {
    const float16_t *input = input_data + b * input_cube_per_batch;
    float16_t *output = output_data + b * output_cube_per_batch;
    BicubicFp16(input, output, input_shape, output_shape, y_tops, x_lefts, y_weights, x_weights, line_buffer, h_begin,
                h_end);
  }
  return NNACL_OK;
}

int ResizeNearestNeighborFp16(const float16_t *input_data, float16_t *output_data, const int *input_shape,
                              const int *output_shape, CalculateOriginalCoordinate calculate,
                              int coordinate_transform_mode, int tid, int thread_num) {
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }
  int c = input_shape[3];
  bool align_corners = coordinate_transform_mode == 1;
  for (int batch = 0; batch < output_shape[0]; batch++) {
    for (int y = tid; y < output_shape[1]; y += thread_num) {
      float16_t actual_y = calculate(y, input_shape[1], output_shape[1]);
      int input_y;
      if (align_corners) {
        input_y = (int)(roundf(actual_y));
      } else {
        input_y = (int)(floorf(actual_y));
      }
      for (int x = 0; x < output_shape[2]; x++) {
        float16_t actual_x = calculate(x, input_shape[2], output_shape[2]);
        int input_x;
        if (align_corners) {
          input_x = (int)(roundf(actual_x));
        } else {
          input_x = (int)(floorf(actual_x));
        }
        int in_offset = Offset(input_shape, batch, input_y, input_x, 0);
        int out_offset = Offset(output_shape, batch, y, x, 0);
        memcpy(output_data + out_offset, input_data + in_offset, c * sizeof(float16_t));
      }
    }
  }
  return NNACL_OK;
}
