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

void CalculateCoordinate(float out, int in, int *bottom, int *top, float *bottom_weight) {
  *bottom = (int)(floorf(out));
  *top = *bottom + 1 < in ? (*bottom + 1) : (in - 1);
  float top_weight = (float)out - (float)(*bottom);
  *bottom_weight = 1.0f - top_weight;
}

static void BicubicBaseFunc(float a, const float x, float *weight) {
  float abs_x = fabsf(x);
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
void CalculateWeightForBicubic(float out, int in, int *index, float *weights, float a) {
  int floor_index = (int)(floorf(out));
  index[0] = (floor_index - 1) < 0 ? 0 : (floor_index - 1);
  index[1] = floor_index;
  index[2] = (floor_index + 1) < in ? (floor_index + 1) : (in - 1);
  index[3] = (floor_index + 2) < in ? (floor_index + 2) : (in - 1);

  // get positive value
  float distance[4] = {-1, 0, 1, 2};
  float tmp_dis = out - (float)floor_index;
  distance[0] -= tmp_dis;
  distance[1] -= tmp_dis;
  distance[2] -= tmp_dis;
  distance[3] -= tmp_dis;

  for (int i = 0; i < 4; ++i) {
    BicubicBaseFunc(a, distance[i], &weights[i]);
  }
}

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

  for (int h = 0; h < new_height; h++) {
    float actual_y = calculate(h, in_h, new_height);
    CalculateCoordinate(actual_y, in_h, y_bottoms + h, y_tops + h, y_bottom_weights + h);
  }
  for (int w = 0; w < new_width; w++) {
    float actual_x = calculate(w, in_w, new_width);
    CalculateCoordinate(actual_x, in_w, x_lefts + w, x_rights + w, x_left_weights + w);
  }
  return NNACL_OK;
}

int PrepareResizeBicubic(const int *input_shape, const int *output_shape, CalculateOriginalCoordinate calculate,
                         int *y_tops, int *x_lefts, float *y_weights, float *x_weights, float cubic_coeff) {
  if (input_shape == NULL || output_shape == NULL || y_tops == NULL || x_lefts == NULL || y_weights == NULL ||
      x_weights == NULL) {
    return NNACL_NULL_PTR;
  }

  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int new_height = output_shape[1];
  int new_width = output_shape[2];

  for (int h = 0; h < new_height; h++) {
    float actual_y = calculate(h, in_h, new_height);
    CalculateWeightForBicubic(actual_y, in_h, y_tops + 4 * h, y_weights + 4 * h, cubic_coeff);
  }
  for (int w = 0; w < new_width; w++) {
    float actual_x = calculate(w, in_w, new_width);
    CalculateWeightForBicubic(actual_x, in_w, x_lefts + 4 * w, x_weights + 4 * w, cubic_coeff);
  }
  return NNACL_OK;
}

int PrepareCropAndResizeBilinear(const int *input_shape, const float *boxes, const int *box_idx,
                                 const int *output_shape, int *y_bottoms, int *y_tops, int *x_lefts, int *x_rights,
                                 float *y_bottom_weights, float *x_left_weights) {
  if (input_shape == NULL || output_shape == NULL || y_bottoms == NULL || y_tops == NULL || x_lefts == NULL ||
      x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }
  int in_h = input_shape[1];
  int in_w = input_shape[2];
  int batch = output_shape[0];
  int new_height = output_shape[1];
  int new_width = output_shape[2];
  float actual_x;
  float actual_y;

  for (int b = 0; b < batch; b++) {
    const float *box = boxes + b * 4;
    float start_h = box[0];
    float end_h = box[2];
    float start_w = box[1];
    float end_w = box[3];
    if (start_h > end_h || start_w > end_w || end_h > 1 || end_w > 1) {
      return NNACL_PARAM_INVALID;
    }

    int *y_bottom = y_bottoms + b * new_height;
    int *y_top = y_tops + b * new_height;
    float *y_bottom_weight = y_bottom_weights + b * new_height;
    int *x_left = x_lefts + b * new_width;
    int *x_right = x_rights + b * new_width;
    float *x_left_weight = x_left_weights + b * new_width;
    for (int h = 0; h < new_height; h++) {
      if (new_height > 1) {
        actual_y = start_h * (in_h - 1) + h * (end_h - start_h) * (in_h - 1) / (new_height - 1);
      } else {
        actual_y = 0.5 * (end_h + start_h) * (in_h - 1);
      }
      CalculateCoordinate(actual_y, in_h, y_bottom + h, y_top + h, y_bottom_weight + h);
    }
    for (int w = 0; w < new_width; w++) {
      if (new_width > 1) {
        actual_x = start_w * (in_w - 1) + w * (end_w - start_w) * (in_w - 1) / (new_width - 1);
      } else {
        actual_x = 0.5 * (end_w + start_w) * (in_w - 1);
      }
      CalculateCoordinate(actual_x, in_w, x_left + w, x_right + w, x_left_weight + w);
    }
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

void Bilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
              const int *y_bottom, const int *y_top, const int *x_left, const int *x_right,
              const float *y_bottom_weight, const float *x_left_weight, float *line0, float *line1, const int h_begin,
              const int h_end) {
  int in_w = input_shape[2];
  int in_c = input_shape[3];
  int new_width = output_shape[2];
  int h_stride = new_width * in_c;

  bool cache_line_used[2] = {false, false};
  int cache_line_num[2] = {-1, -1};
  float *const cache_line_ptr[2] = {line0, line1};
  float *current_line_ptr[2] = {line0, line1};
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
        const float *line = input_data + current_line_num[j] * in_w * in_c;
        for (int k = 0; k < 2; k++) {
          if (!cache_line_used[k]) {
            cache_line_num[k] = current_line_num[j];
            cache_line_used[k] = true;
            current_line_ptr[j] = cache_line_ptr[k];
            InterpRow(line, current_line_ptr[j], new_width, x_left_weight, x_left, x_right, in_c);
            break;
          }
        }
      }
    }
    // do col interp
    InterpCol(current_line_ptr[0], current_line_ptr[1], output_data + h * h_stride, new_width, y_bottom_weight[h],
              in_c);
  }
}

int ResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                   const int *y_bottoms, const int *y_tops, const int *x_lefts, const int *x_rights,
                   const float *y_bottom_weights, const float *x_left_weights, float *line0, float *line1,
                   const int h_begin, const int h_end) {
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
    const float *input = input_data + b * in_h * in_w * in_c;
    float *output = output_data + b * new_height * new_width * in_c;
    Bilinear(input, output, input_shape, output_shape, y_bottoms, y_tops, x_lefts, x_rights, y_bottom_weights,
             x_left_weights, line0, line1, h_begin, h_end);
  }
  return NNACL_OK;
}

void BicubicInterpRow(const float *src, float *dst, const float *weights, const int *lefts, int width, int channel) {
  for (int w = 0; w < width; w++) {
    const float *weight = weights + 4 * w;
    float *dst_w = dst + w * channel;
    const float *src0_w = src + lefts[4 * w] * channel;
    const float *src1_w = src + lefts[4 * w + 1] * channel;
    const float *src2_w = src + lefts[4 * w + 2] * channel;
    const float *src3_w = src + lefts[4 * w + 3] * channel;
    int c = 0;
#ifdef ENABLE_NEON
    float32x4_t weight0_vec = vdupq_n_f32(weight[0]);
    float32x4_t weight1_vec = vdupq_n_f32(weight[1]);
    float32x4_t weight2_vec = vdupq_n_f32(weight[2]);
    float32x4_t weight3_vec = vdupq_n_f32(weight[3]);

    for (; c <= channel - 4; c += 4) {
      float32x4_t src0_vec = vld1q_f32(src0_w + c);
      float32x4_t src1_vec = vld1q_f32(src1_w + c);
      float32x4_t src2_vec = vld1q_f32(src2_w + c);
      float32x4_t src3_vec = vld1q_f32(src3_w + c);

      float32x4_t interp_value =
        src0_vec * weight0_vec + src1_vec * weight1_vec + src2_vec * weight2_vec + src3_vec * weight3_vec;
      vst1q_f32(dst_w + c, interp_value);
    }
#endif
    for (; c < channel; c++) {
      dst_w[c] = src0_w[c] * weight[0] + src1_w[c] * weight[1] + src2_w[c] * weight[2] + src3_w[c] * weight[3];
    }
  }
}

void BicubicInterpCol(const float *src, float *dst, const float *weights, int width, int channel) {
  const float *src0 = src;
  const float *src1 = src + width * channel;
  const float *src2 = src + 2 * width * channel;
  const float *src3 = src + 3 * width * channel;
  for (int w = 0; w < width; w++) {
    float *dst_w = dst + w * channel;
    const float *src0_w = src0 + w * channel;
    const float *src1_w = src1 + w * channel;
    const float *src2_w = src2 + w * channel;
    const float *src3_w = src3 + w * channel;
    int c = 0;
#ifdef ENABLE_NEON
    float32x4_t weight0_vec = vdupq_n_f32(weights[0]);
    float32x4_t weight1_vec = vdupq_n_f32(weights[1]);
    float32x4_t weight2_vec = vdupq_n_f32(weights[2]);
    float32x4_t weight3_vec = vdupq_n_f32(weights[3]);

    for (; c <= channel - 4; c += 4) {
      float32x4_t src0_vec = vld1q_f32(src0_w + c);
      float32x4_t src1_vec = vld1q_f32(src1_w + c);
      float32x4_t src2_vec = vld1q_f32(src2_w + c);
      float32x4_t src3_vec = vld1q_f32(src3_w + c);
      float32x4_t interp_value =
        src0_vec * weight0_vec + src1_vec * weight1_vec + src2_vec * weight2_vec + src3_vec * weight3_vec;
      vst1q_f32(dst_w + c, interp_value);
    }
#endif
    for (; c < channel; c++) {
      dst_w[c] = src0_w[c] * weights[0] + src1_w[c] * weights[1] + src2_w[c] * weights[2] + src3_w[c] * weights[3];
    }
  }
}

void Bicubic(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
             const int *y_tops, const int *x_lefts, const float *y_weights, const float *x_weights, float *line_buffer,
             const int h_begin, const int h_end) {
  int in_w = input_shape[2];
  int in_c = input_shape[3];
  int new_width = output_shape[2];
  int h_stride = new_width * in_c;

  for (int h = h_begin; h < h_end; h++) {
    for (int i = 0; i < 4; ++i) {
      BicubicInterpRow(input_data + y_tops[4 * h + i] * in_w * in_c, line_buffer + i * h_stride, x_weights, x_lefts,
                       new_width, in_c);
    }
    BicubicInterpCol(line_buffer, output_data + h * h_stride, y_weights + 4 * h, new_width, in_c);
  }
}

int ResizeBicubic(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                  const int *y_tops, const int *x_lefts, const float *y_weights, const float *x_weights,
                  float *line_buffer, const int h_begin, const int h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_tops == NULL ||
      x_lefts == NULL || y_weights == NULL || x_weights == NULL) {
    return NNACL_NULL_PTR;
  }
  int input_cube_per_batch = input_shape[1] * input_shape[2] * input_shape[3];
  int output_cube_per_batch = output_shape[1] * output_shape[2] * input_shape[3];
  for (int b = 0; b < input_shape[0]; b++) {
    const float *input = input_data + b * input_cube_per_batch;
    float *output = output_data + b * output_cube_per_batch;
    Bicubic(input, output, input_shape, output_shape, y_tops, x_lefts, y_weights, x_weights, line_buffer, h_begin,
            h_end);
  }
  return NNACL_OK;
}

int CropAndResizeBilinear(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                          const int *y_bottoms, const int *y_tops, const int *x_lefts, const int *x_rights,
                          const float *y_bottom_weights, const float *x_left_weights, float *line0, float *line1,
                          const int h_begin, const int h_end) {
  if (input_data == NULL || output_data == NULL || input_shape == NULL || output_shape == NULL || y_bottoms == NULL ||
      y_tops == NULL || x_lefts == NULL || x_rights == NULL || y_bottom_weights == NULL || x_left_weights == NULL) {
    return NNACL_NULL_PTR;
  }
  int batch = output_shape[0];
  int new_height = output_shape[1];
  int new_width = output_shape[2];
  int new_channel = output_shape[3];

  for (int b = 0; b < batch; b++) {
    const int *y_bottom = y_bottoms + b * new_height;
    const int *y_top = y_tops + b * new_height;
    const float *y_bottom_weight = y_bottom_weights + b * new_height;
    const int *x_left = x_lefts + b * new_width;
    const int *x_right = x_rights + b * new_width;
    const float *x_left_weight = x_left_weights + b * new_width;
    float *output = output_data + b * new_height * new_width * new_channel;

    Bilinear(input_data, output, input_shape, output_shape, y_bottom, y_top, x_left, x_right, y_bottom_weight,
             x_left_weight, line0, line1, h_begin, h_end);
  }
  return NNACL_OK;
}

int ResizeNearestNeighbor(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
                          CalculateOriginalCoordinate calculate, int coordinate_transform_mode, int tid,
                          int thread_num) {
  int c = input_shape[3];
  bool align_corners = coordinate_transform_mode == 1;
  for (int batch = 0; batch < output_shape[0]; batch++) {
    for (int y = tid; y < output_shape[1]; y += thread_num) {
      float actual_y = calculate(y, input_shape[1], output_shape[1]);
      int input_y;
      if (align_corners) {
        input_y = (int)(roundf(actual_y));
      } else {
        input_y = (int)(floorf(actual_y));
      }
      for (int x = 0; x < output_shape[2]; x++) {
        float actual_x = calculate(x, input_shape[2], output_shape[2]);
        int input_x;
        if (align_corners) {
          input_x = (int)(roundf(actual_x));
        } else {
          input_x = (int)(floorf(actual_x));
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
  float scale = (float)(length_resized) / (float)(length_original);
  return (float)(x_resized) / scale;
}

float CalculateAlignCorners(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized - 1) / (float)(length_original - 1);
  return (float)(x_resized) / scale;
}

float CalculateHalfPixel(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / (float)(length_original);
  float actual = (float)(x_resized + 0.5) / scale - 0.5;
  return actual > 0 ? actual : 0;
}
