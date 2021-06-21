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
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "backend/kernel_compiler/gpu/cuda_impl/crop_and_resize_impl.cuh"

// for every position, first calculate position it mirrors from in the new padded array
// adjust calculated position to origin dx array dimensions and copy value
template <typename T>
__global__ void CropAndResize(const size_t size, const T *input_image, float *input_boxes, int *input_box_index,
                              int batch, int input_height, int input_width, int final_height,
                              int final_width, int channel, int method, float extrapol_val, float *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    // input format -> [batch, height, width, channel]
    // adjust pos_temp for accessing different dims channel onwards
    size_t pos_temp = pos;
    const int pos_channel = pos_temp % channel;  // output channel
    pos_temp = pos_temp / channel;
    const int pos_x = pos_temp % final_width;  // output x position
    pos_temp = pos_temp / final_width;
    const int pos_y = pos_temp % final_height;          // output y position
    const int pos_image_idx = pos_temp / final_height;  // output image
    const int box_index = input_box_index[pos_image_idx];
    // crop values
    const float y1 = input_boxes[4 * pos_image_idx + 0];
    const float x1 = input_boxes[4 * pos_image_idx + 1];
    const float y2 = input_boxes[4 * pos_image_idx + 2];
    const float x2 = input_boxes[4 * pos_image_idx + 3];
    // set scale and target pixels
    float scale_height = (y2 - y1) * (input_height - 1) / (final_height - 1);
    float scale_width = (x2 - x1) * (input_width - 1) / (final_width - 1);
    if (final_height <= 1) {
      scale_height = 0;
    }
    if (final_width <= 1) {
      scale_width = 0;
    }
    float target_y = 0;
    float target_x = 0;
    if (final_height > 1) {
      target_y = y1 * (input_height - 1) + pos_y * scale_height;
    } else {
      target_y = 0.5 * (y1 + y2) + (input_height - 1);
    }
    if (final_width > 1) {
      target_x = x1 * (input_width - 1) + pos_x * scale_width;
    } else {
      target_x = 0.5 * (x1 + x2) + (input_width - 1);
    }
    // use extrapolation value if out of range
    if (((target_x < 0) || (target_x > input_width - 1)) || ((target_y < 0) || (target_y > input_height - 1))) {
      output[pos] = extrapol_val;
      continue;
    }
    if ((method == 1) || (method == 3)) {
      // Bilinear/v2
      const int top_y_index = floorf(target_y);
      const int bottom_y_index = ceilf(target_y);
      const int left_x_index = floorf(target_x);
      const int right_x_index = ceilf(target_x);
      const float y_lerp = target_y - top_y_index;
      const float x_lerp = target_x - left_x_index;
      const float top_left = static_cast<float>(
        input_image[((box_index * input_height + top_y_index) * input_width + left_x_index) * channel + pos_channel]);
      const float top_right = static_cast<float>(
        input_image[((box_index * input_height + top_y_index) * input_width + right_x_index) * channel + pos_channel]);
      const float bottom_left = static_cast<float>(
        input_image[((box_index * input_height + bottom_y_index) * input_width + left_x_index) * channel +
                    pos_channel]);
      const float bottom_right = static_cast<float>(
        input_image[((box_index * input_height + bottom_y_index) * input_width + right_x_index) * channel +
                    pos_channel]);
      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
      output[pos] = top + (bottom - top) * y_lerp;
    } else {
      // Nearest Neighbour
      const int closest_x_index = roundf(target_x);
      const int closest_y_index = roundf(target_y);
      const float val = static_cast<float>(
        input_image[((box_index * input_height + closest_y_index) * input_width + closest_x_index) * channel +
                    pos_channel]);
      output[pos] = val;
    }
  }
  return;
}

template <typename T>
void CalCropAndResize(const size_t size, const T *input_image, float *input_boxes, int *input_box_index, int batch,
                      int input_height, int input_width, int final_height, int final_width, int channel,
                      int method, float extrapol_val, float *output, cudaStream_t cuda_stream) {
  CropAndResize<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_image, input_boxes, input_box_index,
                                                                   batch, input_height, input_width, final_height,
                                                                   final_width, channel, method, extrapol_val, output);
  return;
}

template void CalCropAndResize<int8_t>(const size_t size, const int8_t *input_image, float *input_boxes,
                                       int *input_box_index, int batch, int input_height, int input_width,
                                       int final_height, int final_width, int channel, int method,
                                       float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<int16_t>(const size_t size, const int16_t *input_image, float *input_boxes,
                                        int *input_box_index, int batch, int input_height, int input_width,
                                        int final_height, int final_width, int channel, int method,
                                        float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<int32_t>(const size_t size, const int32_t *input_image, float *input_boxes,
                                        int *input_box_index, int batch, int input_height, int input_width,
                                        int final_height, int final_width, int channel, int method,
                                        float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<int64_t>(const size_t size, const int64_t *input_image, float *input_boxes,
                                        int *input_box_index, int batch, int input_height, int input_width,
                                        int final_height, int final_width, int channel, int method,
                                        float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<half>(const size_t size, const half *input_image, float *input_boxes,
                                     int *input_box_index, int batch, int input_height, int input_width,
                                     int final_height, int final_width, int channel, int method,
                                     float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<float>(const size_t size, const float *input_image, float *input_boxes,
                                      int *input_box_index, int batch, int input_height, int input_width,
                                      int final_height, int final_width, int channel, int method,
                                      float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<double>(const size_t size, const double *input_image, float *input_boxes,
                                       int *input_box_index, int batch, int input_height, int input_width,
                                       int final_height, int final_width, int channel, int method,
                                       float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<uint8_t>(const size_t size, const uint8_t *input_image, float *input_boxes,
                                        int *input_box_index, int batch, int input_height, int input_width,
                                        int final_height, int final_width, int channel, int method,
                                        float extrapol_val, float *output, cudaStream_t cuda_stream);
template void CalCropAndResize<uint16_t>(const size_t size, const uint16_t *input_image, float *input_boxes,
                                         int *input_box_index, int batch, int input_height, int input_width,
                                         int final_height, int final_width, int channel, int method,
                                         float extrapol_val, float *output, cudaStream_t cuda_stream);
