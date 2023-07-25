/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <stdint.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/crop_and_resize_grad_image_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T, typename G>
__global__ void CropAndResizeGradImageForwardKernel(const int32_t size, const T *grads, const T *boxes,
                                                    const int *box_ind, int32_t num_boxes, int32_t batch,
                                                    int32_t image_height, int32_t image_width, int32_t crop_height,
                                                    int32_t crop_width, int32_t depth, int method, G *grad_image) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    // input format -> [num_boxes, crop_height, crop_width, depth]
    int pos_temp = pos;
    const int32_t pos_channel = pos_temp % depth;
    pos_temp = pos_temp / depth;
    const int32_t pos_x = pos_temp % crop_width;
    pos_temp = pos_temp / crop_width;
    const int32_t pos_y = pos_temp % crop_height;
    const int32_t pos_box_idx = pos_temp / crop_height;
    const T y1 = boxes[4 * pos_box_idx];
    const T x1 = boxes[4 * pos_box_idx + 1];
    const T y2 = boxes[4 * pos_box_idx + 2];
    const T x2 = boxes[4 * pos_box_idx + 3];
    const int32_t box_in_image = box_ind[pos_box_idx];
    if (box_in_image < 0 || box_in_image >= batch) {
      continue;
    }
    const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;
    float target_y =
      (crop_height > 1) ? y1 * (image_height - 1) + pos_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    float target_x =
      (crop_width > 1) ? x1 * (image_width - 1) + pos_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (target_y < 0 || target_y > image_height - 1) {
      continue;
    }
    if (target_x < 0 || target_x > image_width - 1) {
      continue;
    }
    if ((method == 1) || (method == 3)) {
      const int32_t top_y_index = floorf(target_y);
      const int32_t bottom_y_index = ceilf(target_y);
      const float y_lerp = target_y - top_y_index;
      const int32_t left_x_ind = floorf(target_x);
      const int32_t right_x_ind = ceilf(target_x);
      const float x_lerp = target_x - left_x_ind;
      // Compute the image gradient
      const float top_grad = (1 - y_lerp) * static_cast<float>(grads[pos]);
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + left_x_ind) * depth + pos_channel,
        static_cast<G>((1 - x_lerp) * top_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + right_x_ind) * depth + pos_channel,
        static_cast<G>(x_lerp * top_grad));
      const float bottom_grad = y_lerp * static_cast<float>(grads[pos]);
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + left_x_ind) * depth + pos_channel,
        static_cast<G>((1 - x_lerp) * bottom_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + right_x_ind) * depth + pos_channel,
        static_cast<G>(x_lerp * bottom_grad));
    } else {
      const int32_t closest_x_index = roundf(target_x);
      const int32_t closest_y_index = roundf(target_y);
      MsAtomicAdd(grad_image +
                    ((box_in_image * image_height + closest_y_index) * image_width + closest_x_index) * depth +
                    pos_channel,
                  static_cast<G>(grads[pos]));
    }
  }
  return;
}

template <>
__global__ void CropAndResizeGradImageForwardKernel(const int32_t size, const float *grads, const float *boxes,
                                                    const int *box_ind, int32_t num_boxes, int32_t batch,
                                                    int32_t image_height, int32_t image_width, int32_t crop_height,
                                                    int32_t crop_width, int32_t depth, int method, half *grad_image) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    // input format -> [num_boxes, crop_height, crop_width, depth]
    int pos_temp = pos;
    const int32_t pos_channel = pos_temp % depth;
    pos_temp = pos_temp / depth;
    const int32_t pos_x = pos_temp % crop_width;
    pos_temp = pos_temp / crop_width;
    const int32_t pos_y = pos_temp % crop_height;
    const int32_t pos_box_idx = pos_temp / crop_height;
    const float y1 = boxes[4 * pos_box_idx];
    const float x1 = boxes[4 * pos_box_idx + 1];
    const float y2 = boxes[4 * pos_box_idx + 2];
    const float x2 = boxes[4 * pos_box_idx + 3];
    const int32_t box_in_image = box_ind[pos_box_idx];
    if (box_in_image < 0 || box_in_image >= batch) {
      continue;
    }
    const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;
    float target_y =
      (crop_height > 1) ? y1 * (image_height - 1) + pos_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    float target_x =
      (crop_width > 1) ? x1 * (image_width - 1) + pos_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (target_y < 0 || target_y > image_height - 1) {
      continue;
    }
    if (target_x < 0 || target_x > image_width - 1) {
      continue;
    }
    if ((method == 1) || (method == 3)) {
      const int32_t top_y_index = floorf(target_y);
      const int32_t bottom_y_index = ceilf(target_y);
      const float y_lerp = target_y - top_y_index;
      const int32_t left_x_ind = floorf(target_x);
      const int32_t right_x_ind = ceilf(target_x);
      const float x_lerp = target_x - left_x_ind;
      // Compute the image gradient
      const float top_grad = (1 - y_lerp) * grads[pos];
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + left_x_ind) * depth + pos_channel,
        __float2half((1 - x_lerp) * top_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + right_x_ind) * depth + pos_channel,
        __float2half(x_lerp * top_grad));
      const float bottom_grad = y_lerp * grads[pos];
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + left_x_ind) * depth + pos_channel,
        __float2half((1 - x_lerp) * bottom_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + right_x_ind) * depth + pos_channel,
        __float2half(x_lerp * bottom_grad));
    } else {
      const int32_t closest_x_index = roundf(target_x);
      const int32_t closest_y_index = roundf(target_y);
      MsAtomicAdd(grad_image +
                    ((box_in_image * image_height + closest_y_index) * image_width + closest_x_index) * depth +
                    pos_channel,
                  __float2half(grads[pos]));
    }
  }
  return;
}

template <>
__global__ void CropAndResizeGradImageForwardKernel(const int32_t size, const double *grads, const double *boxes,
                                                    const int *box_ind, int32_t num_boxes, int32_t batch,
                                                    int32_t image_height, int32_t image_width, int32_t crop_height,
                                                    int32_t crop_width, int32_t depth, int method, half *grad_image) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    // input format -> [num_boxes, crop_height, crop_width, depth]
    int pos_temp = pos;
    const int32_t pos_channel = pos_temp % depth;
    pos_temp = pos_temp / depth;
    const int32_t pos_x = pos_temp % crop_width;
    pos_temp = pos_temp / crop_width;
    const int32_t pos_y = pos_temp % crop_height;
    const int32_t pos_box_idx = pos_temp / crop_height;
    const double y1 = boxes[4 * pos_box_idx];
    const double x1 = boxes[4 * pos_box_idx + 1];
    const double y2 = boxes[4 * pos_box_idx + 2];
    const double x2 = boxes[4 * pos_box_idx + 3];
    const int32_t box_in_image = box_ind[pos_box_idx];
    if (box_in_image < 0 || box_in_image >= batch) {
      continue;
    }
    const double height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const double width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;
    double target_y =
      (crop_height > 1) ? y1 * (image_height - 1) + pos_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    double target_x =
      (crop_width > 1) ? x1 * (image_width - 1) + pos_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (target_y < 0 || target_y > image_height - 1) {
      continue;
    }
    if (target_x < 0 || target_x > image_width - 1) {
      continue;
    }
    if ((method == 1) || (method == 3)) {
      const int32_t top_y_index = floorf(target_y);
      const int32_t bottom_y_index = ceilf(target_y);
      const float y_lerp = static_cast<float>(target_y - top_y_index);
      const int32_t left_x_ind = floorf(target_x);
      const int32_t right_x_ind = ceilf(target_x);
      const float x_lerp = static_cast<float>(target_x - left_x_ind);
      // Compute the image gradient
      const float top_grad = (1 - y_lerp) * static_cast<float>(grads[pos]);
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + left_x_ind) * depth + pos_channel,
        __float2half((1 - x_lerp) * top_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + top_y_index) * image_width + right_x_ind) * depth + pos_channel,
        __float2half(x_lerp * top_grad));
      const float bottom_grad = y_lerp * static_cast<float>(grads[pos]);
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + left_x_ind) * depth + pos_channel,
        __float2half((1 - x_lerp) * bottom_grad));
      MsAtomicAdd(
        grad_image + ((box_in_image * image_height + bottom_y_index) * image_width + right_x_ind) * depth + pos_channel,
        __float2half(x_lerp * bottom_grad));
    } else {
      const int32_t closest_x_index = roundf(target_x);
      const int32_t closest_y_index = roundf(target_y);
      MsAtomicAdd(grad_image +
                    ((box_in_image * image_height + closest_y_index) * image_width + closest_x_index) * depth +
                    pos_channel,
                  __float2half(static_cast<float>(grads[pos])));
    }
  }
  return;
}

template <typename G>
__global__ void Reset_zero(const int size, G *list) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const G replace_element = 0.;
    list[pos] = replace_element;
  }
  return;
}

template <typename T, typename G>
cudaError_t CalCropAndResizeGradImage(const int32_t size, const T *grads, const T *boxes, const int *box_ind,
                                      int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width,
                                      int32_t crop_height, int32_t crop_width, int32_t depth, int method, G *grad_image,
                                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  int zero_threads_num = static_cast<int>(batch * image_height * image_width * depth);
  Reset_zero<<<CUDA_BLOCKS(device_id, zero_threads_num), CUDA_THREADS(device_id), 0, cuda_stream>>>(zero_threads_num,
                                                                                                    grad_image);
  CropAndResizeGradImageForwardKernel<<<CUDA_BLOCKS(device_id, static_cast<int>(size)), CUDA_THREADS(device_id), 0,
                                        cuda_stream>>>(size, grads, boxes, box_ind, num_boxes, batch, image_height,
                                                       image_width, crop_height, crop_width, depth, method, grad_image);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<float, half>(
  const int32_t size, const float *grads, const float *boxes, const int32_t *box_ind, int32_t num_boxes, int32_t batch,
  int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth, int method,
  half *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<float, float>(
  const int32_t size, const float *grads, const float *boxes, const int32_t *box_ind, int32_t num_boxes, int32_t batch,
  int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth, int method,
  float *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<float, double>(
  const int32_t size, const float *grads, const float *boxes, const int32_t *box_ind, int32_t num_boxes, int32_t batch,
  int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth, int method,
  double *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<double, half>(
  const int32_t size, const double *grads, const double *boxes, const int32_t *box_ind, int32_t num_boxes,
  int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth,
  int method, half *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<double, float>(
  const int32_t size, const double *grads, const double *boxes, const int32_t *box_ind, int32_t num_boxes,
  int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth,
  int method, float *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage<double, double>(
  const int32_t size, const double *grads, const double *boxes, const int32_t *box_ind, int32_t num_boxes,
  int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width, int32_t depth,
  int method, double *grad_image, const uint32_t &device_id, cudaStream_t cuda_stream);
