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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/crop_and_resize_grad_boxes_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
template <typename T, typename G>
__global__ void CropAndResizeGradBoxesForwardKernel(const int32_t size, const G *grads, const T *image, const G *boxes,
                                                    const int *box_ind, int32_t num_boxes, int32_t batch,
                                                    int32_t image_height, int32_t image_width, int32_t crop_height,
                                                    int32_t crop_width, int32_t depth, G *grad_boxes) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    // input format -> [num_boxes, crop_height, crop_width, depth]
    int pos_temp = pos;
    const int32_t pos_channel = pos_temp % depth;
    pos_temp = pos_temp / depth;
    const int32_t pos_x = pos_temp % crop_width;
    pos_temp = pos_temp / crop_width;
    const int32_t pos_y = pos_temp % crop_height;
    const int32_t pos_box_idx = pos_temp / crop_height;
    const G y1 = boxes[4 * pos_box_idx];
    const G x1 = boxes[4 * pos_box_idx + 1];
    const G y2 = boxes[4 * pos_box_idx + 2];
    const G x2 = boxes[4 * pos_box_idx + 3];
    const int32_t box_in_image = box_ind[pos_box_idx];
    if (box_in_image < 0 || box_in_image >= batch) {
      continue;
    }
    const G height_ratio = (crop_height > 1) ? static_cast<G>(image_height - 1) / (crop_height - 1) : 0;
    const G width_ratio = (crop_width > 1) ? static_cast<G>(image_width - 1) / (crop_width - 1) : 0;
    const G height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const G width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;
    const G target_y =
      (crop_height > 1) ? y1 * (image_height - 1) + pos_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    const G target_x =
      (crop_width > 1) ? x1 * (image_width - 1) + pos_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (target_y < 0 || target_y > image_height - 1) {
      continue;
    }
    if (target_x < 0 || target_x > image_width - 1) {
      continue;
    }
    const int32_t top_y_index = floorf(target_y);
    const int32_t bottom_y_index = ceilf(target_y);
    const G y_lerp = target_y - top_y_index;
    const int32_t left_x_ind = floorf(target_x);
    const int32_t right_x_ind = ceilf(target_x);
    const G x_lerp = target_x - left_x_ind;
    const G top_left_value(static_cast<G>(
      *(image + ((box_in_image * image_height + top_y_index) * image_width + left_x_ind) * depth + pos_channel)));
    const G top_right_value(static_cast<G>(
      *(image + ((box_in_image * image_height + top_y_index) * image_width + right_x_ind) * depth + pos_channel)));
    const G bottom_left_value(static_cast<G>(
      *(image + ((box_in_image * image_height + bottom_y_index) * image_width + left_x_ind) * depth + pos_channel)));
    const G bottom_right_value(static_cast<G>(
      *(image + ((box_in_image * image_height + bottom_y_index) * image_width + right_x_ind) * depth + pos_channel)));
    // Compute the image gradient
    G image_ygrad_value =
      (1 - x_lerp) * (bottom_left_value - top_left_value) + x_lerp * (bottom_right_value - top_right_value);
    G image_xgrad_value =
      (1 - y_lerp) * (top_right_value - top_left_value) + y_lerp * (bottom_right_value - bottom_left_value);
    // Modulate the image gradient with the incoming gradient
    const G top_grad = *(grads + pos);
    image_ygrad_value *= top_grad;
    image_xgrad_value *= top_grad;
    G grady1, grady2;
    if (crop_height > 1) {
      grady1 = image_ygrad_value * (image_height - 1 - pos_y * height_ratio);
      grady2 = image_ygrad_value * (pos_y * height_ratio);
    } else {
      grady1 = image_ygrad_value * 0.5 * (image_height - 1);
      grady2 = image_ygrad_value * 0.5 * (image_height - 1);
    }

    G gradx1, gradx2;
    if (crop_width > 1) {
      gradx1 = image_xgrad_value * (image_width - 1 - pos_x * width_ratio);
      gradx2 = image_xgrad_value * (pos_x * width_ratio);
    } else {
      gradx1 = image_xgrad_value * 0.5 * (image_width - 1);
      gradx2 = image_xgrad_value * 0.5 * (image_width - 1);
    }
    MsAtomicAdd(grad_boxes + pos_box_idx * 4, grady1);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 1, gradx1);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 2, grady2);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 3, gradx2);
  }
  return;
}

template <typename T>
__global__ void CropAndResizeGradBoxesForwardKernel(const int32_t size, const float *grads, const T *image,
                                                    const float *boxes, const int *box_ind, int32_t num_boxes,
                                                    int32_t batch, int32_t image_height, int32_t image_width,
                                                    int32_t crop_height, int32_t crop_width, int32_t depth,
                                                    float *grad_boxes) {
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
    const float height_ratio = (crop_height > 1) ? static_cast<float>(image_height - 1) / (crop_height - 1) : 0;
    const float width_ratio = (crop_width > 1) ? static_cast<float>(image_width - 1) / (crop_width - 1) : 0;
    const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;
    const float target_y =
      (crop_height > 1) ? y1 * (image_height - 1) + pos_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    const float target_x =
      (crop_width > 1) ? x1 * (image_width - 1) + pos_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (target_y < 0 || target_y > image_height - 1) {
      continue;
    }
    if (target_x < 0 || target_x > image_width - 1) {
      continue;
    }
    const int32_t top_y_index = floorf(target_y);
    const int32_t bottom_y_index = ceilf(target_y);
    const float y_lerp = target_y - top_y_index;
    const int32_t left_x_ind = floorf(target_x);
    const int32_t right_x_ind = ceilf(target_x);
    const float x_lerp = target_x - left_x_ind;
    const float top_left_value(static_cast<float>(
      *(image + ((box_in_image * image_height + top_y_index) * image_width + left_x_ind) * depth + pos_channel)));
    const float top_right_value(static_cast<float>(
      *(image + ((box_in_image * image_height + top_y_index) * image_width + right_x_ind) * depth + pos_channel)));
    const float bottom_left_value(static_cast<float>(
      *(image + ((box_in_image * image_height + bottom_y_index) * image_width + left_x_ind) * depth + pos_channel)));
    const float bottom_right_value(static_cast<float>(
      *(image + ((box_in_image * image_height + bottom_y_index) * image_width + right_x_ind) * depth + pos_channel)));
    // Compute the image gradient
    float image_ygrad_value =
      (1 - x_lerp) * (bottom_left_value - top_left_value) + x_lerp * (bottom_right_value - top_right_value);
    float image_xgrad_value =
      (1 - y_lerp) * (top_right_value - top_left_value) + y_lerp * (bottom_right_value - bottom_left_value);
    // Modulate the image gradient with the incoming gradient
    const float top_grad = *(grads + pos);
    image_ygrad_value *= top_grad;
    image_xgrad_value *= top_grad;
    float grady1, grady2;
    if (crop_height > 1) {
      grady1 = image_ygrad_value * (image_height - 1 - pos_y * height_ratio);
      grady2 = image_ygrad_value * (pos_y * height_ratio);
    } else {
      grady1 = image_ygrad_value * 0.5 * (image_height - 1);
      grady2 = image_ygrad_value * 0.5 * (image_height - 1);
    }

    float gradx1, gradx2;
    if (crop_width > 1) {
      gradx1 = image_xgrad_value * (image_width - 1 - pos_x * width_ratio);
      gradx2 = image_xgrad_value * (pos_x * width_ratio);
    } else {
      gradx1 = image_xgrad_value * 0.5 * (image_width - 1);
      gradx2 = image_xgrad_value * 0.5 * (image_width - 1);
    }
    MsAtomicAdd(grad_boxes + pos_box_idx * 4, grady1);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 1, gradx1);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 2, grady2);
    MsAtomicAdd(grad_boxes + pos_box_idx * 4 + 3, gradx2);
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
void CalCropAndResizeGradBoxes(const int32_t size, const G *grads, const T *image, const G *boxes, const int *box_ind,
                               int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width,
                               int32_t crop_height, int32_t crop_width, int32_t depth, G *grad_boxes,
                               const uint32_t &device_id, cudaStream_t cuda_stream) {
  Reset_zero<<<CUDA_BLOCKS(device_id, static_cast<int>(num_boxes * 4)), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    static_cast<int>(num_boxes * 4), grad_boxes);
  CropAndResizeGradBoxesForwardKernel<<<CUDA_BLOCKS(device_id, static_cast<int>(size)), CUDA_THREADS(device_id), 0,
                                        cuda_stream>>>(size, grads, image, boxes, box_ind, num_boxes, batch,
                                                       image_height, image_width, crop_height, crop_width, depth,
                                                       grad_boxes);
  return;
}

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int8_t, float>(
  const int32_t size, const float *grads, const int8_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int16_t, float>(
  const int32_t size, const float *grads, const int16_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int32_t, float>(
  const int32_t size, const float *grads, const int32_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int64_t, float>(
  const int32_t size, const float *grads, const int64_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<half, float>(
  const int32_t size, const float *grads, const half *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<float, float>(
  const int32_t size, const float *grads, const float *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<double, float>(
  const int32_t size, const float *grads, const double *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<uint8_t, float>(
  const int32_t size, const float *grads, const uint8_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<uint16_t, float>(
  const int32_t size, const float *grads, const uint16_t *image, const float *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, float *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int8_t, double>(
  const int32_t size, const double *grads, const int8_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int16_t, double>(
  const int32_t size, const double *grads, const int16_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int32_t, double>(
  const int32_t size, const double *grads, const int32_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<int64_t, double>(
  const int32_t size, const double *grads, const int64_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<half, double>(
  const int32_t size, const double *grads, const half *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<float, double>(
  const int32_t size, const double *grads, const float *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<double, double>(
  const int32_t size, const double *grads, const double *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<uint8_t, double>(
  const int32_t size, const double *grads, const uint8_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCropAndResizeGradBoxes<uint16_t, double>(
  const int32_t size, const double *grads, const uint16_t *image, const double *boxes, const int32_t *box_ind,
  int32_t num_boxes, int32_t batch, int32_t image_height, int32_t image_width, int32_t crop_height, int32_t crop_width,
  int32_t depth, double *grad_boxes, const uint32_t &device_id, cudaStream_t cuda_stream);
