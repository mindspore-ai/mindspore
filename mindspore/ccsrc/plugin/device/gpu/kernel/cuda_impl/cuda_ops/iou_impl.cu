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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/iou_impl.cuh"
#include "include/cuda_fp16.h"

__device__ float CoordinateMax(const float a, const float b) { return (a > b ? a : b); }

__device__ float CoordinateMin(const float a, const float b) { return (a < b ? a : b); }

template <typename T>
__global__ void IOUKernel(const size_t size, const T *box1, const T *box2, T *iou_results, const size_t mode,
                          const size_t input_len_0) {
  float location_coordinate[IOU_LOCATION_NUM][IOU_DIMENSION];
  float overlaps_coordinate[IOU_DIMENSION];
  const float epsilon = 1e-10;
  const float offset = 1.0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    for (size_t j = 0; j < IOU_DIMENSION; j++) {
      location_coordinate[0][j] = static_cast<float>(box1[(i % input_len_0) * IOU_DIMENSION + j]);
      location_coordinate[1][j] = static_cast<float>(box2[(i / input_len_0) * IOU_DIMENSION + j]);
    }

    overlaps_coordinate[0] = CoordinateMax(location_coordinate[0][0], location_coordinate[1][0]);
    overlaps_coordinate[1] = CoordinateMax(location_coordinate[0][1], location_coordinate[1][1]);
    overlaps_coordinate[2] = CoordinateMin(location_coordinate[0][2], location_coordinate[1][2]);
    overlaps_coordinate[3] = CoordinateMin(location_coordinate[0][3], location_coordinate[1][3]);

    float overlaps_w = CoordinateMax(0.0, overlaps_coordinate[2] - overlaps_coordinate[0] + offset);
    float overlaps_h = CoordinateMax(0.0, overlaps_coordinate[3] - overlaps_coordinate[1] + offset);
    float overlaps = overlaps_w * overlaps_h;

    float area1 = (location_coordinate[0][2] - location_coordinate[0][0] + offset) *
                  (location_coordinate[0][3] - location_coordinate[0][1] + offset);
    float area2 = (location_coordinate[1][2] - location_coordinate[1][0] + offset) *
                  (location_coordinate[1][3] - location_coordinate[1][1] + offset);
    if (mode == 0) {
      iou_results[i] = static_cast<T>(overlaps / (area1 + area2 - overlaps + epsilon));
    } else {
      iou_results[i] = static_cast<T>(overlaps / (area2 + epsilon));
    }
  }

  return;
}

template <typename T>
void IOU(const size_t &size, const T *box1, const T *box2, T *iou_results, const size_t &mode,
         const size_t &input_len_0, cudaStream_t cuda_stream) {
  IOUKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, box1, box2, iou_results, mode, input_len_0);
}

template CUDA_LIB_EXPORT void IOU(const size_t &size, const half *box1, const half *box2, half *iou_results,
                                  const size_t &mode, const size_t &input_len_0, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IOU(const size_t &size, const float *box1, const float *box2, float *iou_results,
                                  const size_t &mode, const size_t &input_len_0, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IOU(const size_t &size, const double *box1, const double *box2, double *iou_results,
                                  const size_t &mode, const size_t &input_len_0, cudaStream_t cuda_stream);
