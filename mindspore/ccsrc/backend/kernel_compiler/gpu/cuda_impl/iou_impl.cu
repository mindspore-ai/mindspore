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

#include "backend/kernel_compiler/gpu/cuda_impl/iou_impl.cuh"

template <typename T>
__device__ T CoordinateMax(const T a, const T b) {
  return (a > b ? a : b);
}

template <typename T>
__device__ T CoordinateMin(const T a, const T b) {
  return (a < b ? a : b);
}

template <typename T>
__global__ void IOUKernel(const size_t size, const T *box1, const T *box2, T *iou_results, const size_t mode,
                          const size_t input_len_0) {
  T location_coordinate[IOU_LOCATION_NUM][IOU_DIMENSION];
  T overlaps_coordinate[IOU_DIMENSION];
  const T epsilon = 1e-10;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    for (size_t j = 0; j < IOU_DIMENSION; j++) {
      location_coordinate[0][j] = box1[(i % input_len_0) * IOU_DIMENSION + j];
      location_coordinate[1][j] = box2[(i / input_len_0) * IOU_DIMENSION + j];
    }

    overlaps_coordinate[0] = CoordinateMax(location_coordinate[0][0], location_coordinate[1][0]);
    overlaps_coordinate[1] = CoordinateMax(location_coordinate[0][1], location_coordinate[1][1]);
    overlaps_coordinate[2] = CoordinateMin(location_coordinate[0][2], location_coordinate[1][2]);
    overlaps_coordinate[3] = CoordinateMin(location_coordinate[0][3], location_coordinate[1][3]);

    T overlaps_w = CoordinateMax(0.f, overlaps_coordinate[2] - overlaps_coordinate[0] + 1);
    T overlaps_h = CoordinateMax(0.f, overlaps_coordinate[3] - overlaps_coordinate[1] + 1);
    T overlaps = overlaps_w * overlaps_h;

    T area1 = (location_coordinate[0][2] - location_coordinate[0][0] + 1) * (location_coordinate[0][3] -
               location_coordinate[0][1] + 1);
    T area2 = (location_coordinate[1][2] - location_coordinate[1][0] + 1) * (location_coordinate[1][3] -
                                                                             location_coordinate[1][1] + 1);
    if (mode == 0) {
      iou_results[i] = overlaps / (area1 + area2 - overlaps + epsilon);
    } else {
      iou_results[i] = overlaps / (area2 + epsilon);
    }
  }

  return;
}

template <typename T>
void IOU(const size_t &size, const T *box1, const T *box2, T *iou_results, const size_t &mode,
         const size_t &input_len_0, cudaStream_t cuda_stream) {
  IOUKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, box1, box2, iou_results, mode, input_len_0);
}

template void IOU(const size_t &size, const float *box1, const float *box2, float *iou_results, const size_t &mode,
                  const size_t &input_len_0, cudaStream_t cuda_stream);
