/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/resize_nearest_neighbor_impl.cuh"

template <typename T>
__global__ void ResizeNearestNeighbor(const int size, const T *input, const int s1, const int s2, const int s3,
                                      const int s4, T *output, const int d1, const int d2, const int d3, const int d4,
                                      bool align_corners, float h_scale, float w_scale) {
  // initialization
  // HalfPixelCenters false
  int input_pos;
  int pos_array[RESIZENEARESTNEIGHBOR_DIMENSION];
  int in_height = s3;
  int in_width = s4;
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
  int out_h;
  int out_w;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    pos_array[0] = pos / (d2 * d3 * d4) % d1;
    pos_array[1] = pos / (d3 * d4) % d2;
    pos_array[2] = pos / (d4) % d3;
    pos_array[3] = pos % d4;
    out_h = pos_array[2];
    out_w = pos_array[3];
    const int in_y =
      min((align_corners) ? static_cast<int>(roundf(out_h * h_scale)) : static_cast<int>(floorf(out_h * h_scale)),
          in_height - 1);
    const int in_x =
      min((align_corners) ? static_cast<int>(roundf(out_w * w_scale)) : static_cast<int>(floorf(out_w * w_scale)),
          in_width - 1);
    // pos_array[0] N, pos_array[1] C, in_y H, in_x W
    input_pos = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + in_y * s4 + in_x;
    output[pos] = input[input_pos];
  }
  return;
}

template <typename T>
void CalResizeNearestNeighbor(const int size, const T *input, const int s1, const int s2, const int s3, const int s4,
                              T *output, const int d1, const int d2, const int d3, const int d4, bool align_corners,
                              float h_scale, float w_scale, cudaStream_t cuda_stream) {
  ResizeNearestNeighbor<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, s1, s2, s3, s4, output, d1, d2,
                                                                           d3, d4, align_corners, h_scale, w_scale);
  return;
}

template void CalResizeNearestNeighbor<float>(const int size, const float *input, const int s1, const int s2,
                                              const int s3, const int s4, float *output, const int d1, const int d2,
                                              const int d3, const int d4, bool align_corners, float h_scale,
                                              float w_scale, cudaStream_t cuda_stream);
template void CalResizeNearestNeighbor<half>(const int size, const half *input, const int s1, const int s2,
                                             const int s3, const int s4, half *output, const int d1, const int d2,
                                             const int d3, const int d4, bool align_corners, float h_scale,
                                             float w_scale, cudaStream_t cuda_stream);
template void CalResizeNearestNeighbor<int>(const int size, const int *input, const int s1, const int s2, const int s3,
                                            const int s4, int *output, const int d1, const int d2, const int d3,
                                            const int d4, bool align_corners, float h_scale, float w_scale,
                                            cudaStream_t cuda_stream);
