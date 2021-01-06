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
#include "dynamic_range_impl.cuh"

#include <cuda_runtime.h>

#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__device__ void CheckInputs(const T &start, const T &end, const T &delta) {
  if (delta == 0) {
    asm("trap;");
  }

  if (start < end && delta < 0) {
    asm("trap;");
  }

  if (start > end && delta > 0) {
    asm("trap;");
  }
}

template <typename T>
__global__ void Range(const T *range_start, const T *range_end, const T *range_delta, T *output,
                             int64_t *output_shape, const int64_t max_output_size) {
  T start = range_start[0];
  T end = range_end[0];
  T delta = range_delta[0];

  CheckInputs(start, end, delta);

  int64_t real_output_shape = static_cast<int64_t>(ceil(static_cast<double>(end - start) / delta));
  if (real_output_shape > max_output_size) {
    asm("trap;");
  }
  *output_shape = real_output_shape;

  size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; gt_id < real_output_shape; gt_id += blockDim.x * gridDim.x) {
    output[gt_id] = gt_id * delta + start;
  }
}

template <typename T>
void CalRange(const T *range_start, const T *range_end, const T *range_delta, T *output, int64_t *output_shape,
                     const int64_t max_output_size, cudaStream_t cuda_stream) {
  Range<<<GET_BLOCKS(max_output_size), GET_THREADS, 0, cuda_stream>>>(range_start, range_end, range_delta,
                                                                             output, output_shape, max_output_size);
}

template void CalRange<int>(const int *range_start, const int *range_end, const int *range_delta, int *output,
                                   int64_t *output_shape, const int64_t max_output_size, cudaStream_t cuda_stream);

template void CalRange<int64_t>(const int64_t *range_start, const int64_t *range_end, const int64_t *range_delta,
                                int64_t *output, int64_t *output_shape, const int64_t max_output_size,
                                cudaStream_t cuda_stream);

template void CalRange<float>(const float *range_start, const float *range_end, const float *range_delta, float *output,
                              int64_t *output_shape, const int64_t max_output_size, cudaStream_t cuda_stream);
template void CalRange<double>(const double *range_start, const double *range_end, const double *range_delta,
                               double *output, int64_t *output_shape, const int64_t max_output_size,
                               cudaStream_t cuda_stream);
