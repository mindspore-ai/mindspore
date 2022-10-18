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

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>
#include "unique_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename S>
int CalUnique(const T *input, int num_elements, S *input_index, S *sorted_index, T *output, S *index,
               cudaStream_t cuda_stream) {
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::sequence(policy,
                   thrust::device_pointer_cast(sorted_index),
                   thrust::device_pointer_cast(sorted_index) + num_elements);
  thrust::copy(thrust::device_pointer_cast(input),
               thrust::device_pointer_cast(input) + num_elements,
               thrust::device_pointer_cast(output));
  thrust::stable_sort_by_key(policy,
                             thrust::device_pointer_cast(output),
                             thrust::device_pointer_cast(output) + num_elements,
                             thrust::device_pointer_cast(sorted_index));
  thrust::adjacent_difference(policy,
                              thrust::device_pointer_cast(output),
                              thrust::device_pointer_cast(output) + num_elements,
                              thrust::device_pointer_cast(input_index),
                              thrust::not_equal_to<T>());
  thrust::fill(policy,
               thrust::device_pointer_cast(input_index),
               thrust::device_pointer_cast(input_index) + 1,
               0);
  thrust::inclusive_scan(policy,
                         thrust::device_pointer_cast(input_index),
                         thrust::device_pointer_cast(input_index) + num_elements,
                         thrust::device_pointer_cast(input_index));
  thrust::scatter(policy,
                  thrust::device_pointer_cast(input_index),
                  thrust::device_pointer_cast(input_index) + num_elements,
                  thrust::device_pointer_cast(sorted_index),
                  thrust::device_pointer_cast(index));
  thrust::device_ptr<T> output_end;
  output_end = thrust::unique(policy,
                              thrust::device_pointer_cast(output),
                              thrust::device_pointer_cast(output) + num_elements);
  int output_size = thrust::distance(thrust::device_pointer_cast(output), output_end);
  return output_size;
}

template CUDA_LIB_EXPORT int CalUnique<float, int>(const float *input, int num_elements, int *input_index,
                                                   int *sorted_index, float *output, int *index,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<half, int>(const half *input, int num_elements, int *input_index,
                                                  int *sorted_index, half *output, int *index,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<double, int>(const double *input, int num_elements, int *input_index,
                                                    int *sorted_index, double *output, int *index,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<int, int>(const int *input, int num_elements, int *input_index,
                                                 int *sorted_index, int *output, int *index, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<int8_t, int>(const int8_t *input, int num_elements, int *input_index,
                                                    int *sorted_index, int8_t *output, int *index,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<int16_t, int>(const int16_t *input, int num_elements, int *input_index,
                                                     int *sorted_index, int16_t *output, int *index,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<uint8_t, int>(const uint8_t *input, int num_elements, int *input_index,
                                                     int *sorted_index, uint8_t *output, int *index,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<uint16_t, int>(const uint16_t *input, int num_elements, int *input_index,
                                                      int *sorted_index, uint16_t *output, int *index,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int CalUnique<int64_t, int64_t>(const int64_t *input, int num_elements, int64_t *input_index,
                                                         int64_t *sorted_index, int64_t *output, int64_t *index,
                                                         cudaStream_t cuda_stream);
