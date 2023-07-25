/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <thrust/device_vector.h>
#include <algorithm>
#include "unique_impl.cuh"
#include "unique_with_pad_impl.cuh"

template <typename T, typename S>
cudaError_t CalUniqueWithPad(const T *input, int num_elements, S *input_index, S *sorted_index, T *output, S *index,
                             cudaStream_t cuda_stream, T *pad_num) {
  int post_output_size = 0;
  auto ret = CalUnique(input, num_elements, input_index, sorted_index, output, index, cuda_stream, &post_output_size);
  if (ret != cudaSuccess) {
    return ret;
  }
  auto policy = thrust::cuda::par.on(cuda_stream);

  if (num_elements > post_output_size) {
    thrust::device_reference<T> pad_ref(thrust::device_pointer_cast(pad_num));
    thrust::fill_n(policy, thrust::device_pointer_cast(output) + post_output_size, num_elements - post_output_size,
                   pad_ref);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUniqueWithPad<float, int>(const float *input, int num_elements,
                                                                  int *input_index, int *sorted_index, float *output,
                                                                  int *index, cudaStream_t cuda_stream, float *pad_num);
template CUDA_LIB_EXPORT cudaError_t CalUniqueWithPad<half, int>(const half *input, int num_elements, int *input_index,
                                                                 int *sorted_index, half *output, int *index,
                                                                 cudaStream_t cuda_stream, half *pad_num);
template CUDA_LIB_EXPORT cudaError_t CalUniqueWithPad<int, int>(const int *input, int num_elements, int *input_index,
                                                                int *sorted_index, int *output, int *index,
                                                                cudaStream_t cuda_stream, int *pad_num);
template CUDA_LIB_EXPORT cudaError_t CalUniqueWithPad<int64_t, int64_t>(const int64_t *input, int num_elements,
                                                                        int64_t *input_index, int64_t *sorted_index,
                                                                        int64_t *output, int64_t *index,
                                                                        cudaStream_t cuda_stream, int64_t *pad_num);
