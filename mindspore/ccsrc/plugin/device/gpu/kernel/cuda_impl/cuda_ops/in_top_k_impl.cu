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
#include "in_top_k_impl.cuh"
#include <cuda_runtime.h>
#include "include/cuda_fp16.h"

__device__ __forceinline__ bool Isfinite(half x) {
  return isfinite(static_cast<float>(x));
}

template <typename T>
__device__ __forceinline__ bool Isfinite(T x) {
  return isfinite(x);
}

template <typename T>
__global__ void InTopK(const T *predictions, const int32_t *targets, bool *output, const T *top_k_output,
                       size_t batch_size, size_t class_id_count, int64_t k) {
  size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; gt_id < batch_size; gt_id += blockDim.x * gridDim.x) {
    int32_t target_index = targets[gt_id];
    bool is_invalid = (static_cast<size_t>(target_index) >= class_id_count);
    if (!is_invalid) {
      T predicted_value = predictions[gt_id * class_id_count + target_index];
      T top_k_smallest_value = top_k_output[gt_id * k + k - 1];
      is_invalid = is_invalid || !Isfinite(predicted_value);
      output[gt_id] = is_invalid ? false : predicted_value >= top_k_smallest_value;
    } else {
      output[gt_id] = false;
    }
  }
}

template <typename T>
void CalInTopK(const T *predictions, const int32_t *targets, bool *output, const T *top_k_output, size_t batch_size,
               size_t class_id_count, int64_t k, cudaStream_t cuda_stream) {
  InTopK<<<GET_BLOCKS(class_id_count), GET_THREADS, 0, cuda_stream>>>(predictions, targets, output, top_k_output,
                                                                      batch_size, class_id_count, k);
}

template CUDA_LIB_EXPORT void CalInTopK<half>(const half *predictions, const int32_t *targets, bool *output,
                                              const half *top_k_output, size_t batch_size, size_t class_id_count,
                                              int64_t k, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInTopK<float>(const float *predictions, const int32_t *targets, bool *output,
                                               const float *top_k_output, size_t batch_size, size_t class_id_count,
                                               int64_t k, cudaStream_t cuda_stream);
