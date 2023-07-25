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

__device__ __forceinline__ bool Isfinite(half x) { return isfinite(static_cast<float>(x)); }

template <typename T>
__device__ __forceinline__ bool Isfinite(T x) {
  return isfinite(x);
}

// Need calculate topk for top_k_output before this function.
template <typename T, typename S>
__global__ void InTopK(const T *predictions, const S *targets, bool *output, const T *top_k_output, size_t batch_size,
                       size_t class_id_count, int64_t k) {
  size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; gt_id < batch_size; gt_id += blockDim.x * gridDim.x) {
    S target_index = targets[gt_id];
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

template <typename T, typename S>
__global__ void InTopKV2(const T *predictions, const S *targets, bool *output, size_t batch_size, size_t class_id_count,
                         int64_t k) {
  size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; gt_id < batch_size; gt_id += blockDim.x * gridDim.x) {
    S target_index = targets[gt_id];
    auto target_pred = predictions[gt_id * class_id_count + target_index];
    bool is_invalid = (static_cast<size_t>(target_index) >= class_id_count) || !Isfinite(target_pred);
    int64_t pos_num = 0;
    if (!is_invalid) {
      for (size_t pos = 0; pos < class_id_count; pos++) {
        auto predicted_value = predictions[gt_id * class_id_count + pos];
        if (!Isfinite(predicted_value)) {
          is_invalid = true;
          break;
        } else if (predicted_value > target_pred) {
          pos_num++;
          if (pos_num > k) {
            break;
          }
        }
      }
    }
    output[gt_id] = is_invalid ? false : pos_num < k;
  }
}

template <typename T, typename S>
cudaError_t CalInTopK(const T *predictions, const S *targets, bool *output, const T *top_k_output, size_t batch_size,
                      size_t class_id_count, int64_t k, cudaStream_t cuda_stream) {
  InTopK<<<GET_BLOCKS(class_id_count), GET_THREADS, 0, cuda_stream>>>(predictions, targets, output, top_k_output,
                                                                      batch_size, class_id_count, k);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t ApplyInTopK(const T *predictions, const S *targets, bool *output, size_t batch_size, size_t class_id_count,
                        int64_t k, uint32_t device_id, cudaStream_t cuda_stream) {
  int block = 256;
  block = CUDA_THREADS_MAXSIZE(device_id, block);
  int grid = ((batch_size - 1) / block) + 1;
  grid = CUDA_BLOCKS_MAXSIZE(device_id, grid);
  InTopKV2<<<grid, block, 0, cuda_stream>>>(predictions, targets, output, batch_size, class_id_count, k);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalInTopK<half, int32_t>(const half *predictions, const int32_t *targets,
                                                              bool *output, const half *top_k_output, size_t batch_size,
                                                              size_t class_id_count, int64_t k,
                                                              cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalInTopK<float, int32_t>(const float *predictions, const int32_t *targets,
                                                               bool *output, const float *top_k_output,
                                                               size_t batch_size, size_t class_id_count, int64_t k,
                                                               cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalInTopK<half, int64_t>(const half *predictions, const int64_t *targets,
                                                              bool *output, const half *top_k_output, size_t batch_size,
                                                              size_t class_id_count, int64_t k,
                                                              cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalInTopK<float, int64_t>(const float *predictions, const int64_t *targets,
                                                               bool *output, const float *top_k_output,
                                                               size_t batch_size, size_t class_id_count, int64_t k,
                                                               cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyInTopK<half, int32_t>(const half *predictions, const int32_t *targets,
                                                                bool *output, size_t batch_size, size_t class_id_count,
                                                                int64_t k, uint32_t device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyInTopK<float, int32_t>(const float *predictions, const int32_t *targets,
                                                                 bool *output, size_t batch_size, size_t class_id_count,
                                                                 int64_t k, uint32_t device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyInTopK<half, int64_t>(const half *predictions, const int64_t *targets,
                                                                bool *output, size_t batch_size, size_t class_id_count,
                                                                int64_t k, uint32_t device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyInTopK<float, int64_t>(const float *predictions, const int64_t *targets,
                                                                 bool *output, size_t batch_size, size_t class_id_count,
                                                                 int64_t k, uint32_t device_id,
                                                                 cudaStream_t cuda_stream);
