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

#include "ctcgreedydecoder_impl.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
template <typename T>
__global__ void CTCGreedyDecoder(const T *input, const int bound, const size_t outer_size, const size_t batch_size,
                                 int64_t *decoded_values_temp, T *log_probability) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += gridDim.x * blockDim.x) {
    int idx = 0;
    size_t input_offset = pos * bound;
    T max_data = input[input_offset];
    for (int i = 1; i < bound; i++) {
      input_offset = pos * bound + i;
      auto input_data = input[input_offset];
      if (input_data > max_data) {
        idx = i;
        max_data = input_data;
      }
    }
    decoded_values_temp[pos] = idx;
    log_probability[pos] = -max_data;
  }
  return;
}

template <typename T>
__global__ void values_merge(int64_t *decoded_values_temp, const int32_t *sequence_length, const size_t batch_size,
                             const int bound, const bool merge_ok, T *log_probability, int64_t *nums_count) {
  const int blank_idx = bound - 1;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < batch_size; pos += gridDim.x * blockDim.x) {
    if (sequence_length[pos] <= 0) {
      nums_count[pos] = 0;
      log_probability[pos] = 0;
      nums_count[pos] = 0;
      return;
    }
    size_t cnt = 0;
    for (size_t i = 0, idx = pos; i < sequence_length[pos]; i++, idx += batch_size) {
      if (idx != pos) {
        log_probability[pos] += log_probability[idx];
      }
      if (decoded_values_temp[idx] == blank_idx ||
          merge_ok && idx != pos && decoded_values_temp[idx] == decoded_values_temp[idx - batch_size]) {
        continue;
      }
      decoded_values_temp[cnt * batch_size + pos] = decoded_values_temp[idx];
      cnt++;
    }
    nums_count[pos] = cnt;
  }
  return;
}

__global__ void indicesCompute(const int64_t *decoded_values_temp, const int64_t *nums_count, const size_t batch_size,
                               int64_t *decoded_indices, int64_t *decoded_values, int64_t *decoded_shape,
                               int64_t *nums_count_pre_sum) {
  for (size_t batch_pos = blockIdx.y * blockDim.y + threadIdx.y; batch_pos < batch_size;
       batch_pos += gridDim.y * blockDim.y) {
    for (size_t nums_count_pos = threadIdx.x; nums_count_pos < nums_count[batch_pos]; nums_count_pos += blockDim.x) {
      decoded_indices[(nums_count_pre_sum[batch_pos] + nums_count_pos) * 2] = batch_pos;
      decoded_indices[(nums_count_pre_sum[batch_pos] + nums_count_pos) * 2 + 1] = nums_count_pos;
      decoded_values[nums_count_pre_sum[batch_pos] + nums_count_pos] =
        decoded_values_temp[nums_count_pos * batch_size + batch_pos];
    }
    if (threadIdx.x == 0) {
      MsAtomicMax(decoded_shape + 1, nums_count[batch_pos]);
    }
  }
  decoded_shape[0] = batch_size;
}

template <typename T>
cudaError_t CalCTCGreedyDecoder(const T *input, const int bound, const size_t outer_size, const size_t batch_size,
                                int64_t *decoded_values_temp, T *log_probability, const uint32_t &device_id,
                                cudaStream_t cuda_stream) {
  CTCGreedyDecoder<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, bound, outer_size, batch_size, decoded_values_temp, log_probability);
  return GetCudaStatus();
}

template <typename T>
cudaError_t Calmerge(int64_t *decoded_values_temp, const int32_t *sequence_length, const size_t batch_size,
                     const int bound, const bool merge_ok, T *log_probability, int64_t *nums_count,
                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  values_merge<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    decoded_values_temp, sequence_length, batch_size, bound, merge_ok, log_probability, nums_count);
  return GetCudaStatus();
}

cudaError_t Calindices(const int64_t *decoded_values_temp, const int64_t *nums_count, const size_t batch_size,
                   int64_t *decoded_indices, int64_t *decoded_values, int64_t *decoded_shape, const uint32_t &device_id,
                   cudaStream_t cuda_stream, int64_t *count) {
  size_t temp_storage_bytes = 0;
  int64_t *nums_count_pre_sum = nullptr;
  cudaMalloc(&nums_count_pre_sum, sizeof(int64_t) * (batch_size + 1));
  cudaMemset(nums_count_pre_sum, 0, sizeof(int64_t) * (batch_size + 1));
  cudaMemset(decoded_shape, 0, sizeof(int64_t) * 2);
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, nums_count, nums_count_pre_sum + 1,
                                      static_cast<int64_t>(batch_size), cuda_stream);
  void *d_temp_storage = nullptr;
  cudaStreamSynchronize(cuda_stream);
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nums_count, nums_count_pre_sum + 1,
                                      static_cast<int64_t>(batch_size), cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  (void)cudaFree(d_temp_storage);

  int64_t sum_num_count = 0;
  cudaMemcpy(&sum_num_count, nums_count_pre_sum + batch_size, sizeof(int64_t), cudaMemcpyDeviceToHost);
  int64_t avg_num_count = sum_num_count / batch_size == 0 ? 1 : sum_num_count / batch_size;
  size_t thread_x_num = avg_num_count > 32 ? 32 : avg_num_count;
  size_t thread_y_num = 512 / thread_x_num;

  dim3 thread_num(thread_x_num, thread_y_num);
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num =
    min(static_cast<int>(((avg_num_count * batch_size - 1) / (thread_x_num * thread_y_num)) + 1), max_blocks);

  indicesCompute<<<block_num, thread_num, 0, cuda_stream>>>(
    decoded_values_temp, nums_count, batch_size, decoded_indices, decoded_values, decoded_shape, nums_count_pre_sum);
  cudaFree(nums_count_pre_sum);
  *count = sum_num_count;
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCTCGreedyDecoder<float>(const float *input, const int bound,
                                                                const size_t outer_size, const size_t batch_size,
                                                                int64_t *decoded_values_temp, float *log_probability,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCTCGreedyDecoder<double>(const double *input, const int bound,
                                                                 const size_t outer_size, const size_t batch_size,
                                                                 int64_t *decoded_values_temp, double *log_probability,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t Calmerge<float>(int64_t *decoded_values_temp, const int32_t *sequence_length,
                                                     const size_t batch_size, const int bound, const bool merge_ok,
                                                     float *log_probability, int64_t *nums_count,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t Calmerge<double>(int64_t *decoded_values_temp, const int32_t *sequence_length,
                                                      const size_t batch_size, const int bound, const bool merge_ok,
                                                      double *log_probability, int64_t *nums_count,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
