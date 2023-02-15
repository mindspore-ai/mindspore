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
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/coalesce_impl.cuh"
#include "include/cuda_fp16.h"

__global__ void FlattenIndicesKernel(int64_t *flatten_input_indices, const size_t indices_num, const size_t values_num,
                                     const int64_t *input_indices, const int64_t *input_shape) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    int64_t temp = 0;
    int64_t temp2 = 0;
    if (pos < values_num) {
      for (int x = 0; x < indices_num; x++) {
        if (x != indices_num - 1) {
          temp2 = input_indices[pos + (x * values_num)];
          for (int j = (x + 1); j < indices_num; j++) {
            temp2 *= input_shape[j];
          }
          temp += temp2;
          temp2 = 0;
        } else {
          temp += input_indices[pos + (x * values_num)];
        }
      }
      flatten_input_indices[pos] = temp;
    }
  }
}

template <typename T>
__global__ void CoalesceKernel(int64_t *origin_indices, int64_t newNnz, int64_t *unique_indices,
                               const size_t indices_num, const size_t values_num, const int64_t *input_indices,
                               const T *input_values, int64_t *output_indices, T *output_value) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < indices_num * values_num;
       pos += blockDim.x * gridDim.x) {
    if (pos < newNnz) {
      output_value[pos] = 0;
      const int begin = unique_indices[pos];
      const int end = (pos < newNnz - 1) ? unique_indices[pos + 1] : values_num;
      for (int row = begin; row < end; row++) {
        output_value[pos] += input_values[origin_indices[row]];
      }
      output_indices[pos] = input_indices[origin_indices[unique_indices[pos]]];
    } else if (pos < (newNnz * 2)) {
      for (int x = 0; x < indices_num; x++) {
        output_indices[(pos - newNnz) + (x * newNnz)] =
          input_indices[origin_indices[unique_indices[pos - newNnz]] + x * values_num];
      }
    }
  }
}

__global__ void CoalesceKernelCheck(const int64_t *indices, const int64_t *input_shape, const size_t indices_num,
                                    size_t values_num, int *ret_flag) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indices_num * values_num; i += gridDim.x * blockDim.x) {
    if (indices[i] < 0) {
      *ret_flag = 1;
      return;
    }
    int shape_pos = i / values_num;
    if (input_shape[shape_pos] <= 0) {
      *ret_flag = 2;
      return;
    }
    if (indices[i] >= input_shape[shape_pos]) {
      *ret_flag = 3;
      return;
    }
  }
}

template <typename T>
int Coalesce(int64_t *origin_indices, int64_t *unique_indices, const size_t shape_elements, const size_t indices_num,
             const size_t values_num, int *ret_flag_host, int64_t *flatten_input_indices, const int64_t *input_indices,
             const T *input_values, const int64_t *input_shape, int64_t *output_indices, T *output_value,
             int64_t *output_shape, const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t allelement = indices_num * values_num;
  int *ret_flag_device = nullptr;
  (void)cudaMalloc(&ret_flag_device, sizeof(int));
  (void)cudaMemset(ret_flag_device, 0, sizeof(int));
  CoalesceKernelCheck<<<CUDA_BLOCKS(device_id, allelement), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_indices, input_shape, indices_num, values_num, ret_flag_device);
  (void)cudaMemcpy(ret_flag_host, ret_flag_device, sizeof(int), cudaMemcpyDeviceToHost);
  (void)cudaFree(ret_flag_device);
  if (*ret_flag_host != 0) {
    return -1;
  }
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::copy(thrust::device_pointer_cast(input_shape), thrust::device_pointer_cast(input_shape) + shape_elements,
               thrust::device_pointer_cast(output_shape));
  FlattenIndicesKernel<<<CUDA_BLOCKS(device_id, values_num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    flatten_input_indices, indices_num, values_num, input_indices, input_shape);
  thrust::counting_iterator<int64_t> countIterO(0);
  thrust::counting_iterator<int64_t> countIterI(0);

  thrust::copy(policy, countIterI, countIterI + values_num, origin_indices);
  thrust::sort_by_key(policy, thrust::device_pointer_cast(flatten_input_indices),
                      thrust::device_pointer_cast(flatten_input_indices) + values_num,
                      thrust::device_pointer_cast(origin_indices));
  thrust::copy(policy, countIterO, countIterO + values_num, unique_indices);
  thrust::pair<thrust::device_ptr<int64_t>, thrust::device_ptr<int64_t>> newEnd;
  newEnd = thrust::unique_by_key(policy, thrust::device_pointer_cast(flatten_input_indices),
                                 thrust::device_pointer_cast(flatten_input_indices) + values_num,
                                 thrust::device_pointer_cast(unique_indices));
  int64_t newNnz = newEnd.first - thrust::device_pointer_cast(flatten_input_indices);
  CoalesceKernel<<<CUDA_BLOCKS(device_id, allelement), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    origin_indices, newNnz, unique_indices, indices_num, values_num, input_indices, input_values, output_indices,
    output_value);
  int output_size = newNnz;
  return output_size;
}

template CUDA_LIB_EXPORT int Coalesce<float>(int64_t *origin_indices, int64_t *unique_indices,
                                             const size_t shape_elements, const size_t indices_num,
                                             const size_t values_num, int *ret_flag_host,
                                             int64_t *flatten_input_indices, const int64_t *input_indices,
                                             const float *input_values, const int64_t *input_shape,
                                             int64_t *output_indices, float *output_value, int64_t *output_shape,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int Coalesce<half>(int64_t *origin_indices, int64_t *unique_indices,
                                            const size_t shape_elements, const size_t indices_num,
                                            const size_t values_num, int *ret_flag_host, int64_t *flatten_input_indices,
                                            const int64_t *input_indices, const half *input_values,
                                            const int64_t *input_shape, int64_t *output_indices, half *output_value,
                                            int64_t *output_shape, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int Coalesce<double>(int64_t *origin_indices, int64_t *unique_indices,
                                              const size_t shape_elements, const size_t indices_num,
                                              const size_t values_num, int *ret_flag_host,
                                              int64_t *flatten_input_indices, const int64_t *input_indices,
                                              const double *input_values, const int64_t *input_shape,
                                              int64_t *output_indices, double *output_value, int64_t *output_shape,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);
