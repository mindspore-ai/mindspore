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

#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multilabel_margin_loss_impl.cuh"
#include "include/cuda_fp16.h"
#include "mindapi/base/types.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void PreHandleKernel(const int *target, int *is_target, const int batch_size, const int class_num,
                                T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < batch_size; pos += gridDim.x * blockDim.x) {
    // zero output
    T zero = static_cast<T>(0);
    output[pos] = zero;
    int *is_target_b = is_target + pos * class_num;
    const int *target_b = target + pos * class_num;
    // mark is_target
    for (int dt = 0; dt < class_num; dt++) {
      int cidx = target_b[dt];
      if (cidx < 0) {
        break;
      } else {
        is_target_b[cidx] = 1;
      }
    }
  }
}

template <typename T>
__global__ void MultilabelMarginLossKernel(const T *input, const int *target, int *is_target, const int batch_size,
                                           int class_num, int64_t reduction, T *output) {
  for (int pos = blockIdx.x; pos < batch_size; pos += gridDim.x) {
    const int BLOCK_DIM = 1024;
    const int warp_size = 32;
    __shared__ T smem[BLOCK_DIM];

    // get bid and address
    int bid = pos;
    int *is_target_b = is_target + bid * class_num;
    const int *target_b = target + bid * class_num;
    const T *input_b = input + bid * class_num;
    T *output_b = output + bid;

    // calculate loss
    T sum = 0;
    for (int dt = 0; dt < class_num; dt++) {
      int cidx = target_b[dt];
      if (cidx < 0) {
        break;
      }
      T input_tgt = input_b[cidx];
      for (int ddt = threadIdx.x; ddt < class_num; ddt += blockDim.x) {
        if (is_target_b[ddt] == 0) {
          T z = 1 - input_tgt + input_b[ddt];
          if (z > 0) {
            sum += z;
          }
        }
      }
    }

    T totalSum = 0;
    smem[threadIdx.x] = sum;
    __syncthreads();
    int num_warp = blockDim.x > warp_size ? warp_size : blockDim.x;
    if (threadIdx.x < num_warp) {
      for (int dw = num_warp + threadIdx.x; dw < blockDim.x; dw += num_warp) {
        smem[threadIdx.x] += smem[dw];
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
      for (int dw = 0; dw < num_warp; dw++) {
        totalSum += smem[dw];
      }
      if (reduction == 0 || reduction == 2) {
        totalSum /= class_num;
      } else {
        totalSum /= (class_num * batch_size);
      }
      *output_b = totalSum;
    }
  }
  return;
}

template <typename T>
__global__ void MultilabelMarginLossSumKernel(const T *input, const int *target, int *is_target, const int batch_size,
                                              int class_num, int64_t reduction, T *output, T *output_tmp) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < 1; pos += gridDim.x * blockDim.x) {
    T totalSum = 0;
#pragma unroll
    for (int dt = 0; dt < batch_size; dt++) {
      totalSum += output_tmp[dt];
    }
    *output = totalSum;
  }
  return;
}

template <>
__global__ void MultilabelMarginLossKernel(const half *input, const int *target, int *is_target, const int batch_size,
                                           int class_num, int64_t reduction, half *output) {
  for (int pos = blockIdx.x; pos < batch_size; pos += gridDim.x) {
    const int BLOCK_DIM = 1024;
    const int warp_size = 32;
    __shared__ float smem[BLOCK_DIM];

    // get bid and address
    int bid = pos;
    int *is_target_b = is_target + bid * class_num;
    const int *target_b = target + bid * class_num;
    const half *input_b = input + bid * class_num;
    half *output_b = output + bid;

    // calculate loss
    float sum = 0;
    for (int dt = 0; dt < class_num; dt++) {
      int cidx = target_b[dt];
      if (cidx < 0) {
        break;
      }
      float input_tgt = __half2float(input_b[cidx]);
      for (int ddt = threadIdx.x; ddt < class_num; ddt += blockDim.x) {
        if (is_target_b[ddt] == 0) {
          float z = 1 - input_tgt + __half2float(input_b[ddt]);
          if (z > 0) {
            sum += z;
          }
        }
      }
    }

    float totalSum = 0;
    smem[threadIdx.x] = sum;
    __syncthreads();
    int num_warp = blockDim.x > warp_size ? warp_size : blockDim.x;

    if (threadIdx.x < num_warp) {
      for (int dw = num_warp + threadIdx.x; dw < blockDim.x; dw += num_warp) {
        smem[threadIdx.x] += smem[dw];
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
      for (int dw = 0; dw < num_warp; dw++) {
        totalSum += smem[dw];
      }
      if (reduction == 0 || reduction == 2) {
        totalSum /= class_num;
      } else {
        totalSum /= (class_num * batch_size);
      }
      *output_b = __float2half(totalSum);
    }
  }
  return;
}

template <>
__global__ void MultilabelMarginLossSumKernel(const half *input, const int *target, int *is_target,
                                              const int batch_size, int class_num, int64_t reduction, half *output,
                                              half *output_tmp) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < 1; pos += gridDim.x * blockDim.x) {
    float totalSum = 0;
#pragma unroll
    for (int dt = 0; dt < batch_size; dt++) {
      totalSum += __half2float(output_tmp[dt]);
    }
    *output = __float2half(totalSum);
  }
  return;
}

template <typename T>
void CalMultilabelMarginLoss(const T *input, const int *target, int *is_target, const int batch_size, int class_num,
                             int64_t reduction, T *output, T *output_tmp, const uint32_t &device_id,
                             cudaStream_t cuda_stream) {
  cudaMemset(is_target, 0, sizeof(int) * batch_size * class_num);
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(batch_size, max_blocks);
  int thread_num = class_num > 1024 ? 1024 : class_num;
  PreHandleKernel<<<block_num, thread_num, 0, cuda_stream>>>(target, is_target, batch_size, class_num, output);
  if (reduction == 0) {
    MultilabelMarginLossKernel<<<block_num, thread_num, 0, cuda_stream>>>(input, target, is_target, batch_size,
                                                                          class_num, reduction, output);
  } else {
    MultilabelMarginLossKernel<<<block_num, thread_num, 0, cuda_stream>>>(input, target, is_target, batch_size,
                                                                          class_num, reduction, output_tmp);
    MultilabelMarginLossSumKernel<<<1, 1, 0, cuda_stream>>>(input, target, is_target, batch_size, class_num, reduction,
                                                            output, output_tmp);
  }

  return;
}

template CUDA_LIB_EXPORT void CalMultilabelMarginLoss<half>(const half *input, const int *target, int *is_target,
                                                            const int batch_size, int class_num, int64_t reduction,
                                                            half *output, half *output_tmp, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMultilabelMarginLoss<float>(const float *input, const int *target, int *is_target,
                                                             const int batch_size, int class_num, int64_t reduction,
                                                             float *output, float *output_tmp,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMultilabelMarginLoss<double>(const double *input, const int *target, int *is_target,
                                                              const int batch_size, int class_num, int64_t reduction,
                                                              double *output, double *output_tmp,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
