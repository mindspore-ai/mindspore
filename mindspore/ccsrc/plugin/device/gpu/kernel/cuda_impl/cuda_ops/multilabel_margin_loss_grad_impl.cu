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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multilabel_margin_loss_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "mindapi/base/types.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void MultilabelMarginLossGradKernel(const T *input_grad, const T *input, const int *target,
                                               const int *is_target, const int batch_size, const int class_num,
                                               const int64_t reduction, T *output_grad) {
  for (int pos = blockIdx.x; pos < batch_size; pos += gridDim.x) {
    const int BLOCK_DIM = 1024;
    const int warp_size = 32;
    __shared__ T smem[BLOCK_DIM];

    int bid = pos;
    T g = reduction == 1 ? 1.0 / (batch_size * class_num) : 1.0 / class_num;

    const T *input_b = input + bid * class_num;
    const int *is_target_b = is_target + bid * class_num;
    const int *target_b = target + bid * class_num;
    T *output_grad_b = output_grad + bid * class_num;

    for (int dt = 0; dt < class_num; dt++) {
      int cidx = target_b[dt];
      if (cidx < 0) {
        break;
      }

      T input_tgt = input_b[cidx];

      T sum = 0;
      for (int ddt = threadIdx.x; ddt < class_num; ddt += blockDim.x) {
        if (is_target_b[ddt] == 0) {
          T z = 1 - input_tgt + input_b[ddt];
          if (z > 0) {
            sum -= 1;
            output_grad_b[ddt] += 1;
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
        output_grad_b[cidx] += totalSum;
      }
    }

    __syncthreads();

    int grad_tid_idx = reduction == 0 ? bid : 0;
    for (int dt = threadIdx.x; dt < class_num; dt += blockDim.x) {
      output_grad_b[dt] *= input_grad[grad_tid_idx] * g;
    }
  }
  return;
}

template <>
__global__ void MultilabelMarginLossGradKernel(const half *input_grad, const half *input, const int *target,
                                               const int *is_target, const int batch_size, const int class_num,
                                               const int64_t reduction, half *output_grad) {
  for (int pos = blockIdx.x; pos < batch_size; pos += gridDim.x) {
    const int BLOCK_DIM = 1024;
    const int warp_size = 32;
    __shared__ float smem[BLOCK_DIM];
    int bid = pos;
    float g = reduction == 1 ? 1.0 / (batch_size * class_num) : 1.0 / class_num;

    const half *input_b = input + bid * class_num;
    const int *is_target_b = is_target + bid * class_num;
    const int *target_b = target + bid * class_num;
    half *output_grad_b = output_grad + bid * class_num;

    for (int dt = 0; dt < class_num; dt++) {
      int cidx = target_b[dt];
      if (cidx < 0) {
        break;
      }
      float input_tgt = __half2float(input_b[cidx]);

      float sum = 0;
      for (int ddt = threadIdx.x; ddt < class_num; ddt += blockDim.x) {
        if (is_target_b[ddt] == 0) {
          float z = 1 - input_tgt + __half2float(input_b[ddt]);
          if (z > 0) {
            sum -= 1;
            output_grad_b[ddt] += 1;
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
        output_grad_b[cidx] += totalSum;
      }
    }

    __syncthreads();

    int grad_tid_idx = reduction == 0 ? bid : 0;
    for (int dt = threadIdx.x; dt < class_num; dt += blockDim.x) {
      output_grad_b[dt] = __float2half(__half2float(output_grad_b[dt]) * __half2float(input_grad[grad_tid_idx]) * g);
    }

    __syncthreads();
  }
  return;
}

template <typename T>
void CalMultilabelMarginLossGrad(const T *input_grad, const T *input, const int *target, int *is_target,
                                 const int batch_size, const int class_num, int64_t reduction, T *output_grad,
                                 const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(batch_size, max_blocks);
  int thread_num = class_num > 1024 ? 1024 : class_num;
  MultilabelMarginLossGradKernel<<<block_num, thread_num, 0, cuda_stream>>>(
    input_grad, input, target, is_target, batch_size, class_num, reduction, output_grad);

  return;
}

template CUDA_LIB_EXPORT void CalMultilabelMarginLossGrad<half>(const half *input_grad, const half *input,
                                                                const int *target, int *is_target, const int batch_size,
                                                                const int class_num, int64_t reduction,
                                                                half *output_grad, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMultilabelMarginLossGrad<float>(const float *input_grad, const float *input,
                                                                 const int *target, int *is_target,
                                                                 const int batch_size, const int class_num,
                                                                 int64_t reduction, float *output_grad,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMultilabelMarginLossGrad<double>(const double *input_grad, const double *input,
                                                                  const int *target, int *is_target,
                                                                  const int batch_size, const int class_num,
                                                                  int64_t reduction, double *output_grad,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
