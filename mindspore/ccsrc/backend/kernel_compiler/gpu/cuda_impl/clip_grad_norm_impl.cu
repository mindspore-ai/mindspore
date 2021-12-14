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

#include "backend/kernel_compiler/gpu/cuda_impl/clip_grad_norm_impl.cuh"

// The implement of ScalingGradOp
template <typename T>
__global__ void ScalingGradKernel(const size_t size, const T *x, const float *scaling_factor, float *scaling_out_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    scaling_out_addr[i] = x[i] * (1.0 / scaling_factor[0]);
  }
}

template <>
__global__ void ScalingGradKernel(const size_t size, const half *x, const float *scaling_factor,
                                  float *scaling_out_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    scaling_out_addr[i] = __half2float(x[i]) * (1.0 / scaling_factor[0]);
  }
}

template <typename T>
void ScalingGradOp(const size_t size, const T *x, const float *scaling_factor, float *scaling_out_addr,
                   cudaStream_t cuda_stream) {
  ScalingGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, x, scaling_factor, scaling_out_addr);
}

template void ScalingGradOp<float>(const size_t size, const float *x, const float *scaling_factor,
                                   float *scaling_out_addr, cudaStream_t cuda_stream);

template void ScalingGradOp<half>(const size_t size, const half *x, const float *scaling_factor,
                                  float *scaling_out_addr, cudaStream_t cuda_stream);

// The implement of ClipGradNormOp
template <typename T>
__global__ void ClipGradNormKernel(const size_t size, const float *x, const T *clip_norm, const float *reduce_sum_value,
                                   float *output_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (reduce_sum_value[0] > clip_norm[0]) {
      output_addr[i] = x[i] * clip_norm[0] / reduce_sum_value[0];
    } else {
      output_addr[i] = x[i];
    }
  }
}

template <>
__global__ void ClipGradNormKernel(const size_t size, const float *x, const half *clip_norm,
                                   const float *reduce_sum_value, float *output_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const float clip_norm_float = __half2float(clip_norm[0]);
    if (reduce_sum_value[0] > clip_norm_float) {
      output_addr[i] = x[i] * clip_norm_float / reduce_sum_value[0];
    } else {
      output_addr[i] = x[i];
    }
  }
}

template <typename T>
void ClipGradNormOp(const size_t size, const float *x, const T *clip_norm, const float *reduce_sum_value,
                    float *output_addr, cudaStream_t cuda_stream) {
  ClipGradNormKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, x, clip_norm, reduce_sum_value,
                                                                        output_addr);
}

template void ClipGradNormOp<float>(const size_t size, const float *x, const float *clip_norm,
                                    const float *reduce_sum_value, float *output_addr, cudaStream_t cuda_stream);

template void ClipGradNormOp<half>(const size_t size, const float *x, const half *clip_norm,
                                   const float *reduce_sum_value, float *output_addr, cudaStream_t cuda_stream);
