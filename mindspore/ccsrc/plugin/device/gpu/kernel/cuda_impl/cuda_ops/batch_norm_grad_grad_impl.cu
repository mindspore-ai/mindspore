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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "batch_norm_grad_grad_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
struct DynamicSharedMem;

template <>
struct DynamicSharedMem<float> {
  __device__ float *addr() {
    extern __shared__ float addr_float[];
    return addr_float;
  }
};

__global__ void ComputeInvStdKernel(const float *variance, ShapeInfo shape_info, float epsilon, float *inv_std) {
  float a = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < shape_info.c; i += blockDim.x * gridDim.x) {
    inv_std[i] = a / std::sqrt(variance[i] + epsilon);
  }
}

template <typename T>
__global__ void ComputeXHatKernel(const T *x, const float *mean, ShapeInfo shape_info, DataFormat format,
                                  float *inv_std, float *x_hat) {
  size_t size = shape_info.n * shape_info.c * shape_info.h * shape_info.w;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    size_t c = format == DataFormat::NCHW ? i / (shape_info.h * shape_info.w) % shape_info.c : i % shape_info.c;
    x_hat[i] = inv_std[c] * (static_cast<float>(x[i]) - mean[c]);
  }
}

template <typename T>
__device__ void BatchReduce(const T *dy, const T *dout_dx, const float *x_hat, float *mean_dy_tmp,
                            float *mean_dout_dx_tmp, float *mean_dy_mul_x_hat_tmp, float *mean_dout_dx_mul_x_hat_tmp,
                            float *mean_dy_mul_dout_dx_tmp, float *mean_dy, float *mean_dout_dx,
                            float *mean_dy_mul_x_hat, float *mean_dout_dx_mul_x_hat, float *mean_dy_mul_dout_dx,
                            size_t c, const ShapeInfo &shape_info, DataFormat format) {
  bool is_nchw = format == DataFormat::NCHW;
  for (size_t n = threadIdx.x; n < shape_info.n; n += blockDim.x) {
    size_t offset = is_nchw ? n * shape_info.c * shape_info.h * shape_info.w + c * shape_info.h * shape_info.w
                            : n * shape_info.c * shape_info.h * shape_info.w + c;
    const auto *dy_offset = dy + offset;
    const auto *dout_dx_offset = dout_dx + offset;
    const auto *x_hat_offset = x_hat + offset;
    mean_dy_tmp[n] = 0.0;
    mean_dout_dx_tmp[n] = 0.0;
    mean_dy_mul_x_hat_tmp[n] = 0.0;
    mean_dout_dx_mul_x_hat_tmp[n] = 0.0;
    mean_dy_mul_dout_dx_tmp[n] = 0.0;

    for (size_t k = 0; k < shape_info.h * shape_info.w; k++) {
      size_t index = is_nchw ? k : k * shape_info.c;
      mean_dy_tmp[n] += static_cast<float>(dy_offset[index]);
      mean_dout_dx_tmp[n] += static_cast<float>(dout_dx_offset[index]);
      mean_dy_mul_x_hat_tmp[n] += static_cast<float>(dy_offset[index]) * x_hat_offset[index];
      mean_dout_dx_mul_x_hat_tmp[n] += static_cast<float>(dout_dx_offset[index]) * x_hat_offset[index];
      mean_dy_mul_dout_dx_tmp[n] += static_cast<float>(dy_offset[index]) * static_cast<float>(dout_dx_offset[index]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mean_dy[c] = 0.0;
    mean_dout_dx[c] = 0.0;
    mean_dy_mul_x_hat[c] = 0.0;
    mean_dout_dx_mul_x_hat[c] = 0.0;
    mean_dy_mul_dout_dx[c] = 0.0;
    for (size_t n = 0; n < shape_info.n; n++) {
      mean_dy[c] += mean_dy_tmp[n];
      mean_dout_dx[c] += mean_dout_dx_tmp[n];
      mean_dy_mul_x_hat[c] += mean_dy_mul_x_hat_tmp[n];
      mean_dout_dx_mul_x_hat[c] += mean_dout_dx_mul_x_hat_tmp[n];
      mean_dy_mul_dout_dx[c] += mean_dy_mul_dout_dx_tmp[n];
    }
    auto size = static_cast<float>(shape_info.n * shape_info.h * shape_info.w);
    mean_dy[c] /= size;
    mean_dout_dx[c] /= size;
    mean_dy_mul_x_hat[c] /= size;
    mean_dout_dx_mul_x_hat[c] /= size;
    mean_dy_mul_dout_dx[c] /= size;
  }
}

template <typename T>
__global__ void ReduceMeanKernel(const T *dy, const T *dout_dx, const float *x_hat, ShapeInfo shape_info,
                                 DataFormat format, float *mean_dy, float *mean_dout_dx, float *mean_dy_mul_x_hat,
                                 float *mean_dout_dx_mul_x_hat, float *mean_dy_mul_dout_dx) {
  DynamicSharedMem<float> share_mem;
  float *mean_dy_tmp = share_mem.addr();
  float *mean_dout_dx_tmp = share_mem.addr() + shape_info.n;
  float *mean_dy_mul_x_hat_tmp = share_mem.addr() + shape_info.n * 2;
  float *mean_dout_dx_mul_x_hat_tmp = share_mem.addr() + shape_info.n * 3;
  float *mean_dy_mul_dout_dx_tmp = share_mem.addr() + shape_info.n * 4;
  for (size_t c = blockIdx.x; c < shape_info.c; c += gridDim.x) {
    __syncthreads();
    BatchReduce(dy, dout_dx, x_hat, mean_dy_tmp, mean_dout_dx_tmp, mean_dy_mul_x_hat_tmp, mean_dout_dx_mul_x_hat_tmp,
                mean_dy_mul_dout_dx_tmp, mean_dy, mean_dout_dx, mean_dy_mul_x_hat, mean_dout_dx_mul_x_hat,
                mean_dy_mul_dout_dx, c, shape_info, format);
  }
}

template <typename T>
__global__ void ComputeTrainingGradsKernel(const T *dy, const float *scale, const T *dout_dx, const float *dout_dscale,
                                           const float *dout_dbias, T *ddy, T *dx, float *inv_std, float *x_hat,
                                           float *mean_dy, float *mean_dout_dx, float *mean_dy_mul_x_hat,
                                           float *mean_dout_dx_mul_x_hat, float *mean_dy_mul_dout_dx,
                                           ShapeInfo shape_info, DataFormat format) {
  float a = 3.0;
  float dx_term_0 = 0.0;
  float dx_term_1 = 0.0;
  float dx_term_2 = 0.0;
  float dx_term = 0.0;
  float scale_term = 0.0;
  size_t size = shape_info.n * shape_info.c * shape_info.h * shape_info.w;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    size_t c = format == DataFormat::NCHW ? i / (shape_info.h * shape_info.w) % shape_info.c : i % shape_info.c;
    ddy[i] = static_cast<T>(
      scale[c] * inv_std[c] *
        (static_cast<float>(dout_dx[i]) - mean_dout_dx[c] - static_cast<float>(x_hat[i]) * mean_dout_dx_mul_x_hat[c]) +
      dout_dscale[c] * static_cast<float>(x_hat[i]) + dout_dbias[c]);
    dx_term_0 = static_cast<float>(x_hat[i]) * (mean_dout_dx[c] * mean_dy[c] - mean_dy_mul_dout_dx[c] +
                                                a * mean_dy_mul_x_hat[c] * mean_dout_dx_mul_x_hat[c]);
    dx_term_1 = mean_dout_dx_mul_x_hat[c] * (mean_dy[c] - static_cast<float>(dy[i]));
    dx_term_2 = mean_dy_mul_x_hat[c] * (mean_dout_dx[c] - static_cast<float>(dout_dx[i]));
    dx_term = scale[c] * inv_std[c] * inv_std[c] * (dx_term_0 + dx_term_1 + dx_term_2);
    scale_term = dout_dscale[c] * inv_std[c] *
                 (static_cast<float>(dy[i]) - mean_dy[c] - mean_dy_mul_x_hat[c] * static_cast<float>(x_hat[i]));
    dx[i] = static_cast<T>(dx_term + scale_term);
    x_hat[i] = static_cast<float>(dout_dx[i]) * inv_std[c] *
               (static_cast<float>(dy[i]) - mean_dy[c] - mean_dy_mul_x_hat[c] * static_cast<float>(x_hat[i]));
  }
}

__global__ void ComputeBpropScaleKernel(const float *tmp, float *dscale, ShapeInfo shape_info, DataFormat format) {
  DynamicSharedMem<float> share_mem;
  float *dscale_tmp = share_mem.addr();
  bool is_nchw = format == DataFormat::NCHW;
  for (size_t c = blockIdx.x; c < shape_info.c; c += gridDim.x) {
    __syncthreads();
    for (size_t n = threadIdx.x; n < shape_info.n; n += blockDim.x) {
      size_t offset = is_nchw ? n * shape_info.c * shape_info.h * shape_info.w + c * shape_info.h * shape_info.w
                              : n * shape_info.c * shape_info.h * shape_info.w + c;
      const auto *tmp_offset = tmp + offset;
      dscale_tmp[n] = 0.0;
      for (size_t k = 0; k < shape_info.h * shape_info.w; k++) {
        size_t index = is_nchw ? k : k * shape_info.c;
        dscale_tmp[n] += tmp_offset[index];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      dscale[c] = 0.0;
      for (size_t n = 0; n < shape_info.n; n++) {
        dscale[c] += dscale_tmp[n];
      }
    }
  }
}

template <typename T>
__global__ void ComputeInferenceGradsKernel(const T *dy, const T *x, const float *scale, const float *mean,
                                            const T *dout_dx, const float *dout_dscale, const float *dout_dbias,
                                            const float *inv_std, T *ddy, T *dx, float *tmp, ShapeInfo shape_info,
                                            DataFormat format) {
  size_t size = shape_info.n * shape_info.c * shape_info.h * shape_info.w;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    size_t c = format == DataFormat::NCHW ? i / (shape_info.h * shape_info.w) % shape_info.c : i % shape_info.c;
    dx[i] = static_cast<T>(dout_dscale[c] * inv_std[c] * static_cast<float>(dy[i]));
    ddy[i] = static_cast<T>(static_cast<float>(dout_dx[i]) * inv_std[c] * scale[c] +
                            dout_dscale[c] * inv_std[c] * (static_cast<float>(x[i]) - mean[c]) + dout_dbias[c]);
    tmp[i] = static_cast<float>(dout_dx[i]) * static_cast<float>(dy[i]) * inv_std[c];
  }
}

template <typename T>
cudaError_t BatchNormGradGradTraining(const T *dy, const T *x, const float *scale, const float *mean,
                                      const float *variance, const T *dout_dx, const float *dout_dscale,
                                      const float *dout_dbias, T *ddy, T *dx, float *dscale, float *inv_std,
                                      float *x_hat, float *mean_dy, float *mean_dout_dx, float *mean_dy_mul_x_hat,
                                      float *mean_dout_dx_mul_x_hat, float *mean_dy_mul_dout_dx,
                                      const ShapeInfo &shape_info, DataFormat format, float epsilon, uint32_t device_id,
                                      cudaStream_t stream) {
  size_t size = shape_info.n * shape_info.c * shape_info.h * shape_info.w;
  ComputeInvStdKernel<<<CUDA_BLOCKS(device_id, shape_info.c), CUDA_THREADS(device_id), 0, stream>>>(
    variance, shape_info, epsilon, inv_std);
  ComputeXHatKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(x, mean, shape_info, format,
                                                                                          inv_std, x_hat);
  ReduceMeanKernel<<<CUDA_BLOCKS_MAXSIZE(device_id, shape_info.c), CUDA_THREADS_MAXSIZE(device_id, shape_info.n),
                     shape_info.n * sizeof(float), stream>>>(dy, dout_dx, x_hat, shape_info, format, mean_dy,
                                                             mean_dout_dx, mean_dy_mul_x_hat, mean_dout_dx_mul_x_hat,
                                                             mean_dy_mul_dout_dx);

  ComputeTrainingGradsKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
    dy, scale, dout_dx, dout_dscale, dout_dbias, ddy, dx, inv_std, x_hat, mean_dy, mean_dout_dx, mean_dy_mul_x_hat,
    mean_dout_dx_mul_x_hat, mean_dy_mul_dout_dx, shape_info, format);
  ComputeBpropScaleKernel<<<CUDA_BLOCKS_MAXSIZE(device_id, shape_info.c), CUDA_THREADS_MAXSIZE(device_id, shape_info.n),
                            shape_info.n * sizeof(float), stream>>>(x_hat, dscale, shape_info, format);
  return GetCudaStatus();
}

template <typename T>
cudaError_t BatchNormGradGradInference(const T *dy, const T *x, const float *scale, const float *mean,
                                       const float *variance, const T *dout_dx, const float *dout_dscale,
                                       const float *dout_dbias, T *ddy, T *dx, float *dscale, float *inv_std,
                                       float *tmp, const ShapeInfo &shape_info, DataFormat format, float epsilon,
                                       uint32_t device_id, cudaStream_t stream) {
  size_t size = shape_info.n * shape_info.c * shape_info.h * shape_info.w;
  ComputeInvStdKernel<<<CUDA_BLOCKS(device_id, shape_info.c), CUDA_THREADS(device_id), 0, stream>>>(
    variance, shape_info, epsilon, inv_std);

  ComputeInferenceGradsKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
    dy, x, scale, mean, dout_dx, dout_dscale, dout_dbias, inv_std, ddy, dx, tmp, shape_info, format);
  ComputeBpropScaleKernel<<<CUDA_BLOCKS_MAXSIZE(device_id, shape_info.c), CUDA_THREADS_MAXSIZE(device_id, shape_info.n),
                            shape_info.n * sizeof(float), stream>>>(tmp, dscale, shape_info, format);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t BatchNormGradGradTraining<float>(
  const float *dy, const float *x, const float *scale, const float *mean, const float *variance, const float *dout_dx,
  const float *dout_dscale, const float *dout_dbias, float *dx, float *ddy, float *dscale, float *inv_std, float *x_hat,
  float *mean_dy, float *mean_dout_dx, float *mean_dy_mul_x_hat, float *mean_dout_dx_mul_x_hat,
  float *mean_dy_mul_dout_dx, const ShapeInfo &shape_info, DataFormat format, float epsilon, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BatchNormGradGradTraining<half>(
  const half *dy, const half *x, const float *scale, const float *mean, const float *variance, const half *dout_dx,
  const float *dout_dscale, const float *dout_dbias, half *dx, half *ddy, float *dscale, float *inv_std, float *x_hat,
  float *mean_dy, float *mean_dout_dx, float *mean_dy_mul_x_hat, float *mean_dout_dx_mul_x_hat,
  float *mean_dy_mul_dout_dx, const ShapeInfo &shape_info, DataFormat format, float epsilon, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BatchNormGradGradInference<float>(
  const float *dy, const float *x, const float *scale, const float *mean, const float *variance, const float *dout_dx,
  const float *dout_dscale, const float *dout_dbias, float *dx, float *ddy, float *dscale, float *inv_std, float *tmp,
  const ShapeInfo &shape_info, DataFormat format, float epsilon, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BatchNormGradGradInference<half>(
  const half *dy, const half *x, const float *scale, const float *mean, const float *variance, const half *dout_dx,
  const float *dout_dscale, const float *dout_dbias, half *dx, half *ddy, float *dscale, float *inv_std, float *tmp,
  const ShapeInfo &shape_info, DataFormat format, float epsilon, uint32_t device_id, cudaStream_t stream);
