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

#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include "batchnorm_fold2_impl.cuh"
#include "batchnorm_fold_impl.cuh"
#include "include/cuda_runtime.h"

template <typename T>
__global__ void BatchNormFold2Kernel(const T *x, const T *beta, const T *gamma, const T *batch_std, const T *batch_mean,
                                     const T *running_std, const T *running_mean, const int *global_step, T *y,
                                     int freeze_bn, size_t N, size_t C, size_t H, size_t W) {
  int c = 0;
  size_t num_count = N * C * H * W;
  if (*global_step < freeze_bn) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
      c = i / (H * W) % C;
      y[i] = x[i] * running_std[c] / batch_std[c] + beta[c] - gamma[c] * batch_mean[c] / batch_std[c];
    }
  } else {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
      c = i / (H * W) % C;
      y[i] = x[i] + beta[c] - gamma[c] * running_mean[c] / running_std[c];
    }
  }
}

template <typename T>
__global__ void BatchNormFold2GradReduce1(const T *dout, T *tmp, const T *x, T *tmp2, size_t N, size_t C, size_t HW) {
  int n = 0;
  int c = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C; i += blockDim.x * gridDim.x) {
    n = i / C;
    c = i % C;
    tmp[c * N + n] = thrust::reduce(thrust::seq, dout + i * HW, dout + (i + 1) * HW, 0.f, thrust::plus<T>());
    tmp2[c * N + n] = thrust::reduce(thrust::seq, x + i * HW, x + (i + 1) * HW, 0.f, thrust::plus<T>());
  }
}

template <typename T>
__global__ void BatchNormFold2GradReduce2(const T *tmp, T *d_beta, const T *tmp2, T *reduce_x, size_t N, size_t C) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += blockDim.x * gridDim.x) {
    d_beta[i] = thrust::reduce(thrust::seq, tmp + i * N, tmp + (i + 1) * N, 0.f, thrust::plus<T>());
    reduce_x[i] = thrust::reduce(thrust::seq, tmp2 + i * N, tmp2 + (i + 1) * N, 0.f, thrust::plus<T>());
  }
}

template <typename T>
__global__ void BatchNormFold2GradNotFreeze(const T *d_beta, const T *reduce_x, const T *batch_mean, const T *batch_std,
                                            const T *running_mean, const T *running_std, const T *gamma, T *d_gamma,
                                            T *d_batch_mean, T *d_batch_std, size_t C) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += blockDim.x * gridDim.x) {
    d_gamma[i] = -d_beta[i] * batch_mean[i] / batch_std[i];
    d_batch_mean[i] = -d_beta[i] * gamma[i] / batch_std[i];
    d_batch_std[i] =
      (d_beta[i] * gamma[i] * batch_mean[i] - reduce_x[i] * running_std[i]) / batch_std[i] / batch_std[i];
  }
}

template <typename T>
__global__ void BatchNormFold2GradFreeze(const T *d_beta, const T *running_mean, const T *running_std, T *d_gamma,
                                         size_t C) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += blockDim.x * gridDim.x) {
    d_gamma[i] = -d_beta[i] * running_mean[i] / running_std[i];
  }
}

template <typename T>
__global__ void BatchNormFold2GradMul(const T *dout, const T *x, T *tmp_x, size_t NCHW) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < NCHW; i += blockDim.x * gridDim.x) {
    tmp_x[i] = dout[i] * x[i];
  }
}

template <typename T>
__global__ void DxMul(size_t N, size_t C, size_t HW, const T *batch_std, const T *running_std, T *d_x) {
  int c = 0;
  size_t num_count = N * C * HW;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    c = (i / HW) % C;
    d_x[i] = d_x[i] * running_std[c] / batch_std[c];
  }
}

template <typename T>
cudaError_t BatchNormFold2Forward(const T *x, const T *beta, const T *gamma, const T *batch_std, const T *batch_mean,
                                  const T *running_std, const T *running_mean, const int *global_step, T *y,
                                  int freeze_bn, size_t N, size_t C, size_t H, size_t W, cudaStream_t cuda_stream) {
  auto num_count = N * C * H * W;
  BatchNormFold2Kernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(
    x, beta, gamma, batch_std, batch_mean, running_std, running_mean, global_step, y, freeze_bn, N, C, H, W);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t BatchNormFold2Forward<float>(const float *x, const float *beta, const float *gamma,
                                                                  const float *batch_std, const float *batch_mean,
                                                                  const float *running_std, const float *running_mean,
                                                                  const int *global_step, float *y, int freeze_bn,
                                                                  size_t N, size_t C, size_t H, size_t W,
                                                                  cudaStream_t cuda_stream);

template <typename T>
cudaError_t BatchNormFold2GradReduce(const T *dout, const T *x, T *d_beta, T *tmp, T *reduce_x, T *tmp2, T *tmp_x,
                                     size_t N, size_t C, size_t H, size_t W, cudaStream_t cuda_stream) {
  auto hw = H * W;
  auto num_count = N * C * H * W;
  BatchNormFold2GradMul<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(dout, x, tmp_x, num_count);
  BatchNormFold2GradReduce1<<<GET_BLOCKS(N * C), GET_THREADS, 0, cuda_stream>>>(dout, tmp, tmp_x, tmp2, N, C, hw);
  BatchNormFold2GradReduce2<<<GET_BLOCKS(C), GET_THREADS, 0, cuda_stream>>>(tmp, d_beta, tmp2, reduce_x, N, C);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t BatchNormFold2GradReduce<float>(const float *dout, const float *x, float *d_beta,
                                                                     float *tmp, float *reduce_x, float *tmp2,
                                                                     float *tmp_x, size_t N, size_t C, size_t H,
                                                                     size_t W, cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalBatchNormFold2GradNotFreeze(const T *d_beta, const T *reduce_x, const T *batch_mean, const T *batch_std,
                                           const T *running_mean, const T *running_std, const T *gamma, T *d_gamma,
                                           T *d_batch_mean, T *d_batch_std, size_t C, cudaStream_t cuda_stream) {
  BatchNormFold2GradNotFreeze<<<GET_BLOCKS(C), GET_THREADS, 0, cuda_stream>>>(
    d_beta, reduce_x, batch_mean, batch_std, running_mean, running_std, gamma, d_gamma, d_batch_mean, d_batch_std, C);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchNormFold2GradNotFreeze<float>(
  const float *d_beta, const float *reduce_x, const float *batch_mean, const float *batch_std,
  const float *running_mean, const float *running_std, const float *gamma, float *d_gamma, float *d_batch_mean,
  float *d_batch_std, size_t C, cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalBatchNormFold2GradFreeze(const T *d_beta, const T *reduce_x, const T *batch_mean, const T *batch_std,
                                        const T *running_mean, const T *running_std, const T *gamma, T *d_gamma,
                                        T *d_batch_mean, T *d_batch_std, size_t C, cudaStream_t cuda_stream) {
  BatchNormFold2GradFreeze<<<GET_BLOCKS(C), GET_THREADS, 0, cuda_stream>>>(d_beta, running_mean, running_std, d_gamma,
                                                                           C);
  ThrustFillWith(d_batch_mean, C, (T)0.f, cuda_stream);
  ThrustFillWith(d_batch_std, C, (T)0.f, cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchNormFold2GradFreeze<float>(
  const float *d_beta, const float *reduce_x, const float *batch_mean, const float *batch_std,
  const float *running_mean, const float *running_std, const float *gamma, float *d_gamma, float *d_batch_mean,
  float *d_batch_std, size_t C, cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalBatchNormFold2GradNotFreezeDxMul(const T *batch_std, const T *running_std, T *d_x, size_t N, size_t C,
                                                size_t H, size_t W, cudaStream_t cuda_stream) {
  DxMul<<<GET_BLOCKS(N * C * H * W), GET_THREADS, 0, cuda_stream>>>(N, C, H * W, batch_std, running_std, d_x);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchNormFold2GradNotFreezeDxMul<float>(const float *batch_std,
                                                                                const float *running_std, float *d_x,
                                                                                size_t N, size_t C, size_t H, size_t W,
                                                                                cudaStream_t cuda_stream);
