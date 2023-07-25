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

#include <thrust/reduce.h>
#include "correction_mul_impl.cuh"

template <typename T>
__global__ void CorrectionMul(const T *weight, const T *gamma, const T *running_std, const int batchsize, const int chw,
                              T *output) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batchsize * chw; i += blockDim.x * gridDim.x) {
    int n = i / chw;
    output[i] = weight[i] * gamma[n] / running_std[n];
  }
  return;
}

template <typename T>
__global__ void Mul(int N, const T *a, const T *b, T *c) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    c[i] = a[i] * b[i];
  }
  return;
}

template <typename T>
__global__ void Reduce(int N, int CHW, const T *tmp, const T *running_std, T *d_gamma) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    d_gamma[i] = thrust::reduce(thrust::seq, tmp + i * CHW, tmp + (i + 1) * CHW, 0.f, thrust::plus<T>());
    d_gamma[i] = d_gamma[i] / running_std[i];
  }
  return;
}

template <typename T>
cudaError_t CalCorrectionMul(const T *weight, const T *gamma, const T *running_std, int N, int C, int H, int W,
                             T *output, cudaStream_t cuda_stream) {
  CorrectionMul<<<GET_BLOCKS(N * C * H * W), GET_THREADS, 0, cuda_stream>>>(weight, gamma, running_std, N, C * H * W,
                                                                            output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCorrectionMul<float>(const float *weight, const float *gamma,
                                                             const float *running_std, int N, int C, int H, int W,
                                                             float *output, cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalCorrectionMulGrad(const T *d_out, const T *weight, const T *running_std, int N, int C, int H, int W,
                                 T *d_gamma, T *tmp, cudaStream_t cuda_stream) {
  Mul<<<GET_BLOCKS(N * C * H * W), GET_THREADS, 0, cuda_stream>>>(N * C * H * W, d_out, weight, tmp);
  Reduce<<<GET_BLOCKS(N), GET_THREADS, 0, cuda_stream>>>(N, C * H * W, tmp, running_std, d_gamma);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCorrectionMulGrad<float>(const float *d_out, const float *weight,
                                                                 const float *running_std, int N, int C, int H, int W,
                                                                 float *d_gamma, float *tmp, cudaStream_t cuda_stream);
