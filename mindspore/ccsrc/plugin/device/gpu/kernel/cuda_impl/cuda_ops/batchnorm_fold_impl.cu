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

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>
#include "batchnorm_fold_impl.cuh"

template <typename T>
__global__ void UpdateRunningStd(int channel_size, const double epsilon, T *running_std) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channel_size; i += blockDim.x * gridDim.x) {
    running_std[i] = sqrtf(running_std[i] + epsilon);
  }
  return;
}

template <typename T>
__global__ void UpdateBatchStd(int channel_size, T *batch_std) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channel_size; i += blockDim.x * gridDim.x) {
    batch_std[i] = 1 / batch_std[i];
  }
  return;
}

template <typename T>
__global__ void CalDx(const T *d_batch_mean, const T *d_batch_std, const T *x, const T *batch_mean, const T *batch_std,
                      int batch_size, int channel_size, int height, int width, T *dx) {
  int n = batch_size * channel_size * height * width;
  int normal_size = batch_size * height * width;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int channel_index = i / (height * width) % channel_size;
    dx[i] = d_batch_mean[channel_index] / normal_size +
            d_batch_std[channel_index] * (x[i] - batch_mean[channel_index]) / batch_std[channel_index] / normal_size;
  }
  return;
}

template <typename T>
cudaError_t CalUpdateRunningStd(int channel_size, double epsilon, T *running_std, cudaStream_t cuda_stream) {
  UpdateRunningStd<<<GET_BLOCKS(channel_size), GET_THREADS, 0, cuda_stream>>>(channel_size, epsilon, running_std);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpdateRunningStd<float>(int channel_size, double epsilon, float *running_std,
                                                                cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalUpdateBatchStd(int channel_size, T *batch_std, cudaStream_t cuda_stream) {
  UpdateBatchStd<<<GET_BLOCKS(channel_size), GET_THREADS, 0, cuda_stream>>>(channel_size, batch_std);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpdateBatchStd<float>(int channel_size, float *batch_std,
                                                              cudaStream_t cuda_stream);

template <typename T>
cudaError_t CalBatchNormFoldGrad(const T *d_batch_mean, const T *d_batch_std, const T *x, const T *batch_mean,
                                 const T *batch_std, int batch_size, int channel_size, int height, int width, T *dx,
                                 cudaStream_t cuda_stream) {
  CalDx<<<GET_BLOCKS(batch_size * channel_size * height * width), GET_THREADS, 0, cuda_stream>>>(
    d_batch_mean, d_batch_std, x, batch_mean, batch_std, batch_size, channel_size, height, width, dx);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchNormFoldGrad<float>(const float *d_batch_mean, const float *d_batch_std,
                                                                 const float *x, const float *batch_mean,
                                                                 const float *batch_std, int batch_size,
                                                                 int channel_size, int height, int width, float *dx,
                                                                 cudaStream_t cuda_stream);

template <typename T>
cudaError_t ThrustFillWith(T *array, int size, T tofill, cudaStream_t cuda_stream) {
  thrust::device_ptr<T> dev_ptr(array);
  thrust::fill(thrust::cuda::par.on(cuda_stream), dev_ptr, dev_ptr + size, tofill);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ThrustFillWith<float>(float *array, int size, float tofill,
                                                           cudaStream_t cuda_stream);
