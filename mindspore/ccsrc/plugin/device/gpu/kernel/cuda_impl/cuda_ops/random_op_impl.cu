/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "random_op_impl.cuh"
#include <math.h>
#include "include/cuda_fp16.h"
template <typename T>
__global__ void NormalKernel(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState, T *output,
                             size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &globalState[i]);
    output[i] = (T)curand_normal(&globalState[i]);
  }
  return;
}

__device__ bool dev_error_res = false;

template <typename T>
__global__ void UniformIntKernel(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState,
                                 T *input1, size_t input_size_1, T *input2, size_t input_size_2, T *output,
                                 size_t count) {
  if (!(input1[0] < input2[0])) {
    dev_error_res = false;
    return;
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &globalState[i]);
    output[i] = (T)(curand_uniform(&globalState[i]) * (input2[0] - input1[0])) + input1[0];
  }
  dev_error_res = true;
  return;
}

template <typename T>
__global__ void UniformRealKernel(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState,
                                  T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &globalState[i]);
    output[i] = (T)curand_uniform(&globalState[i]);
  }
  return;
}

template <typename S>
__global__ void TruncatedNormalKernel(uint64_t seed, uint64_t seed_offset, curandState *globalState, S *output,
                                      size_t count) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    S random_data;
    curand_init(seed, i, seed_offset, &globalState[i]);
    random_data = (S)curand_normal(&globalState[i]);
    do {
      random_data = (S)curand_normal(&globalState[i]);
    } while (random_data < -(S)0.2 || random_data > (S)0.2);
    output[i] = random_data;
  }
  return;
}

template <typename R, typename T>
__global__ void RandomPoissonKernel(uint64_t seed, uint64_t seed_offset, curandState *globalState, R *rate,
                                    int rate_size, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &globalState[i]);
    auto j = i % rate_size;
    output[i] = (T)curand_poisson(&globalState[i], rate[j]);
  }
  return;
}

template <typename T>
__global__ void StandardLaplaceKernel(uint64_t seed, uint64_t seed_offset, curandState *globalState, T *output,
                                      size_t count, T min_num) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &globalState[i]);
    T temp = (T)(curand_uniform(&globalState[i]) * 2 - 1);
    T temp2 = temp < 0 ? temp + min_num : temp - min_num;
    T sign = copysignf(1.0, temp2);
    output[i] = -sign * log(1.0 - abs(temp2));
  }
  return;
}

template <typename T>
cudaError_t StandardNormal(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState, T *output,
                           size_t count, cudaStream_t cuda_stream) {
  NormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(seed, seed_offset, globalState, output, count);
  return GetCudaStatus();
}

template <typename T>
cudaError_t UniformInt(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState, T *input1,
                       size_t input_size_1, T *input2, size_t input_size_2, T *output, size_t count,
                       cudaStream_t cuda_stream, bool *host_error_res) {
  UniformIntKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(
    seed, seed_offset, globalState, input1, input_size_1, input2, input_size_2, output, count);
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(host_error_res, dev_error_res, sizeof(bool));
  return GetCudaStatus();
}

template <typename T>
cudaError_t UniformReal(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState, T *output,
                        size_t count, cudaStream_t cuda_stream) {
  UniformRealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(seed, seed_offset, globalState, output, count);
  return GetCudaStatus();
}

template <typename S>
cudaError_t TruncatedNormal(uint64_t seed, uint64_t seed_offset, curandState *globalState, S *output, size_t count,
                            cudaStream_t cuda_stream) {
  TruncatedNormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(seed, seed_offset, globalState, output,
                                                                            count);
  return GetCudaStatus();
}

template <typename R, typename T>
cudaError_t RandomPoisson(uint64_t seed, uint64_t seed_offset, curandState *globalState, R *rate, int64_t rate_size,
                          T *output, size_t count, cudaStream_t cuda_stream) {
  RandomPoissonKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(seed, seed_offset, globalState, rate,
                                                                          rate_size, output, count);
  return GetCudaStatus();
}

template <typename T>
cudaError_t StandardLaplace(uint64_t seed, uint64_t seed_offset, curandState *globalState, T *output, size_t count,
                            cudaStream_t cuda_stream) {
  T min_num = std::nextafter(0, 1);
  StandardLaplaceKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(seed, seed_offset, globalState, output,
                                                                            count, min_num);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t StandardNormal<float>(uint64_t seed, uint64_t seed_offset,
                                                           curandStatePhilox4_32_10_t *globalState, float *output,
                                                           size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StandardNormal<int>(uint64_t seed, uint64_t seed_offset,
                                                         curandStatePhilox4_32_10_t *globalState, int *output,
                                                         size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t UniformInt<float>(uint64_t seed, uint64_t seed_offset,
                                                       curandStatePhilox4_32_10_t *globalState, float *input1,
                                                       size_t input_size_1, float *input2, size_t input_size_2,
                                                       float *output, size_t count, cudaStream_t cuda_stream,
                                                       bool *host_error_res);
template CUDA_LIB_EXPORT cudaError_t UniformInt<int>(uint64_t seed, uint64_t seed_offset,
                                                     curandStatePhilox4_32_10_t *globalState, int *input1,
                                                     size_t input_size_1, int *input2, size_t input_size_2, int *output,
                                                     size_t count, cudaStream_t cuda_stream, bool *host_error_res);
template CUDA_LIB_EXPORT cudaError_t UniformReal<float>(uint64_t seed, uint64_t seed_offset,
                                                        curandStatePhilox4_32_10_t *globalState, float *output,
                                                        size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t UniformReal<int>(uint64_t seed, uint64_t seed_offset,
                                                      curandStatePhilox4_32_10_t *globalState, int *output,
                                                      size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TruncatedNormal<half>(uint64_t seed, uint64_t seed_offset,
                                                           curandState *globalState, half *output, size_t count,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TruncatedNormal<float>(uint64_t seed, uint64_t seed_offset,
                                                            curandState *globalState, float *output, size_t count,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TruncatedNormal<double>(uint64_t seed, uint64_t seed_offset,
                                                             curandState *globalState, double *output, size_t count,
                                                             cudaStream_t cuda_stream);
#define ADD_RANDOM_POISSON(rate_type, output_type)                                                       \
  template CUDA_LIB_EXPORT cudaError_t RandomPoisson<rate_type, output_type>(                            \
    uint64_t seed, uint64_t seed_offset, curandState * globalState, rate_type * rate, int64_t rate_size, \
    output_type * output, size_t count, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t StandardLaplace<float>(uint64_t seed, uint64_t seed_offset,
                                                            curandState *globalState, float *output, size_t count,
                                                            cudaStream_t cuda_stream);

ADD_RANDOM_POISSON(half, half)
ADD_RANDOM_POISSON(half, float)
ADD_RANDOM_POISSON(half, double)
ADD_RANDOM_POISSON(half, int)
ADD_RANDOM_POISSON(half, int64_t)

ADD_RANDOM_POISSON(float, half)
ADD_RANDOM_POISSON(float, float)
ADD_RANDOM_POISSON(float, double)
ADD_RANDOM_POISSON(float, int)
ADD_RANDOM_POISSON(float, int64_t)

ADD_RANDOM_POISSON(double, half)
ADD_RANDOM_POISSON(double, float)
ADD_RANDOM_POISSON(double, double)
ADD_RANDOM_POISSON(double, int)
ADD_RANDOM_POISSON(double, int64_t)

ADD_RANDOM_POISSON(int, half)
ADD_RANDOM_POISSON(int, float)
ADD_RANDOM_POISSON(int, double)
ADD_RANDOM_POISSON(int, int)
ADD_RANDOM_POISSON(int, int64_t)

ADD_RANDOM_POISSON(int64_t, half)
ADD_RANDOM_POISSON(int64_t, float)
ADD_RANDOM_POISSON(int64_t, double)
ADD_RANDOM_POISSON(int64_t, int)
ADD_RANDOM_POISSON(int64_t, int64_t)
