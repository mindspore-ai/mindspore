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

#include "random_op_impl.cuh"
#include <math.h>
#include "include/cuda_fp16.h"
template <typename T>
__global__ void NormalKernel(int seed, curandStatePhilox4_32_10_t *globalState, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    output[i] = (T)curand_normal(&globalState[i]);
  }
  return;
}

__device__ bool dev_error_res = false;

template <typename T>
__global__ void UniformIntKernel(int seed, curandStatePhilox4_32_10_t *globalState, T *input1, size_t input_size_1,
                                 T *input2, size_t input_size_2, T *output, size_t count) {
  if (!(input1[0] < input2[0])) {
    dev_error_res = false;
    return;
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    output[i] = (T)(curand_uniform(&globalState[i]) * (input2[0] - input1[0])) + input1[0];
  }
  dev_error_res = true;
  return;
}

template <typename T>
__global__ void UniformRealKernel(int seed, curandStatePhilox4_32_10_t *globalState, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    output[i] = (T)curand_uniform(&globalState[i]);
  }
  return;
}

template<typename S>
__global__ void TruncatedNormalKernel(int seed, curandState *globalState, S *output, size_t count) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    S random_data;
    curand_init(seed, i, 0, &globalState[i]);
    random_data = (S)curand_normal(&globalState[i]);
    do {
      random_data = (S)curand_normal(&globalState[i]);
    }while(random_data < -(S)0.2 || random_data > (S)0.2);
    output[i] = random_data;
  }
  return;
}

template <typename R, typename T>
__global__ void RandomPoissonKernel(int seed, curandState *globalState, R *rate, int rate_size, T *output,
                                    size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    auto j = i % rate_size;
    output[i] = (T)curand_poisson(&globalState[i], rate[j]);
  }
  return;
}

template <typename T>
__global__ void StandardLaplaceKernel(int seed, curandState *globalState, T *output, size_t count, T min_num) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    T temp = (T)(curand_uniform(&globalState[i]) * 2 - 1);
    T temp2 = temp < 0 ? temp + min_num : temp - min_num;
    T sign = copysignf(1.0, temp2);
    output[i] = -sign * log(1.0 - abs(temp2));
  }
  return;
}

template <typename T>
void StandardNormal(int seed, int seed2, curandStatePhilox4_32_10_t *globalState, T *output, size_t count,
                    cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  NormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count);
  return;
}

template <typename T>
bool UniformInt(int seed, int seed2, curandStatePhilox4_32_10_t *globalState, T *input1, size_t input_size_1,
                T *input2, size_t input_size_2, T *output, size_t count, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  bool host_error_res = false;
  UniformIntKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>
               (RNG_seed, globalState, input1, input_size_1, input2, input_size_2, output, count);
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&host_error_res, dev_error_res, sizeof(bool));
  return host_error_res;
}

template <typename T>
void UniformReal(int seed, int seed2, curandStatePhilox4_32_10_t *globalState, T *output, size_t count,
                 cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  UniformRealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count);
  return;
}

template<typename S>
void TruncatedNormal(int seed, int seed2, curandState *globalState, S *output, size_t count, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  TruncatedNormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count);
  return;
}

template <typename R, typename T>
void RandomPoisson(int seed, int seed2, curandState *globalState, R *rate, int64_t rate_size, T *output, size_t count,
                   cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  RandomPoissonKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, rate, rate_size,
                                                                          output, count);
  return;
}

template <typename T>
void StandardLaplace(int seed, int seed2, curandState *globalState, T *output, size_t count, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  T min_num = std::nextafter(0, 1);
  StandardLaplaceKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count,
                                                                            min_num);
  return;
}

template CUDA_LIB_EXPORT void StandardNormal<float>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState,
                                                    float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void StandardNormal<int>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState,
                                                  int *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool UniformInt<float>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState,
                                                float *input1, size_t input_size_1, float *input2, size_t input_size_2,
                                                float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool UniformInt<int>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState, int *input1,
                                              size_t input_size_1, int *input2, size_t input_size_2, int *output,
                                              size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void UniformReal<float>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState,
                                                 float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void UniformReal<int>(int seed, int seed2, curandStatePhilox4_32_10_t *globalState,
                                               int *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TruncatedNormal<half>(int seed, int seed2,  curandState *globalState,
                                                    half *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TruncatedNormal<float>(int seed, int seed2,  curandState *globalState,
                                                     float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TruncatedNormal<double>(int seed, int seed2,  curandState *globalState,
                                                      double *output, size_t count, cudaStream_t cuda_stream);
#define ADD_RANDOM_POISSON(rate_type, output_type) \
  template CUDA_LIB_EXPORT void RandomPoisson<rate_type, output_type>(int seed, int seed2, curandState *globalState, \
                                                                      rate_type *rate, int64_t rate_size, \
                                                                      output_type *output, size_t count, \
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void StandardLaplace<float>(int seed, int seed2, curandState *globalState,
                                                    float *output, size_t count, cudaStream_t cuda_stream);

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
