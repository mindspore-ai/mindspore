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
#include "include/cuda_fp16.h"
__global__ void SetupKernel(int seed, curandState *globalState) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, id, 0, &globalState[id]);
}

template <typename T>
__global__ void NormalKernel(curandState *globalState, T *output, size_t count) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    globalState[id] = localState;
    output[id] = (T)curand_normal(&localState);
    id += blockDim.x * gridDim.x;
  }
  return;
}

__device__ bool dev_error_res = false;

template <typename T>
__global__ void UniformIntKernel(curandState *globalState, T *input1, size_t input_size_1, T *input2,
                                 size_t input_size_2, T *output, size_t count) {
  if (!(input1[0] < input2[0])) {
    dev_error_res = false;
    return;
  }

  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    globalState[id] = localState;
    output[id] = (T)(curand_uniform(&localState) * (input2[0] - input1[0])) + input1[0];
    id += blockDim.x * gridDim.x;
  }
  dev_error_res = true;
  return;
}

template <typename T>
__global__ void UniformRealKernel(curandState *globalState, T *output, size_t count) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    globalState[id] = localState;
    output[id] = (T)curand_uniform(&localState);
    id += blockDim.x * gridDim.x;
  }
  return;
}

template<typename S>
__global__ void TruncatedNormalKernel(curandState *globalState, S *output, size_t count) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    S random_data;
    globalState[id] = localState;
    random_data = (S)curand_normal(&localState);
    auto curState = localState;
    do {
      random_data = (S)curand_normal(&curState);
    }while(random_data < -(S)0.2 || random_data > (S)0.2);
    output[id] = random_data;
    id += blockDim.x * gridDim.x;
  }
  return;
}

template <typename R, typename T>
__global__ void RandomPoissonKernel(curandState *globalState, R *rate, int rate_size, T *output, size_t count) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    auto j = id % rate_size;
    globalState[id] = localState;
    output[id] = (T)curand_poisson(&localState, rate[j]);
    id += blockDim.x * gridDim.x;
  }
  return;
}

template <typename T>
__global__ void StandardLaplaceKernel(curandState *globalState, T *output, size_t count, T min_num) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto localState = globalState[id];

  while (id < count) {
    globalState[id] = localState;
    T temp = (T)(curand_uniform(&localState) * 2 - 1);
    T temp2 = temp < 0 ? temp + min_num : temp - min_num;
    T sign = std::copysignf(1.0, temp2);
    output[id] = -sign * std::log(1.0 - std::abs(temp2));
    id += blockDim.x * gridDim.x;
  }
  return;
}

template <typename T>
void StandardNormal(int seed, int seed2, curandState *globalState, T *output, size_t count, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  NormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(globalState, output, count);
  return;
}

template <typename T>
bool UniformInt(int seed, int seed2, curandState *globalState, T *input1, size_t input_size_1,
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
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  UniformIntKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>
               (globalState, input1, input_size_1, input2, input_size_2, output, count);
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&host_error_res, dev_error_res, sizeof(bool));
  return host_error_res;
}

template <typename T>
void UniformReal(int seed, int seed2, curandState *globalState, T *output, size_t count, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  UniformRealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(globalState, output, count);
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
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  TruncatedNormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(globalState, output, count);
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
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  RandomPoissonKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(globalState, rate, rate_size, output, count);
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
  SetupKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState);
  StandardLaplaceKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(globalState, output, count, min_num);
  return;
}

template CUDA_LIB_EXPORT void StandardNormal<float>(int seed, int seed2, curandState *globalState,
                                                    float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void StandardNormal<int>(int seed, int seed2, curandState *globalState,
                                                  int *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool UniformInt<float>(int seed, int seed2, curandState *globalState, float *input1,
                                                size_t input_size_1, float *input2, size_t input_size_2, float *output,
                                                size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool UniformInt<int>(int seed, int seed2, curandState *globalState, int *input1,
                                              size_t input_size_1, int *input2, size_t input_size_2, int *output,
                                              size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void UniformReal<float>(int seed, int seed2, curandState *globalState,
                                                 float *output, size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void UniformReal<int>(int seed, int seed2, curandState *globalState,
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
