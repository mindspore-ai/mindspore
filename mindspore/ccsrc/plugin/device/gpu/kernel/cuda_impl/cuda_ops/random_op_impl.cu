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
__global__ void UniformIntKernel(curandState *globalState, T *input1, size_t input_size_1,
                                 T *input2, size_t input_size_2, T *output, size_t count) {
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
