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
template <typename T>
__global__ void NormalKernel(int seed, curandState *globalState, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    output[i] = (T)curand_normal(&globalState[i]);
  }
  return;
}

__device__ bool dev_error_res = false;

template <typename T>
__global__ void UniformIntKernel(int seed, curandState *globalState, T *input1, size_t input_size_1,
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
__global__ void UniformRealKernel(int seed, curandState *globalState, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    output[i] = (T)curand_uniform(&globalState[i]);
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
  NormalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count);
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
  UniformIntKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>
               (RNG_seed, globalState, input1, input_size_1, input2, input_size_2, output, count);
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
  UniformRealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, output, count);
  return;
}

template void StandardNormal<float>(int seed, int seed2, curandState *globalState,
                                    float *output, size_t count, cudaStream_t cuda_stream);
template void StandardNormal<int>(int seed, int seed2, curandState *globalState,
                                  int *output, size_t count, cudaStream_t cuda_stream);
template bool UniformInt<float>(int seed, int seed2, curandState *globalState, float *input1, size_t input_size_1,
                                float *input2, size_t input_size_2, float *output, size_t count,
                              cudaStream_t cuda_stream);
template bool UniformInt<int>(int seed, int seed2, curandState *globalState, int *input1, size_t input_size_1,
                              int *input2, size_t input_size_2, int *output, size_t count,
                              cudaStream_t cuda_stream);
template void UniformReal<float>(int seed, int seed2, curandState *globalState,
                                 float *output, size_t count, cudaStream_t cuda_stream);
template void UniformReal<int>(int seed, int seed2, curandState *globalState,
                               int *output, size_t count, cudaStream_t cuda_stream);
