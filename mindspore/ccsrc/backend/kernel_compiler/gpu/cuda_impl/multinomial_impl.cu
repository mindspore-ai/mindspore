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

#include <random>
#include "multinomial_impl.cuh"

template <typename T>
__global__ void CheckZeroKernel(const size_t distributions, const size_t categories, const T *input, T *out) {
  out[0] = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (distributions); pos += blockDim.x * gridDim.x) {
    if (input[(1 + pos) * categories - 1] <= 0) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
void CheckZero(const size_t distributions, const size_t categories, const T *input, T *output,
               cudaStream_t cuda_stream) {
  CheckZeroKernel<<<GET_BLOCKS(distributions), GET_THREADS, 0, cuda_stream>>>(distributions, categories, input, output);
}

template <typename T>
__global__ void CheckNonNegKernel(const size_t size, const T *input, T *out) {
  out[0] = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (input[pos] < 0) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
void CheckNonNeg(const size_t size, const T *input, T *output, cudaStream_t cuda_stream) {
  CheckNonNegKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
}

template <typename T>
__global__ void NormInputKernel(T *input, const size_t distributions, const size_t categories) {
  size_t size = distributions * categories;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if ((pos + 1) % categories != 0) {
      int de_pos = (1 + pos / categories) * categories - 1;
      input[pos] /= input[de_pos];
    }
  }
  return;
}

template <typename T>
void NormInput(T *input, const size_t distributions, const size_t categories, cudaStream_t cuda_stream) {
  int count1 = distributions * categories;
  NormInputKernel<<<GET_BLOCKS(count1), GET_THREADS, 0, cuda_stream>>>(input, distributions, categories);
}

template <typename T>
__device__ int BinarySearchForMultinomial(T *start_addr, int size, T rand) {
  int start = 0;
  int end = size;
  while (end - start > 0) {
    int mid = start + (end - start) / 2;
    T mid_val = start_addr[mid];
    if (mid_val < rand) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  if (start == size) {
    start = size - 1;
  }
  return start;
}

template <typename T>
__global__ void MultinomialKernel(int seed, T *input, int num_sample, curandState *globalState, int *output,
                                  size_t distributions, size_t categories) {
  int count = num_sample * distributions;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    int j = i / num_sample % distributions;
    curand_init(seed, i, 0, &globalState[i]);
    auto rand = curand_uniform(&globalState[i]);
    int pick = BinarySearchForMultinomial(input + j * categories, categories, rand);
    output[i] = pick;
  }
  return;
}

template <typename T>
void Multinomial(int seed, int seed2, T *input, int num_sample, curandState *globalState, int *output,
                 size_t distributions, size_t categories, cudaStream_t cuda_stream) {
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  int count = distributions * num_sample;
  MultinomialKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, input, num_sample, globalState,
                                                                        output, distributions, categories);
  return;
}

template void Multinomial<float>(int seed, int seed2, float *input, int num_sample, curandState *globalState,
                                 int *output, size_t distributions, size_t categories, cudaStream_t cuda_stream);
template void CheckNonNeg<float>(const size_t size, const float *input, float *output, cudaStream_t cuda_stream);
template void CheckZero<float>(const size_t distributions, const size_t categories, const float *input, float *output,
                               cudaStream_t cuda_stream);
template void NormInput<float>(float *input, const size_t distributions, const size_t categories,
                               cudaStream_t cuda_stream);
