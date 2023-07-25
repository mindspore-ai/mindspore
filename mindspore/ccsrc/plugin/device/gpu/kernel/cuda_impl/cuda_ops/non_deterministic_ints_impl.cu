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

#include "non_deterministic_ints_impl.cuh"
template <typename T>
__global__ void NonDeterministicIntsKernel(int seed, curandStatePhilox4_32_10_t *globalState, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count / 4); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    uint4 i4 = curand4(&globalState[i]);
    output[i * 4] = i4.x;
    output[i * 4 + 1] = i4.y;
    output[i * 4 + 2] = i4.z;
    output[i * 4 + 3] = i4.w;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    curand_init(seed, 0, 0, &globalState[0]);
    uint4 i4 = curand4(&globalState[0]);
    size_t val = count % 4;
    for (size_t i = 0; i < val; i++) {
      output[count - i - 1] = (&i4.x)[i];
    }
  }
  return;
}

template <>
__global__ void NonDeterministicIntsKernel(int seed, curandStatePhilox4_32_10_t *globalState, int32_t *output,
                                           size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count / 4); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    uint4 i4 = curand4(&globalState[i]);
    output[i * 4] = i4.x;
    output[i * 4 + 1] = i4.y;
    output[i * 4 + 2] = i4.z;
    output[i * 4 + 3] = i4.w;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    curand_init(seed, 0, 0, &globalState[0]);
    uint4 i4 = curand4(&globalState[0]);
    size_t val = count % 4;
    for (size_t i = 0; i < val; i++) {
      output[count - i - 1] = (&i4.x)[i];
    }
  }
  return;
}

template <>
__global__ void NonDeterministicIntsKernel(int seed, curandStatePhilox4_32_10_t *globalState, int64_t *output,
                                           size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count / 2); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    uint4 i4 = curand4(&globalState[i]);
    output[i * 2] = ((int64_t)i4.x << 32) | i4.y;
    output[i * 2 + 1] = ((int64_t)i4.z << 32) | i4.w;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    curand_init(seed, 0, 0, &globalState[0]);
    uint4 i4 = curand4(&globalState[0]);
    if (count & 1) {
      output[count - 1] = ((int64_t)i4.x << 32) | i4.y;
    }
  }
  return;
}

template <>
__global__ void NonDeterministicIntsKernel(int seed, curandStatePhilox4_32_10_t *globalState, uint32_t *output,
                                           size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count / 4); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    uint4 i4 = curand4(&globalState[i]);
    output[i * 4] = i4.x;
    output[i * 4 + 1] = i4.y;
    output[i * 4 + 2] = i4.z;
    output[i * 4 + 3] = i4.w;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    curand_init(seed, 0, 0, &globalState[0]);
    uint4 i4 = curand4(&globalState[0]);
    size_t val = count % 4;
    for (size_t i = 0; i < val; i++) {
      output[count - i - 1] = (&i4.x)[i];
    }
  }
  return;
}

template <>
__global__ void NonDeterministicIntsKernel(int seed, curandStatePhilox4_32_10_t *globalState, uint64_t *output,
                                           size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count / 2); i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &globalState[i]);
    uint4 i4 = curand4(&globalState[i]);
    output[i * 2] = ((int64_t)i4.x << 32) | i4.y;
    output[i * 2 + 1] = ((int64_t)i4.z << 32) | i4.w;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    curand_init(seed, 0, 0, &globalState[0]);
    uint4 i4 = curand4(&globalState[0]);
    if (count & 1) {
      output[count - 1] = ((int64_t)i4.x << 32) | i4.y;
    }
  }
  return;
}

template <typename T>
cudaError_t LaunchNonDeterministicInts(curandStatePhilox4_32_10_t *globalState, T *output, size_t count,
                                       const uint32_t &device_id, cudaStream_t cuda_stream) {
  std::random_device rd;
  int seed = static_cast<int>(rd());
  NonDeterministicIntsKernel<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    seed, globalState, output, count);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t LaunchNonDeterministicInts<int32_t>(curandStatePhilox4_32_10_t *globalState,
                                                                         int32_t *output, size_t count,
                                                                         const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LaunchNonDeterministicInts<int64_t>(curandStatePhilox4_32_10_t *globalState,
                                                                         int64_t *output, size_t count,
                                                                         const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LaunchNonDeterministicInts<uint32_t>(curandStatePhilox4_32_10_t *globalState,
                                                                          uint32_t *output, size_t count,
                                                                          const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LaunchNonDeterministicInts<uint64_t>(curandStatePhilox4_32_10_t *globalState,
                                                                          uint64_t *output, size_t count,
                                                                          const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);
