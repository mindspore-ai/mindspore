/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/parameterized_truncated_normal_impl.cuh"
#include <stdint.h>
#include <cmath>
#include <limits>
#include "include/cuda_device_runtime_api.h"
#include "include/cuda_fp16.h"

// sqrt for all the dtypes
template <typename T>
__device__ __forceinline__ T sqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half sqrtFunc(half input) {
  return hsqrt(input);
}

// isinf for all the dtypes
template <typename T>
__device__ __forceinline__ T isinfFunc(T input) {
  return isinf(input);
}

template <>
__device__ __forceinline__ half isinfFunc(half input) {
  return __hisinf(input);
}

// exp for all the dtypes
template <typename T>
__device__ __forceinline__ T expFunc(T input) {
  return exp(input);
}

template <>
__device__ __forceinline__ half expFunc(half input) {
  return hexp(input);
}

// log for all the dtypes
template <typename T>
__device__ __forceinline__ T logFunc(T input) {
  return log(input);
}

template <>
__device__ __forceinline__ half logFunc(half input) {
  return hlog(input);
}

// in abnormal scenarios, output "nan" for fp32 and fp64, "65500" for half
template <typename T>
__device__ __forceinline__ T nanOutput(T num) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <>
__device__ __forceinline__ half nanOutput(half num) {
  return half(65500);
}

template <typename T>
__global__ void __launch_bounds__(1024)
  Generate(uint64_t seed, uint64_t seed_offset, int64_t batch_size, int64_t samples_per_batch, T *mean, T *stdevs,
           T *min, T *max, T *output, bool scalar_mean, bool scalar_stdevs, bool scalar_min, bool scalar_max) {
  // set bounds
  const T kStdDevsInsideBoundsToUseRandnSampler = T(1.7);

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * samples_per_batch;
       index += blockDim.x * gridDim.x) {
    // use philox generator to generate random numbers
    curandStatePhilox4_32_10_t state;
    curand_init(seed, index, seed_offset, &state);

    // set values
    const int batch_id = index / samples_per_batch;
    T mean_value = mean[scalar_mean ? 0 : batch_id];
    T stdevs_ori = stdevs[scalar_stdevs ? 0 : batch_id];
    T min_value = min[scalar_min ? 0 : batch_id];
    T max_value = max[scalar_max ? 0 : batch_id];

    // flip the distribution if we can make the lower bound positive
    T stdevs_value;
    if (isinfFunc(min_value) || max_value < mean_value) {
      T temp = min_value;
      min_value = max_value;
      max_value = temp;
      stdevs_value = -stdevs_ori;
    } else {
      stdevs_value = stdevs_ori;
    }

    // calculate normalized samples, then scale them
    const T normMin = (min_value - mean_value) / stdevs_value;
    const T normMax = (max_value - mean_value) / stdevs_value;

    // determine the method to use
    const T sqrtFactor = sqrtFunc((normMin * normMin) + T(4));
    const T cutOff = T(2) * expFunc(T(0.5) + (normMin * (normMin - sqrtFactor)) / T(4)) / (normMin + sqrtFactor);
    const T diff = normMax - normMin;

    // calculate by different methods
    if (!(stdevs_ori > T(0) && normMin < normMax && (!isinfFunc(normMin) || !isinfFunc(normMax)))) {
      output[index] = nanOutput(T(0));
    } else if (((normMin < -kStdDevsInsideBoundsToUseRandnSampler) && (normMax >= T(0.))) ||
               ((normMax > kStdDevsInsideBoundsToUseRandnSampler) && (normMin <= T(0.)))) {
      GenarateCase1(&state, index, stdevs_value, mean_value, normMin, normMax, output);
    } else if (diff < cutOff) {
      GenarateCase2(&state, index, stdevs_value, mean_value, normMin, normMax, output);
    } else {
      GenarateCase3(&state, index, stdevs_value, mean_value, normMin, normMax, output);
    }
  }
}

template <typename T>
__device__ void GenarateCase1(curandStatePhilox4_32_10_t *state, int index, T stdevs_value, T mean_value, T normMin,
                              T normMax, T *output) {
  T randn[kCounterNum];
  int numIterations = 0;
  while (numIterations < kMaxIterations) {
    float4 rand = curand_normal4(state);
    randn[0] = static_cast<T>(rand.x);
    randn[1] = static_cast<T>(rand.y);
    randn[2] = static_cast<T>(rand.z);
    randn[3] = static_cast<T>(rand.w);
#pragma unroll
    for (int i = 0; i < kCounterNum; i++) {
      if ((randn[i] >= normMin) && randn[i] <= normMax) {
        output[index] = randn[i] * stdevs_value + mean_value;
        numIterations = kMaxIterations;
        break;
      } else if (numIterations + 1 == kMaxIterations) {
        // If we did not successfully sample after all these iterations, something is wrong. Output a nan.
        output[index] = nanOutput(T(0));
        numIterations = kMaxIterations;
        break;
      } else {
        numIterations++;
      }
    }
  }
}

template <typename T>
__device__ void GenarateCase2(curandStatePhilox4_32_10_t *state, int index, T stdevs_value, T mean_value, T normMin,
                              T normMax, T *output) {
  T randu_1[kCounterNum];
  T randu_2[kCounterNum];
  T z[kCounterNum];
  T g[kCounterNum];

  const T plusFactor = (normMin < T(0)) ? T(0) : T(normMin * normMin);

  int numIterations = 0;
  while (numIterations < kMaxIterations) {
    float4 rand_1 = curand_uniform4(state);
    randu_1[0] = static_cast<T>(rand_1.x);
    randu_1[1] = static_cast<T>(rand_1.y);
    randu_1[2] = static_cast<T>(rand_1.z);
    randu_1[3] = static_cast<T>(rand_1.w);
#pragma unroll
    for (int i = 0; i < kCounterNum; i++) {
      z[i] = randu_1[i] * (normMax - normMin) + normMin;
    }
#pragma unroll
    for (int i = 0; i < kCounterNum; i++) {
      g[i] = (plusFactor - z[i] * z[i]) / T(2);
    }
    float4 rand_2 = curand_uniform4(state);
    randu_2[0] = static_cast<T>(rand_2.x);
    randu_2[1] = static_cast<T>(rand_2.y);
    randu_2[2] = static_cast<T>(rand_2.z);
    randu_2[3] = static_cast<T>(rand_2.w);
#pragma unroll
    for (int i = 0; i < kCounterNum; i++) {
      bool accept = randu_2[i] <= expFunc(g[i]);
      if (accept) {
        output[index] = z[i] * stdevs_value + mean_value;
        numIterations = kMaxIterations;
        break;
      } else if (numIterations + 1 >= kMaxIterations) {
        output[index] = nanOutput(T(0));
        numIterations = kMaxIterations;
        break;
      } else {
        numIterations++;
      }
    }
  }
}

template <typename T>
__device__ void GenarateCase3(curandStatePhilox4_32_10_t *state, int index, T stdevs_value, T mean_value, T normMin,
                              T normMax, T *output) {
  T randu[kCounterNum];
  const T alpha = (normMin + sqrtFunc((normMin * normMin) + T(4))) / T(2);
  int numIterations = 0;
  while (numIterations < kMaxIterations) {
    float4 rand = curand_uniform4(state);
    randu[0] = static_cast<T>(rand.x);
    randu[1] = static_cast<T>(rand.y);
    randu[2] = static_cast<T>(rand.z);
    randu[3] = static_cast<T>(rand.w);
#pragma unroll
    for (int i = 0; i < kCounterNum; i += 2) {
      const T z = -logFunc(randu[i]) / alpha + normMin;
      const T x = normMin < alpha ? alpha - z : normMin - alpha;
      const T g = expFunc(-x * x / T(2));
      const T u = randu[i + 1];
      bool accept = (u <= g && z < normMax);
      if (accept) {
        output[index] = z * stdevs_value + mean_value;
        numIterations = kMaxIterations;
        break;
      } else if (numIterations + 1 >= kMaxIterations) {
        output[index] = nanOutput(T(0));
        numIterations = kMaxIterations;
        break;
      } else {
        numIterations++;
      }
    }
  }
}

template <typename T>
cudaError_t ParameterizedTruncatedNormal(uint64_t seed, uint64_t seed_offset, int64_t batch_size,
                                         int64_t samples_per_batch, T *mean, T *stdevs, T *min, T *max, T *output,
                                         bool scalar_mean, bool scalar_stdevs, bool scalar_min, bool scalar_max,
                                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  Generate<<<CUDA_BLOCKS(device_id, batch_size * samples_per_batch), CUDA_THREADS(device_id), device_id, cuda_stream>>>(
    seed, seed_offset, batch_size, samples_per_batch, mean, stdevs, min, max, output, scalar_mean, scalar_stdevs,
    scalar_min, scalar_max);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ParameterizedTruncatedNormal<half>(
  uint64_t seed, uint64_t seed_offset, int64_t batch_size, int64_t samples_per_batch, half *mean, half *stdevs,
  half *min, half *max, half *output, bool scalar_mean, bool scalar_stdevs, bool scalar_min, bool scalar_max,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ParameterizedTruncatedNormal<float>(
  uint64_t seed, uint64_t seed_offset, int64_t batch_size, int64_t samples_per_batch, float *mean, float *stdevs,
  float *min, float *max, float *output, bool scalar_mean, bool scalar_stdevs, bool scalar_min, bool scalar_max,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ParameterizedTruncatedNormal<double>(
  uint64_t seed, uint64_t seed_offset, int64_t batch_size, int64_t samples_per_batch, double *mean, double *stdevs,
  double *min, double *max, double *output, bool scalar_mean, bool scalar_stdevs, bool scalar_min, bool scalar_max,
  const uint32_t &device_id, cudaStream_t cuda_stream);
