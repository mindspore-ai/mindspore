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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/polygamma_impl.cuh"
#include <limits>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"
constexpr size_t kThreadsPerBlock = 256;

template <typename T>
inline __device__ T trigamma(const T input) {
  double sign = +1;
  double result = 0;
  constexpr double PI = static_cast<double>(3.141592653589793238462643383279502);
  auto y = static_cast<double>(input);
  if (y < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(PI * y);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    y = 1 - y;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (y * y);
    y += 1;
  }
  const double ixx = 1 / (y * y);
  result += (1 + 1 / (2 * y) + ixx * (1. / 6 - ixx * (1. / 30 - ixx * (1. / 42)))) / y;
  return static_cast<T>(sign * result);
}

template <>
inline __device__ float trigamma(const float input) {
  float sign = +1;
  float result = 0;
  constexpr float PI = static_cast<float>(3.141592653589793238462643383279502);
  auto y = static_cast<float>(input);
  if (y < 0.5f) {
    sign = -1;
    const float sin_pi_x = sinf(PI * y);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    y = 1 - y;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (y * y);
    y += 1;
  }
  const float ixx = 1 / (y * y);
  result += (1 + 1 / (2 * y) + ixx * (1.f / 6 - ixx * (1.f / 30 - ixx * (1.f / 42)))) / y;
  return sign * result;
}

template <typename T>
inline __device__ T calc_polygamma(const int64_t y, const T input) {
  auto poly_n = y;
  auto zeta_n = static_cast<double>(poly_n);
  constexpr double one = static_cast<double>(1.0);
  double poly_result = ((poly_n % 2) ? one : -one) * exp(lgamma(zeta_n + one));
  double p = zeta_n + one;
  double q = static_cast<double>(input);
  const double MACHEP = static_cast<double>(1.11022302462515654042E-16);
  constexpr double zero = static_cast<double>(0.0);
  constexpr double half = static_cast<double>(0.5);
  static const double A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691*/
    7.47242496e10,
    -2.950130727918164224e12,  /*1.067062284288e16/3617*/
    1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
    -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
    1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
    -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };
  int i = 0;
  double a, b, k, s, t, w;
  if (p == one) {
    return static_cast<T>(std::numeric_limits<double>::infinity() * poly_result);
  }
  if (p < one) {
    return static_cast<T>(std::numeric_limits<double>::quiet_NaN() * poly_result);
  }
  if (q <= zero) {
    if (q == floor(q)) {
      return static_cast<T>(std::numeric_limits<double>::infinity() * poly_result);
    }
    if (p != floor(p)) {
      return static_cast<T>(std::numeric_limits<double>::quiet_NaN() * poly_result);
    }
  }
  s = pow(q, -p);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= static_cast<double>(9.0))) {
    i += 1;
    a += one;
    b = pow(a, -p);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<T>(s * poly_result);
    }
  }

  w = a;
  s += b * w / (p - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= p + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = fabs(t / s);
    if (t < MACHEP) {
      return static_cast<T>(s * poly_result);
    }
    k += one;
    a *= p + k;
    b /= w;
    k += one;
  }
  return static_cast<T>(s * poly_result);
}

template <>
inline __device__ float calc_polygamma(const int64_t y, const float input) {
  auto poly_n = y;
  auto zeta_n = static_cast<float>(poly_n);
  constexpr float one = static_cast<float>(1.0);
  float poly_result = ((poly_n % static_cast<int64_t>(2)) ? one : -one) * exp(lgammaf(zeta_n + one));
  float p = zeta_n + one;
  float q = static_cast<float>(input);
  constexpr float MACHEP = static_cast<float>(1.11022302462515654042E-16);
  constexpr float zero = static_cast<float>(0.0);
  constexpr float half = static_cast<float>(0.5);
  static const float A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691*/
    7.47242496e10,
    -2.950130727918164224e12,  /*1.067062284288e16/3617*/
    1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
    -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
    1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
    -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };
  int i = 0;
  float a, b, k, s, t, w;
  if (p == one) {
    return std::numeric_limits<float>::infinity() * poly_result;
  }
  if (p < one) {
    return std::numeric_limits<float>::quiet_NaN() * poly_result;
  }
  if (q <= zero) {
    if (q == floor(q)) {
      return std::numeric_limits<float>::infinity() * poly_result;
    }
    if (p != floor(p)) {
      return std::numeric_limits<float>::quiet_NaN() * poly_result;
    }
  }
  s = pow(q, -p);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= static_cast<float>(9.0))) {
    i += 1;
    a += one;
    b = pow(a, -p);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return s * poly_result;
    }
  }

  w = a;
  s += b * w / (p - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= p + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = fabs(t / s);
    if (t < MACHEP) {
      return s * poly_result;
    }
    k += one;
    a *= p + k;
    b /= w;
    k += one;
  }
  return s * poly_result;
}

template <uint vec_size, typename T>
__device__ __forceinline__ void VectorizedCallTri(const T *input, T *output, uint offset) {
  uint tid = threadIdx.x;
  using VecT = cuda::elementwise::AlignVec<T, vec_size>;
  auto vec_input = reinterpret_cast<const VecT *>(input + offset);
  auto vec_output = reinterpret_cast<VecT *>(output + offset);
  VecT cache = vec_input[tid];
  VecT out1{0};

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    auto output_pair = trigamma(cache.elements_[j]);
    out1.elements_[j] = output_pair;
  }
  vec_output[tid] = out1;
}

template <uint vec_size, typename T>
__device__ __forceinline__ void NormalCallTri(const T *input, T *output, uint offset, uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      auto output_pair = trigamma(input[index]);
      output[index] = output_pair;
    }
  }
}

template <uint vec_size, typename T1, typename T2>
__device__ __forceinline__ void VectorizedCall(const T1 *a, const T2 *input, T2 *output, uint offset) {
  uint tid = threadIdx.x;

  using VecT = cuda::elementwise::AlignVec<T2, vec_size>;
  auto a_int = static_cast<int64_t>(a[0]);
  auto vec_input = reinterpret_cast<const VecT *>(input + offset);
  auto vec_output = reinterpret_cast<VecT *>(output + offset);
  VecT cache = vec_input[tid];
  VecT out1{0};

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    auto output_pair = calc_polygamma(a_int, cache.elements_[j]);
    out1.elements_[j] = output_pair;
  }
  vec_output[tid] = out1;
}

template <uint vec_size, typename T1, typename T2>
__device__ __forceinline__ void NormalCall(const T1 *a, const T2 *input, T2 *output, uint offset, uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  auto a_int = static_cast<int64_t>(a[0]);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      auto output_pair = calc_polygamma(a_int, input[index]);
      output[index] = output_pair;
    }
  }
}

template <uint vec_size, typename T1, typename T2>
__global__ void CalPolygammaKernel(size_t num_count, const T1 *a, const T2 *input, T2 *output) {
  auto y = a[0];
  uint elements_per_block = kThreadsPerBlock * vec_size;
  if (y == T1(1)) {
    for (uint offset = elements_per_block * blockIdx.x; offset < num_count; offset += elements_per_block * gridDim.x) {
      uint remaining = num_count - offset;
      if (remaining < elements_per_block) {
        NormalCallTri<vec_size, T2>(input, output, offset, remaining);
      } else {
        VectorizedCallTri<vec_size, T2>(input, output, offset);
      }
    }
  } else if (y > T1(1)) {
    for (uint offset = elements_per_block * blockIdx.x; offset < num_count; offset += elements_per_block * gridDim.x) {
      uint remaining = num_count - offset;
      if (remaining < elements_per_block) {
        NormalCall<vec_size, T1, T2>(a, input, output, offset, remaining);
      } else {
        VectorizedCall<vec_size, T1, T2>(a, input, output, offset);
      }
    }
  }
  return;
}

template <typename T1, typename T2>
cudaError_t CalPolygamma(const size_t num_count, const T1 *a, const T2 *input, T2 *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  constexpr size_t vec_size = cuda::elementwise::VecSize<T2>();
  const size_t block_x = kThreadsPerBlock < num_count ? kThreadsPerBlock : num_count;
  const size_t elements_per_block = kThreadsPerBlock * vec_size;
  const size_t grid_x = UP_DIV(num_count, elements_per_block);
  CalPolygammaKernel<vec_size, T1, T2><<<grid_x, block_x, 0, cuda_stream>>>(num_count, a, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int32_t *a, const float *input,
                                                  float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int32_t *a, const double *input,
                                                  double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int32_t *a, const half *input,
                                                  half *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int64_t *a, const float *input,
                                                  float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int64_t *a, const double *input,
                                                  double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPolygamma(const size_t num_count, const int64_t *a, const half *input,
                                                  half *output, const uint32_t &device_id, cudaStream_t cuda_stream);
