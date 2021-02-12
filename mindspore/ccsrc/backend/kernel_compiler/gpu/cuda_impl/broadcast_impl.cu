/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <iostream>

#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

// Basic function
template <typename T>
struct GreaterFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs > rhs ? true : false; }
};

template <typename T>
struct LessFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs < rhs ? true : false; }
};

template <typename T>
struct EqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs == rhs ? true : false; }
};

template <>
struct EqualFunc <half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return std::abs(__half2float(lhs) - __half2float(rhs)) < 1e-9 ? true : false;
  }
};

template <>
struct EqualFunc <float>  {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return std::abs(lhs - rhs) < 1e-9 ? true : false;
  }
};

template <typename T>
struct MinimumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs < rhs ? lhs : rhs; }
};

template <typename T>
struct MaximumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs > rhs ? lhs : rhs; }
};

template <typename T>
struct PowerFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return pow(lhs, rhs); }
};

template <>
struct PowerFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct PowerFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

template <typename T>
struct RealDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
};

template <typename T>
struct DivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
};

template <typename T>
struct MulFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
};

template <typename T>
struct SubFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs - rhs); }
};

template <typename T>
struct AddFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
};

// DivNoNan check if rhs is less than epsilon
template <typename T>
struct DivNoNanFunc {
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return rhs < kFloatEplison && rhs > -kFloatEplison ? 0.0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<int> {
  __device__ __host__ __forceinline__ int operator()(const int &lhs, const int &rhs) {
    return rhs == 0 ? 0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    if (__half2float(rhs) < (0.00007) && __half2float(rhs) > -0.00007) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct DivNoNanFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    if ((r.x < kFloatEplison && r.x > -kFloatEplison) || (r.y < kFloatEplison && r.y > -kFloatEplison)) {
      l.x = 0.0;
      l.y = 0.0;
    } else {
      l.x = l.x / r.x;
      l.y = l.y / r.y;
    }
    return __float22half2_rn(l);
  }
};

// convert to float to fix accuracy issue
template <typename T>
struct FloorDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return floorf(static_cast<float>(lhs) / static_cast<float>(rhs));
  }
};

template <>
struct FloorDivFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return floorf(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct FloorDivFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    l.x = floorf(l.x / r.x);
    l.y = floorf(l.y / r.y);
    return __float22half2_rn(l);
  }
};

template <typename T>
struct AbsGradFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T zero = 0.0;
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <>
struct AbsGradFunc<half2> {
  __device__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    half2 zero(0.0, 0.0);
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <typename T>
struct SquaredDifferenceFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T diff = lhs - rhs;
    return diff * diff;
  }
};

// Element-wise Comparison
template <typename T, typename Func>
__global__ void ElewiseCmpKernel(const int nums, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseCmp(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, bool *y, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return ElewiseCmpKernel<T, GreaterFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_LESS:
      return ElewiseCmpKernel<T, LessFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_EQUAL:
      return ElewiseCmpKernel<T, EqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const double *x0, const double *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const float *x0, const float *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const half *x0, const half *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const int *x0, const int *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const int8_t *x0, const int8_t *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const uint8_t *x0, const uint8_t *x1, bool *y,
                         cudaStream_t stream);
template void ElewiseCmp(const int &nums, enum BroadcastOpType op, const int64_t *x0, const int64_t *x1, bool *y,
                         cudaStream_t stream);

// Element-wise ArithMetic
template <typename T, typename Func>
__global__ void ElewiseArithKernel(const int nums, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseArithKernel(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_TYPE_MINIMUM:
      return ElewiseArithKernel<T, MinimumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MAXIMUM:
      return ElewiseArithKernel<T, MaximumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_POWER:
      return ElewiseArithKernel<T, PowerFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return ElewiseArithKernel<T, RealDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return ElewiseArithKernel<T, MulFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return ElewiseArithKernel<T, SubFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_ADD:
      return ElewiseArithKernel<T, AddFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_FLOORDIV:
      return ElewiseArithKernel<T, FloorDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_ABSGRAD:
      return ElewiseArithKernel<T, AbsGradFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return ElewiseArithKernel<T, DivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_DIVNONAN:
      return ElewiseArithKernel<T, DivNoNanFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_SQUARED_DIFFERENCE:
      return ElewiseArithKernel<T, SquaredDifferenceFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template <typename T>
void ElewiseArith(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  return ElewiseArithKernel(nums, op, x0, x1, y, stream);
}

template <>
void ElewiseArith(const int &nums, enum BroadcastOpType op, const half *x0, const half *x1, half *y,
                  cudaStream_t stream) {
  // `>` return true iff both half result are true. fallback to half
  if (nums % 2 == 0 && op != BROADCAST_TYPE_MINIMUM && op != BROADCAST_TYPE_MAXIMUM && op != BROADCAST_TYPE_ABSGRAD) {
    ElewiseArithKernel<half2>(nums / 2, op, reinterpret_cast<const half2 *>(x0), reinterpret_cast<const half2 *>(x1),
                              reinterpret_cast<half2 *>(y), stream);
  } else {
    return ElewiseArithKernel(nums, op, x0, x1, y, stream);
  }
}

template void ElewiseArith(const int &nums, enum BroadcastOpType op, const double *x0, const double *x1, double *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const float *x0, const float *x1, float *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const half *x0, const half *x1, half *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const int *x0, const int *x1, int *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const int8_t *x0, const int8_t *x1, int8_t *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const uint8_t *x0, const uint8_t *x1, uint8_t *y,
                           cudaStream_t stream);
template void ElewiseArith(const int &nums, enum BroadcastOpType op, const int64_t *x0, const int64_t *x1, int64_t *y,
                           cudaStream_t stream);

// Broadcast comparison
__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastCmpKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                   const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                   const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                   const size_t d6, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                  const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1, bool *y,
                  cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }

  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return BroadcastCmpKernel<T, GreaterFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_LESS:
      return BroadcastCmpKernel<T, LessFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_EQUAL:
      return BroadcastCmpKernel<T, EqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const double *x0,
                           const double *x1, bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const float *x0, const float *x1,
                           bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const half *x0, const half *x1,
                           bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int *x0, const int *x1,
                           bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int8_t *x0,
                           const int8_t *x1, bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const uint8_t *x0,
                           const uint8_t *x1, bool *y, cudaStream_t stream);
template void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int64_t *x0,
                           const int64_t *x1, bool *y, cudaStream_t stream);

// Broadcast Arithmetic
template <typename T, typename Func>
__global__ void BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                     const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                     const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                     const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                     const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                     const size_t d6, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                    const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1, T *y,
                    cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  switch (op) {
    case BROADCAST_TYPE_MAXIMUM:
      return BroadcastArithKernel<T, MaximumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MINIMUM:
      return BroadcastArithKernel<T, MinimumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_POWER:
      return BroadcastArithKernel<T, PowerFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return BroadcastArithKernel<T, RealDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return BroadcastArithKernel<T, MulFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return BroadcastArithKernel<T, SubFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_ADD:
      return BroadcastArithKernel<T, AddFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_FLOORDIV:
      return BroadcastArithKernel<T, FloorDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_ABSGRAD:
      return BroadcastArithKernel<T, AbsGradFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return BroadcastArithKernel<T, DivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_DIVNONAN:
      return BroadcastArithKernel<T, DivNoNanFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_SQUARED_DIFFERENCE:
      return BroadcastArithKernel<T, SquaredDifferenceFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const double *x0,
                             const double *x1, double *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const float *x0,
                             const float *x1, float *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const half *x0, const half *x1,
                             half *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int *x0, const int *x1,
                             int *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int8_t *x0,
                             const int8_t *x1, int8_t *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const uint8_t *x0,
                             const uint8_t *x1, uint8_t *y, cudaStream_t stream);
template void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int64_t *x0,
                             const int64_t *x1, int64_t *y, cudaStream_t stream);

// BroadcastTo
template <typename T>
__global__ void BroadcastToKernel(const size_t i0, const size_t i1, const size_t i2, const size_t i3, const size_t o0,
                                  const size_t o1, const size_t o2, const size_t o3, const T *input_addr,
                                  T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3) % o0;
    size_t j = pos / (o2 * o3) % o1;
    size_t k = pos / o3 % o2;
    size_t l = pos % o3;

    size_t input_idx = Index(i, i0) * i1 * i2 * i3 + Index(j, i1) * i2 * i3 + Index(k, i2) * i3 + Index(l, i3);
    output_addr[pos] = input_addr[input_idx];
  }
}

template <typename T>
void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &o0,
                 const size_t &o1, const size_t &o2, const size_t &o3, const T *input_addr, T *output_addr,
                 cudaStream_t stream) {
  size_t nums = o0 * o1 * o2 * o3;
  BroadcastToKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(i0, i1, i2, i3, o0, o1, o2, o3, input_addr,
                                                                  output_addr);
}

template void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &o0,
                          const size_t &o1, const size_t &o2, const size_t &o3, const float *input_addr,
                          float *output_addr, cudaStream_t stream);
template void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &o0,
                          const size_t &o1, const size_t &o2, const size_t &o3, const half *input_addr,
                          half *output_addr, cudaStream_t stream);
template void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &o0,
                          const size_t &o1, const size_t &o2, const size_t &o3, const int64_t *input_addr,
                          int64_t *output_addr, cudaStream_t stream);
