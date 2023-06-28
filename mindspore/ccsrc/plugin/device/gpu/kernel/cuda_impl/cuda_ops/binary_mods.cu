/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_common.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_pub_impl.cuh"

template <typename T>
struct BinaryFunc<BinaryOpType::kMod, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return lhs % rhs; }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kMod, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return fmod(lhs, rhs); }
};
template <>
struct BinaryFunc<BinaryOpType::kMod, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    return __float2half(fmod(__half2float(lhs), __half2float(rhs)));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMod);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMod);

template <typename T>
struct BinaryFunc<BinaryOpType::kFloorMod, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    T res = lhs - floorf(lhs / rhs) * rhs;
    res = (abs(res) > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kFloorMod, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - floorf(l / r) * r;
    res = (abs(res) > 1e-9) && ((res < 0.0) != (r < 0.0)) ? res + r : res;
    return __float2half_rn(res);
  }
};

template <typename T>
struct BinaryFunc<BinaryOpType::kFloorMod, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    T res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kFloorMod);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kFloorMod);

template <typename T>
struct BinaryFunc<BinaryOpType::kTruncateMod, T, T, T> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    T res = static_cast<T>(lhs - static_cast<int>(lhs / rhs) * rhs);
    return res;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kTruncateMod, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - static_cast<int>(l / r) * r;
    return __float2half_rn(res);
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kTruncateMod);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kTruncateMod);
