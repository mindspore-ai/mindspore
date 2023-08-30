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

// XDivy check if lhs is less than epsilon, XDivy support half, float, double
template <typename T>
struct BinaryFunc<BinaryOpType::kXdivy, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return lhs < Eps<T>() && lhs > -Eps<T>() ? 0.0 : (lhs / rhs);
  }
};
template <>
struct BinaryFunc<BinaryOpType::kXdivy, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    if (__half2float(lhs) < (0.00007) && __half2float(lhs) > -0.00007) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kXdivy, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    Complex<Out_t> res(0.0, 0.0);
    Complex<Out_t> complex_lhs(lhs);
    Complex<Out_t> complex_rhs(rhs);
    if ((complex_lhs.real() >= Eps<float>() || complex_lhs.real() <= -Eps<float>()) ||
        (complex_lhs.imag() >= Eps<float>() || complex_lhs.imag() <= -Eps<float>())) {
      res = complex_lhs / complex_rhs;
    }
    return res;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kXdivy);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kXdivy);

template <typename T>
struct BinaryFunc<BinaryOpType::kFloorDiv, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return floor(lhs / rhs); }
};
template <>
struct BinaryFunc<BinaryOpType::kFloorDiv, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    return __float2half_rn(floorf(__half2float(lhs) / __half2float(rhs)));
  }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kFloorDiv, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return static_cast<T>(floor(static_cast<float>(lhs) / static_cast<float>(rhs)));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kFloorDiv);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kFloorDiv);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kFloorDiv);

template <typename T>
struct BinaryFunc<BinaryOpType::kTruncateDiv, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return static_cast<T>(trunc(lhs / rhs)); }
};
template <>
struct BinaryFunc<BinaryOpType::kTruncateDiv, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) const { return htrunc(lhs / rhs); }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kTruncateDiv, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return static_cast<T>(trunc(static_cast<float>(lhs) / static_cast<float>(rhs)));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kTruncateDiv);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kTruncateDiv);
