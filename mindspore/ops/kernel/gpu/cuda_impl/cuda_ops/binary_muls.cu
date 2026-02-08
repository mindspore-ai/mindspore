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
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_common.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_pub_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_mul_impl.cuh"

template <typename T>
struct IsFloatOrHalf : std::is_floating_point<T> {};
template <>
struct IsFloatOrHalf<half> : std::true_type {};

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kMul, In0_t, In1_t, Out_t> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const { return val0 * val1; }
};

// Specializations for float and half to resolve ambiguity
template <>
struct BinaryFunc<BinaryOpType::kMul, float, half, float> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ float operator()(float val0, half val1) const { return val0 * __half2float(val1); }
};
template <>
struct BinaryFunc<BinaryOpType::kMul, half, float, float> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ float operator()(half val0, float val1) const { return __half2float(val0) * val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kMul, double, half, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(double val0, half val1) const {
    return val0 * static_cast<double>(__half2float(val1));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kMul, half, double, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(half val0, double val1) const {
    return static_cast<double>(__half2float(val0)) * val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kMul, float, double, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(float val0, double val1) const {
    return static_cast<double>(val0) * val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kMul, double, float, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(double val0, float val1) const {
    return val0 * static_cast<double>(val1);
  }
};

// Specializations for half and integer/bool types to resolve ambiguity
#define MUL_HALF_INT_BOOL_SPEC(T)                                                             \
  template <>                                                                                 \
  struct BinaryFunc<BinaryOpType::kMul, half, T, half> {                                      \
    __device__ __host__ __forceinline__ BinaryFunc() {}                                       \
    __device__ __forceinline__ half operator()(half val0, T val1) const {                     \
      return __float2half(static_cast<float>(__half2float(val0) * static_cast<float>(val1))); \
    }                                                                                         \
  };                                                                                          \
  template <>                                                                                 \
  struct BinaryFunc<BinaryOpType::kMul, T, half, half> {                                      \
    __device__ __host__ __forceinline__ BinaryFunc() {}                                       \
    __device__ __forceinline__ half operator()(T val0, half val1) const {                     \
      return __float2half(static_cast<float>(static_cast<float>(val0) * __half2float(val1))); \
    }                                                                                         \
  };
MUL_HALF_INT_BOOL_SPEC(int8_t)
MUL_HALF_INT_BOOL_SPEC(uint8_t)
MUL_HALF_INT_BOOL_SPEC(int16_t)
MUL_HALF_INT_BOOL_SPEC(int32_t)
MUL_HALF_INT_BOOL_SPEC(int64_t)
MUL_HALF_INT_BOOL_SPEC(bool)

template <>
struct BinaryFunc<BinaryOpType::kMul, bool, bool, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(bool val0, bool val1) const { return val0 && val1; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kMul);
REGISTER_MUL_MIX_INT_TYPE(int8_t, uint8_t, int16_t);
REGISTER_MUL_MIX_INT_TYPE(int8_t, int16_t, int16_t);
REGISTER_MUL_MIX_INT_TYPE(int8_t, int32_t, int32_t);
REGISTER_MUL_MIX_INT_TYPE(int8_t, int64_t, int64_t);
REGISTER_MUL_MIX_INT_TYPE(uint8_t, int16_t, int16_t);
REGISTER_MUL_MIX_INT_TYPE(uint8_t, int32_t, int32_t);
REGISTER_MUL_MIX_INT_TYPE(uint8_t, int64_t, int64_t);
REGISTER_MUL_MIX_INT_TYPE(int16_t, int32_t, int32_t);
REGISTER_MUL_MIX_INT_TYPE(int16_t, int64_t, int64_t);
REGISTER_MUL_MIX_INT_TYPE(int32_t, int64_t, int64_t);

REGISTER_MUL_MIX_FLOAT_TYPE(half, float, float);
REGISTER_MUL_MIX_FLOAT_TYPE(half, double, double);
REGISTER_MUL_MIX_FLOAT_TYPE(float, double, double);

REGISTER_MUL_MIX_FLOAT_INT_TYPE(half, int8_t, half);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(half, uint8_t, half);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(half, int16_t, half);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(half, int32_t, half);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(half, int64_t, half);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(float, int8_t, float);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(float, uint8_t, float);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(float, int16_t, float);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(float, int32_t, float);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(float, int64_t, float);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(double, int8_t, double);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(double, uint8_t, double);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(double, int16_t, double);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(double, int32_t, double);
REGISTER_MUL_MIX_FLOAT_INT_TYPE(double, int64_t, double);

REGISTER_MUL_MIX_BOOL_TYPE(int8_t, bool, int8_t);
REGISTER_MUL_MIX_BOOL_TYPE(uint8_t, bool, uint8_t);
REGISTER_MUL_MIX_BOOL_TYPE(int16_t, bool, int16_t);
REGISTER_MUL_MIX_BOOL_TYPE(int32_t, bool, int32_t);
REGISTER_MUL_MIX_BOOL_TYPE(int64_t, bool, int64_t);
REGISTER_MUL_MIX_BOOL_TYPE(half, bool, half);
REGISTER_MUL_MIX_BOOL_TYPE(float, bool, float);
REGISTER_MUL_MIX_BOOL_TYPE(double, bool, double);

// MulNoNan
template <typename T>
struct BinaryFunc<BinaryOpType::kMulNoNan, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return rhs < Eps<T>() && rhs > -Eps<T>() ? 0.0 : (lhs * rhs);
  }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kMulNoNan, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return rhs == 0 ? 0 : (lhs * rhs);
  }
};
template <>
struct BinaryFunc<BinaryOpType::kMulNoNan, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    bool bool1 = __half2float(rhs) < (0.00001) && __half2float(rhs) > -0.00001;
    if (bool1) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) * __half2float(rhs));
  }
};

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kMulNoNan, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    Complex<Out_t> complex_rhs(rhs);
    if ((complex_rhs.real() < Eps<float>() && complex_rhs.real() > -Eps<float>()) ||
        (complex_rhs.imag() < Eps<float>() && complex_rhs.imag() > -Eps<float>())) {
      Complex<Out_t> res(0.0, 0.0);
      return res;
    }
    return lhs * rhs;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMulNoNan);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kMulNoNan);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMulNoNan);
