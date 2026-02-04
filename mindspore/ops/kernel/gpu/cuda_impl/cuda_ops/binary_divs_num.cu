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
#include "kernel/gpu/cuda_impl/cuda_ops/binary_div_impl.cuh"

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kDiv, In0_t, In1_t, Out_t> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const {
    // For mixed dtypes, compute in Out_t to match frontend promoted output type.
    return static_cast<Out_t>(val0) / static_cast<Out_t>(val1);
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, Complex<float>, Complex<float>, Complex<float>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Complex<float> operator()(Complex<float> val0, Complex<float> val1) const {
    return val0 / val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, Complex<double>, Complex<double>, Complex<double>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Complex<double> operator()(Complex<double> val0, Complex<double> val1) const {
    return val0 / val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ half operator()(half val0, half val1) const { return val0 / val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, float, float, float> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ float operator()(float val0, float val1) const { return val0 / val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, double, double, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(double val0, double val1) const { return val0 / val1; }
};

// Specializations for float/double output with half input (half cannot be directly cast to float/double).
template <>
struct BinaryFunc<BinaryOpType::kDiv, float, half, float> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ float operator()(float val0, half val1) const { return val0 / __half2float(val1); }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, half, float, float> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ float operator()(half val0, float val1) const { return __half2float(val0) / val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, double, half, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(double val0, half val1) const {
    return val0 / static_cast<double>(__half2float(val1));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, half, double, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(half val0, double val1) const {
    return static_cast<double>(__half2float(val0)) / val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, float, double, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(float val0, double val1) const {
    return static_cast<double>(val0) / val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDiv, double, float, double> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ double operator()(double val0, float val1) const {
    return val0 / static_cast<double>(val1);
  }
};

// Specializations for half output with integer/bool types (resolve conversion ambiguity).
#define DIV_HALF_INT_BOOL_SPEC(T)                                            \
  template <>                                                                \
  struct BinaryFunc<BinaryOpType::kDiv, half, T, half> {                     \
    __device__ __host__ __forceinline__ BinaryFunc() {}                      \
    __device__ __forceinline__ half operator()(half val0, T val1) const {    \
      return __float2half_rn(__half2float(val0) / static_cast<float>(val1)); \
    }                                                                        \
  };                                                                         \
  template <>                                                                \
  struct BinaryFunc<BinaryOpType::kDiv, T, half, half> {                     \
    __device__ __host__ __forceinline__ BinaryFunc() {}                      \
    __device__ __forceinline__ half operator()(T val0, half val1) const {    \
      return __float2half_rn(static_cast<float>(val0) / __half2float(val1)); \
    }                                                                        \
  };
DIV_HALF_INT_BOOL_SPEC(int8_t)
DIV_HALF_INT_BOOL_SPEC(uint8_t)
DIV_HALF_INT_BOOL_SPEC(int16_t)
DIV_HALF_INT_BOOL_SPEC(int32_t)
DIV_HALF_INT_BOOL_SPEC(int64_t)
DIV_HALF_INT_BOOL_SPEC(bool)

REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kDiv);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kDiv);
REGISTER_BINARY_OP_CUDA_DIV_INT_TYPE(BinaryOpType::kDiv);

// Mixed dtype instantiations to match Div frontend InferType (no bfloat16).
// Mixed float types
REGISTER_DIV_MIX_TYPE(half, float, float)
REGISTER_DIV_MIX_TYPE(half, double, double)
REGISTER_DIV_MIX_TYPE(float, double, double)
// float16 with bool/int -> float16
REGISTER_DIV_MIX_TYPE(half, bool, half)
REGISTER_DIV_MIX_TYPE(half, int8_t, half)
REGISTER_DIV_MIX_TYPE(half, uint8_t, half)
REGISTER_DIV_MIX_TYPE(half, int16_t, half)
REGISTER_DIV_MIX_TYPE(half, int32_t, half)
REGISTER_DIV_MIX_TYPE(half, int64_t, half)
// float32 with bool/int -> float32
REGISTER_DIV_MIX_TYPE(float, bool, float)
REGISTER_DIV_MIX_TYPE(float, int8_t, float)
REGISTER_DIV_MIX_TYPE(float, uint8_t, float)
REGISTER_DIV_MIX_TYPE(float, int16_t, float)
REGISTER_DIV_MIX_TYPE(float, int32_t, float)
REGISTER_DIV_MIX_TYPE(float, int64_t, float)
// float64 with bool/int -> float64
REGISTER_DIV_MIX_TYPE(double, bool, double)
REGISTER_DIV_MIX_TYPE(double, int8_t, double)
REGISTER_DIV_MIX_TYPE(double, uint8_t, double)
REGISTER_DIV_MIX_TYPE(double, int16_t, double)
REGISTER_DIV_MIX_TYPE(double, int32_t, double)
REGISTER_DIV_MIX_TYPE(double, int64_t, double)
// integral/bool mixed pairs -> float32 (Div always outputs float32 for integral/bool promote type)
REGISTER_DIV_SAME_TYPE(bool, float)
REGISTER_DIV_MIX_TYPE(bool, int8_t, float)
REGISTER_DIV_MIX_TYPE(bool, uint8_t, float)
REGISTER_DIV_MIX_TYPE(bool, int16_t, float)
REGISTER_DIV_MIX_TYPE(bool, int32_t, float)
REGISTER_DIV_MIX_TYPE(bool, int64_t, float)
REGISTER_DIV_MIX_TYPE(int8_t, uint8_t, float)
REGISTER_DIV_MIX_TYPE(int8_t, int16_t, float)
REGISTER_DIV_MIX_TYPE(int8_t, int32_t, float)
REGISTER_DIV_MIX_TYPE(int8_t, int64_t, float)
REGISTER_DIV_MIX_TYPE(uint8_t, int16_t, float)
REGISTER_DIV_MIX_TYPE(uint8_t, int32_t, float)
REGISTER_DIV_MIX_TYPE(uint8_t, int64_t, float)
REGISTER_DIV_MIX_TYPE(int16_t, int32_t, float)
REGISTER_DIV_MIX_TYPE(int16_t, int64_t, float)
REGISTER_DIV_MIX_TYPE(int32_t, int64_t, float)

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kRealDiv, In0_t, In1_t, Out_t> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Out_t operator()(const In0_t &lhs, const In1_t &rhs) const { return (lhs / rhs); }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kRealDiv);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kRealDiv);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kRealDiv);

// DivNoNan check if rhs is less than epsilon
template <typename T>
struct BinaryFunc<BinaryOpType::kDivNoNan, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return rhs < Eps<T>() && rhs > -Eps<T>() ? 0.0 : (lhs / rhs);
  }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kDivNoNan, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ int operator()(const T &lhs, const T &rhs) const {
    return rhs == 0 ? 0 : (lhs / rhs);
  }
};
template <>
struct BinaryFunc<BinaryOpType::kDivNoNan, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    if (__half2float(rhs) < (0.00001) && __half2float(rhs) > -0.00001) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kDivNoNan, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    Complex<Out_t> complex_rhs(rhs);
    if ((complex_rhs.real() < Eps<float>() && complex_rhs.real() > -Eps<float>()) ||
        (complex_rhs.imag() < Eps<float>() && complex_rhs.imag() > -Eps<float>())) {
      Complex<Out_t> res(0.0, 0.0);
      return res;
    }
    return lhs / rhs;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kDivNoNan);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kDivNoNan);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kDivNoNan);
