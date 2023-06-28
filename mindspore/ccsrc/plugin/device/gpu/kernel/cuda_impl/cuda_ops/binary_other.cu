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

// Xlogy check if lhs is less than epsilon, Xlogy support half, float, double
template <typename T>
struct BinaryFunc<BinaryOpType::kXlogy, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return lhs < Eps<T>() && lhs > -Eps<T>() ? 0.0 : (lhs * log(rhs));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kXlogy, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    half zero = 0.0;
    half eps = 6.105e-5;
    return (lhs < eps && lhs > -eps) ? zero : __float2half_rn(__half2float(lhs) * log(__half2float(rhs)));
  }
};
template <typename IN, typename Out_t>
__device__ __host__ __forceinline__ Out_t CalMid(const IN &inp_val) {
  Out_t res(0.5 * log(inp_val * inp_val * 2), 1.0);
  return res;
}
template <>
__device__ __host__ __forceinline__ Complex<float> CalMid(const Complex<float> &inp_val) {
  Complex<float> res(0.5 * log(inp_val.real() * inp_val.real() + inp_val.imag() * inp_val.imag()),
                     atan2(inp_val.imag(), inp_val.real()));
  return res;
}
template <>
__device__ __host__ __forceinline__ Complex<double> CalMid(const Complex<double> &inp_val) {
  Complex<double> res(0.5 * log(inp_val.real() * inp_val.real() + inp_val.imag() * inp_val.imag()),
                      atan2(inp_val.imag(), inp_val.real()));
  return res;
}

template <typename IN>
__device__ __host__ __forceinline__ bool IsZero(const IN &inp_val) {
  return inp_val < Eps<IN>() && inp_val > -Eps<IN>();
}
template <>
__device__ __host__ __forceinline__ bool IsZero(const Complex<float> &inp_val) {
  return inp_val.real() < Eps<float>() && inp_val.real() > -Eps<float>() && inp_val.imag() < Eps<float>() &&
         inp_val.imag() > -Eps<float>();
}
template <>
__device__ __host__ __forceinline__ bool IsZero(const Complex<double> &inp_val) {
  return inp_val.real() < Eps<double>() && inp_val.real() > -Eps<double>() && inp_val.imag() < Eps<double>() &&
         inp_val.imag() > -Eps<double>();
}
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kXlogy, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    if (IsZero<In0_t>(lhs)) {
      Complex<Out_t> res(0.0, 0.0);
      return res;
    }
    Complex<Out_t> mid = CalMid<In1_t, Complex<Out_t>>(rhs);
    return lhs * mid;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kXlogy);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kXlogy);

template <typename T>
struct BinaryFunc<BinaryOpType::kSquaredDifference, T, T, T, typename std::is_arithmetic<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    T diff = lhs - rhs;
    return diff * diff;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kSquaredDifference, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    half diff = lhs - rhs;
    return diff * diff;
  }
};
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kSquaredDifference, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    Complex<Out_t> diff = lhs - rhs;
    Complex<Out_t> conj_diff(diff.real(), -diff.imag());
    return conj_diff * diff;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kSquaredDifference);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kSquaredDifference);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kSquaredDifference);

template <typename T>
struct BinaryFunc<BinaryOpType::kAtan2, T, T, T, typename std::is_floating_point<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return atan2(lhs, rhs); }
};
template <typename T>
struct BinaryFunc<BinaryOpType::kAtan2, T, T, T, typename std::is_integral<T>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    return static_cast<T>(atan2(static_cast<float>(lhs), static_cast<float>(rhs)));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kAtan2, half, half, half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = atan2f(l, r);
    return __float2half_rn(res);
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kAtan2);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kAtan2);

template <typename T>
struct BinaryFunc<BinaryOpType::kAbsGrad, T, T, T> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) const {
    T zero = 0.0;
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kAbsGrad);
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kAbsGrad);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kAbsGrad);

// now only for complex op
#define REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX(op)                                                                      \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, float, float, Complex<float>>(               \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, float *input0, float *input1, Complex<float> *output, size_t device_id,    \
    cudaStream_t cuda_stream);                                                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, double, double, Complex<double>>(            \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, double *input0, double *input1, Complex<double> *output, size_t device_id, \
    cudaStream_t cuda_stream)

template <typename T>
struct BinaryFunc<BinaryOpType::kComplex, T, T, Complex<T>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const T &rhs) const {
    return Complex<T>(lhs, rhs);
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX(BinaryOpType::kComplex);
