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

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kMul, In0_t, In1_t, Out_t> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const { return val0 * val1; }
};

template <>
struct BinaryFunc<BinaryOpType::kMul, bool, bool, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(bool val0, bool val1) const { return val0 && val1; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kMul);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kMul);

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
