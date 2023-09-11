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
struct BinaryFunc<BinaryOpType::kPow, In0_t, In1_t, Out_t, typename std::is_floating_point<Out_t>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Out_t operator()(const In0_t &lhs, const In1_t &rhs) const {
    return static_cast<Out_t>(pow(static_cast<double>(lhs), static_cast<double>(rhs)));
  }
};

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kPow, In0_t, In1_t, Out_t, typename std::is_integral<Out_t>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Out_t operator()(const In0_t &lhs, const In1_t &rhs) const {
    In0_t base = lhs;
    In1_t exp = rhs;
    if (exp < 0) {
      if (base == 1) {
        return 1;
      } else if (base == -1) {
        return ((-exp) % 2 == 1) ? -1 : 1;
      } else {
        return 0;
      }
    }
    Out_t result = 1;
    while (exp) {
      if (exp & 1) {
        result *= base;
      }
      exp = exp >> 1;
      base *= base;
    }
    return result;
  }
};

template <>
struct BinaryFunc<BinaryOpType::kPow, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<BinaryOpType::kPow, In0_t, In1_t, Complex<Out_t>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<Out_t> operator()(const In0_t &lhs, const In1_t &rhs) const {
    Complex<Out_t> result;
#if defined(__CUDACC__)
    auto thrust_res = thrust::pow(thrust::complex<Out_t>(lhs), thrust::complex<Out_t>(rhs));
    result.real(thrust_res.real());
    result.imag(thrust_res.imag());
#else
    std::complex<Out_t> lhs_complex(lhs);
    std::complex<Out_t> rhs_complex(rhs);
    std::complex<Out_t> host_res = std::pow(lhs_complex, rhs_complex);
    result.real(host_res.real());
    result.imag(host_res.imag());
#endif
    return result;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kPow);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kPow);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kPow);
