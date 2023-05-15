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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_pub_impl.cuh"

template <typename IN0, typename IN1, typename OUT>
struct BinaryFunc<BinaryOpType::kAdd, IN0, IN1, OUT> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ OUT operator()(IN0 val0, IN1 val1) const { return val0 + val1; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kAdd);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kAdd);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kAdd);

template <typename IN0, typename IN1, typename OUT>
struct BinaryFunc<BinaryOpType::kSub, IN0, IN1, OUT> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ OUT operator()(IN0 val0, IN1 val1) const { return val0 - val1; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kSub);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kSub);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kSub);

template <typename IN0, typename IN1, typename OUT>
struct BinaryFunc<BinaryOpType::kPow, IN0, IN1, OUT, typename std::is_floating_point<OUT>::type> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ OUT operator()(const IN0 &lhs, const IN1 &rhs) const {
    return static_cast<OUT>(pow(lhs, rhs));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kPow, half, half, half> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) const {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

#define POW_INTEGER_IMPL(T)                                                         \
  template <>                                                                       \
  struct BinaryFunc<BinaryOpType::kPow, T, T, T> {                                  \
    __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {  \
      T ret = 1;                                                                    \
      T base = lhs;                                                                 \
      T exp = rhs;                                                                  \
      while (exp) {                                                                 \
        if (exp & 1) {                                                              \
          ret *= base;                                                              \
        }                                                                           \
        base *= base;                                                               \
        exp /= 2;                                                                   \
      }                                                                             \
      return ret;                                                                   \
    }                                                                               \
  };

POW_INTEGER_IMPL(uint8_t)
POW_INTEGER_IMPL(uint16_t)
POW_INTEGER_IMPL(uint32_t)
POW_INTEGER_IMPL(uint64_t)
POW_INTEGER_IMPL(int8_t)
POW_INTEGER_IMPL(int16_t)
POW_INTEGER_IMPL(int32_t)
POW_INTEGER_IMPL(int64_t)


template <typename IN0, typename IN1, typename OUT>
struct BinaryFunc<BinaryOpType::kPow, IN0, IN1, Complex<OUT>> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ Complex<OUT> operator()(const IN0 &lhs, const IN1 &rhs) const {
    Complex<OUT> result;
#if defined(__CUDACC__)
    auto thrust_res = thrust::pow(thrust::complex<OUT>(lhs), thrust::complex<OUT>(rhs));
    result.real(thrust_res.real());
    result.imag(thrust_res.imag());
#else
    std::complex<OUT> lhs_complex(lhs);
    std::complex<OUT> rhs_complex(rhs);
    std::complex<OUT> host_res = std::pow(lhs_complex, rhs_complex);
    result.real(host_res.real());
    result.imag(host_res.imag());
#endif
    return result;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kPow);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kPow);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kPow);
