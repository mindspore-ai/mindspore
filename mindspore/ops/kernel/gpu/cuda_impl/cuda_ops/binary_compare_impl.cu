/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda_impl/cuda_ops/binary_compare_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_common.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_pub_impl.cuh"

template <typename T>
struct BinaryFunc<BinaryOpType::kGreater, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs > rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kGreater);

template <typename T>
struct BinaryFunc<BinaryOpType::kLess, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs < rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kLess);

template <typename In0_t, typename In1_t>
struct BinaryFunc<BinaryOpType::kLess, In0_t, In1_t, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(In0_t val0, In1_t val1) const { return val0 < val1; }
};

template <>
struct BinaryFunc<BinaryOpType::kLess, float, half, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(float val0, half val1) const { return val0 < __half2float(val1); }
};
template <>
struct BinaryFunc<BinaryOpType::kLess, half, float, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(half val0, float val1) const { return __half2float(val0) < val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kLess, double, half, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(double val0, half val1) const {
    return val0 < static_cast<double>(__half2float(val1));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kLess, half, double, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(half val0, double val1) const {
    return static_cast<double>(__half2float(val0)) < val1;
  }
};
template <>
struct BinaryFunc<BinaryOpType::kLess, float, double, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(float val0, double val1) const { return static_cast<double>(val0) < val1; }
};
template <>
struct BinaryFunc<BinaryOpType::kLess, double, float, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(double val0, float val1) const { return val0 < static_cast<double>(val1); }
};

#define LESS_HALF_INT_BOOL_SPEC(T)                                        \
  template <>                                                             \
  struct BinaryFunc<BinaryOpType::kLess, half, T, bool> {                 \
    __device__ __host__ __forceinline__ BinaryFunc() {}                   \
    __device__ __forceinline__ bool operator()(half val0, T val1) const { \
      return __half2float(val0) < static_cast<float>(val1);               \
    }                                                                     \
  };                                                                      \
  template <>                                                             \
  struct BinaryFunc<BinaryOpType::kLess, T, half, bool> {                 \
    __device__ __host__ __forceinline__ BinaryFunc() {}                   \
    __device__ __forceinline__ bool operator()(T val0, half val1) const { \
      return static_cast<float>(val0) < __half2float(val1);               \
    }                                                                     \
  };

LESS_HALF_INT_BOOL_SPEC(int8_t)
LESS_HALF_INT_BOOL_SPEC(uint8_t)
LESS_HALF_INT_BOOL_SPEC(int16_t)
LESS_HALF_INT_BOOL_SPEC(int32_t)
LESS_HALF_INT_BOOL_SPEC(int64_t)
LESS_HALF_INT_BOOL_SPEC(bool)

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int8_t, uint8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int8_t, int16_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int8_t, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int8_t, int64_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, uint8_t, int16_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, uint8_t, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, uint8_t, int64_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int16_t, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int16_t, int64_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int32_t, int64_t);

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, float);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, double);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, double);

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, int8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, uint8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, int16_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, int64_t);

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, int8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, uint8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, int16_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, int64_t);

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, int8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, uint8_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, int16_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, int32_t);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, int64_t);

REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int8_t, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, uint8_t, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int16_t, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int32_t, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, int64_t, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, half, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, float, bool);
REGISTER_COMPARE_MIX_TYPE(BinaryOpType::kLess, double, bool);
