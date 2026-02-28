/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_common.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_pub_impl.cuh"

template <typename T>
struct BinaryFunc<BinaryOpType::kEqual, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs == rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kEqual);

// Mixed dtype for Equal: compare after casting to a promoted compute type (aligned with mul mixed rules).
#define DEFINE_EQUAL_MIX_FUNC(IN0_T, IN1_T, CAST_T)                                              \
  template <>                                                                                    \
  struct BinaryFunc<BinaryOpType::kEqual, IN0_T, IN1_T, bool> {                                  \
    __device__ __host__ __forceinline__ BinaryFunc() {}                                          \
    __device__ __forceinline__ bool operator()(const IN0_T &lhs, const IN1_T &rhs) const {       \
      return static_cast<CAST_T>(lhs) == static_cast<CAST_T>(rhs);                               \
    }                                                                                            \
  };

// int mixes
DEFINE_EQUAL_MIX_FUNC(int8_t, uint8_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(uint8_t, int8_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(int8_t, int16_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(int16_t, int8_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(int8_t, int32_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int32_t, int8_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int8_t, int64_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int64_t, int8_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(uint8_t, int16_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(int16_t, uint8_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(uint8_t, int32_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int32_t, uint8_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(uint8_t, int64_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int64_t, uint8_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int16_t, int32_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int32_t, int16_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int16_t, int64_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int64_t, int16_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int32_t, int64_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(int64_t, int32_t, int64_t)

// float mixes
DEFINE_EQUAL_MIX_FUNC(half, float, float)
DEFINE_EQUAL_MIX_FUNC(float, half, float)
DEFINE_EQUAL_MIX_FUNC(half, double, double)
DEFINE_EQUAL_MIX_FUNC(double, half, double)
DEFINE_EQUAL_MIX_FUNC(float, double, double)
DEFINE_EQUAL_MIX_FUNC(double, float, double)

// float-int mixes (half keeps half)
DEFINE_EQUAL_MIX_FUNC(half, int8_t, half)
DEFINE_EQUAL_MIX_FUNC(int8_t, half, half)
DEFINE_EQUAL_MIX_FUNC(half, uint8_t, half)
DEFINE_EQUAL_MIX_FUNC(uint8_t, half, half)
DEFINE_EQUAL_MIX_FUNC(half, int16_t, half)
DEFINE_EQUAL_MIX_FUNC(int16_t, half, half)
DEFINE_EQUAL_MIX_FUNC(half, int32_t, half)
DEFINE_EQUAL_MIX_FUNC(int32_t, half, half)
// NOTE: `static_cast<half>(int64_t)` is ambiguous for CUDA `__half` on some platforms/toolchains
// (multiple converting constructors match). Route through float to make the conversion explicit.
template <>
struct BinaryFunc<BinaryOpType::kEqual, half, int64_t, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const half &lhs, const int64_t &rhs) const {
    return static_cast<half>(lhs) == static_cast<half>(static_cast<float>(rhs));
  }
};
template <>
struct BinaryFunc<BinaryOpType::kEqual, int64_t, half, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const int64_t &lhs, const half &rhs) const {
    return static_cast<half>(static_cast<float>(lhs)) == static_cast<half>(rhs);
  }
};

DEFINE_EQUAL_MIX_FUNC(float, int8_t, float)
DEFINE_EQUAL_MIX_FUNC(int8_t, float, float)
DEFINE_EQUAL_MIX_FUNC(float, uint8_t, float)
DEFINE_EQUAL_MIX_FUNC(uint8_t, float, float)
DEFINE_EQUAL_MIX_FUNC(float, int16_t, float)
DEFINE_EQUAL_MIX_FUNC(int16_t, float, float)
DEFINE_EQUAL_MIX_FUNC(float, int32_t, float)
DEFINE_EQUAL_MIX_FUNC(int32_t, float, float)
DEFINE_EQUAL_MIX_FUNC(float, int64_t, float)
DEFINE_EQUAL_MIX_FUNC(int64_t, float, float)

DEFINE_EQUAL_MIX_FUNC(double, int8_t, double)
DEFINE_EQUAL_MIX_FUNC(int8_t, double, double)
DEFINE_EQUAL_MIX_FUNC(double, uint8_t, double)
DEFINE_EQUAL_MIX_FUNC(uint8_t, double, double)
DEFINE_EQUAL_MIX_FUNC(double, int16_t, double)
DEFINE_EQUAL_MIX_FUNC(int16_t, double, double)
DEFINE_EQUAL_MIX_FUNC(double, int32_t, double)
DEFINE_EQUAL_MIX_FUNC(int32_t, double, double)
DEFINE_EQUAL_MIX_FUNC(double, int64_t, double)
DEFINE_EQUAL_MIX_FUNC(int64_t, double, double)

// mixes with bool
DEFINE_EQUAL_MIX_FUNC(int8_t, bool, int8_t)
DEFINE_EQUAL_MIX_FUNC(bool, int8_t, int8_t)
DEFINE_EQUAL_MIX_FUNC(uint8_t, bool, uint8_t)
DEFINE_EQUAL_MIX_FUNC(bool, uint8_t, uint8_t)
DEFINE_EQUAL_MIX_FUNC(int16_t, bool, int16_t)
DEFINE_EQUAL_MIX_FUNC(bool, int16_t, int16_t)
DEFINE_EQUAL_MIX_FUNC(int32_t, bool, int32_t)
DEFINE_EQUAL_MIX_FUNC(bool, int32_t, int32_t)
DEFINE_EQUAL_MIX_FUNC(int64_t, bool, int64_t)
DEFINE_EQUAL_MIX_FUNC(bool, int64_t, int64_t)
DEFINE_EQUAL_MIX_FUNC(half, bool, half)
DEFINE_EQUAL_MIX_FUNC(bool, half, half)
DEFINE_EQUAL_MIX_FUNC(float, bool, float)
DEFINE_EQUAL_MIX_FUNC(bool, float, float)
DEFINE_EQUAL_MIX_FUNC(double, bool, double)
DEFINE_EQUAL_MIX_FUNC(bool, double, double)

#undef DEFINE_EQUAL_MIX_FUNC

// Explicit instantiations for Equal mixed dtypes (needed for kernel registry linking).
#define REGISTER_EQUAL_MIX_OP(IN0_T, IN1_T)                                                                     \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<BinaryOpType::kEqual, IN0_T, IN1_T, bool>( \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,      \
    const std::vector<int64_t> &out_shape, IN0_T *in0, IN1_T *in1, bool *out, size_t device_id,                 \
    cudaStream_t cuda_stream);

REGISTER_EQUAL_MIX_OP(int8_t, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, int8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, uint8_t)
REGISTER_EQUAL_MIX_OP(int16_t, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, int16_t)
REGISTER_EQUAL_MIX_OP(int32_t, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, int32_t)

REGISTER_EQUAL_MIX_OP(half, float)
REGISTER_EQUAL_MIX_OP(float, half)
REGISTER_EQUAL_MIX_OP(half, double)
REGISTER_EQUAL_MIX_OP(double, half)
REGISTER_EQUAL_MIX_OP(float, double)
REGISTER_EQUAL_MIX_OP(double, float)

REGISTER_EQUAL_MIX_OP(half, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, half)
REGISTER_EQUAL_MIX_OP(half, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, half)
REGISTER_EQUAL_MIX_OP(half, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, half)
REGISTER_EQUAL_MIX_OP(half, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, half)
REGISTER_EQUAL_MIX_OP(half, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, half)

REGISTER_EQUAL_MIX_OP(float, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, float)
REGISTER_EQUAL_MIX_OP(float, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, float)
REGISTER_EQUAL_MIX_OP(float, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, float)
REGISTER_EQUAL_MIX_OP(float, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, float)
REGISTER_EQUAL_MIX_OP(float, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, float)

REGISTER_EQUAL_MIX_OP(double, int8_t)
REGISTER_EQUAL_MIX_OP(int8_t, double)
REGISTER_EQUAL_MIX_OP(double, uint8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, double)
REGISTER_EQUAL_MIX_OP(double, int16_t)
REGISTER_EQUAL_MIX_OP(int16_t, double)
REGISTER_EQUAL_MIX_OP(double, int32_t)
REGISTER_EQUAL_MIX_OP(int32_t, double)
REGISTER_EQUAL_MIX_OP(double, int64_t)
REGISTER_EQUAL_MIX_OP(int64_t, double)

REGISTER_EQUAL_MIX_OP(int8_t, bool)
REGISTER_EQUAL_MIX_OP(bool, int8_t)
REGISTER_EQUAL_MIX_OP(uint8_t, bool)
REGISTER_EQUAL_MIX_OP(bool, uint8_t)
REGISTER_EQUAL_MIX_OP(int16_t, bool)
REGISTER_EQUAL_MIX_OP(bool, int16_t)
REGISTER_EQUAL_MIX_OP(int32_t, bool)
REGISTER_EQUAL_MIX_OP(bool, int32_t)
REGISTER_EQUAL_MIX_OP(int64_t, bool)
REGISTER_EQUAL_MIX_OP(bool, int64_t)
REGISTER_EQUAL_MIX_OP(half, bool)
REGISTER_EQUAL_MIX_OP(bool, half)
REGISTER_EQUAL_MIX_OP(float, bool)
REGISTER_EQUAL_MIX_OP(bool, float)
REGISTER_EQUAL_MIX_OP(double, bool)
REGISTER_EQUAL_MIX_OP(bool, double)

#undef REGISTER_EQUAL_MIX_OP

template <typename T>
struct BinaryFunc<BinaryOpType::kGreaterEqual, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs >= rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kGreaterEqual);

template <typename T>
struct BinaryFunc<BinaryOpType::kLessEqual, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs <= rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kLessEqual);

template <typename T>
struct BinaryFunc<BinaryOpType::kNotEqual, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs != rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kNotEqual);
