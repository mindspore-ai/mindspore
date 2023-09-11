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
struct BinaryFunc<BinaryOpType::kEqual, T, T, bool> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const { return lhs == rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(BinaryOpType::kEqual);

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
