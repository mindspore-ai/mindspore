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

#define REGISTER_BINARY_OP_CUDA_FUNC_COMPARE_TYPE(op)                                                          \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, double, double, bool>(                \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, double *input0, double *input1, bool *output, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, float, float, bool>(                  \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, float *input0, float *input1, bool *output, size_t device_id,       \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, half, half, bool>(                    \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, half *input0, half *input1, bool *output, size_t device_id,         \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, bool, bool, bool>(                    \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, bool *input0, bool *input1, bool *output, size_t device_id,         \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int8_t, int8_t, bool>(                \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, int8_t *input0, int8_t *input1, bool *output, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint8_t, uint8_t, bool>(              \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, uint8_t *input0, uint8_t *input1, bool *output, size_t device_id,   \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int16_t, int16_t, bool>(              \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, int16_t *input0, int16_t *input1, bool *output, size_t device_id,   \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint16_t, uint16_t, bool>(            \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, uint16_t *input0, uint16_t *input1, bool *output, size_t device_id, \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int32_t, int32_t, bool>(              \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, int32_t *input0, int32_t *input1, bool *output, size_t device_id,   \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint32_t, uint32_t, bool>(            \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, uint32_t *input0, uint32_t *input1, bool *output, size_t device_id, \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int64_t, int64_t, bool>(              \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, int64_t *input0, int64_t *input1, bool *output, size_t device_id,   \
    cudaStream_t cuda_stream);                                                                                 \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint64_t, uint64_t, bool>(            \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,     \
    const std::vector<int64_t> &out_shape, uint64_t *input0, uint64_t *input1, bool *output, size_t device_id, \
    cudaStream_t cuda_stream)

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

template <typename T>
struct BinaryFunc<BinaryOpType::kMaximum, T, T, T> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return lhs > rhs ? lhs : rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMaximum);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMaximum);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kMaximum);

template <typename T>
struct BinaryFunc<BinaryOpType::kMinimum, T, T, T> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) const { return lhs < rhs ? lhs : rhs; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kMinimum);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kMinimum);
REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(BinaryOpType::kMinimum);
