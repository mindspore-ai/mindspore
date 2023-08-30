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
struct BinaryFunc<BinaryOpType::kSub, In0_t, In1_t, Out_t> {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const { return val0 - val1; }
};
REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(BinaryOpType::kSub);
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(BinaryOpType::kSub);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(BinaryOpType::kSub);
