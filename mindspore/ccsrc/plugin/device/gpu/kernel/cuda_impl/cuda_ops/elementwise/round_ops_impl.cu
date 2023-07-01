/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHType WARRANTIES OR CONDITIONS OF ANY KTypeD, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/eltwise_ops_func.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elt_unary_impl.cuh"

template <typename Type>
struct UnaryFunc<ElwiseOpType::kFloor, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return floor(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kFloor, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return floorf(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kFloor, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return hfloor(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kFloor);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kCeil, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return ceil(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kCeil, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return ceilf(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kCeil, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return hceil(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kCeil);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kTrunc, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return trunc(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kTrunc, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return truncf(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kTrunc, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return htrunc(val); }
};

#define TRUNC_OP_INT(T)                                       \
  template <>                                                 \
  struct UnaryFunc<ElwiseOpType::kTrunc, T, T> {              \
    DEVICE_HOST UnaryFunc() {}                                \
    DEVICE T operator()(const T val) const { return val; } \
  }
TRUNC_OP_INT(int8_t);
TRUNC_OP_INT(int16_t);
TRUNC_OP_INT(int32_t);
TRUNC_OP_INT(int64_t);
TRUNC_OP_INT(uint8_t);
TRUNC_OP_INT(uint16_t);
TRUNC_OP_INT(uint32_t);
TRUNC_OP_INT(uint64_t);
TRUNC_OP_INT(bool);

REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kTrunc);
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kTrunc);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kTrunc);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kTrunc);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kRound, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return nearbyintf(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kRound, double, double> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE double operator()(const double val) const { return nearbyint(val); }
};

REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kRound);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kRound);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kRint, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return rint(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kRint, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return rintf(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kRint, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return hrint(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kRint);
