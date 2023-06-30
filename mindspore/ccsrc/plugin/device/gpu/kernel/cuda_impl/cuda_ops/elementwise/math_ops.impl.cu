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
struct UnaryFunc<ElwiseOpType::kAbs, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Abs<Type>(val);
  }
};
template <>
struct UnaryFunc<ElwiseOpType::kAbs, uint8_t, uint8_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE uint8_t operator()(const uint8_t val) const {
    return val;
  }
};
template <>
struct UnaryFunc<ElwiseOpType::kAbs, uint16_t, uint16_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE uint16_t operator()(const uint16_t val) const {
    return val;
  }
};
template <>
struct UnaryFunc<ElwiseOpType::kAbs, uint32_t, uint32_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE uint32_t operator()(const uint32_t val) const {
    return val;
  }
};
template <>
struct UnaryFunc<ElwiseOpType::kAbs, uint64_t, uint64_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE uint64_t operator()(const uint64_t val) const {
    return val;
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kAbs);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kAbs);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kAbs);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAbs);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAbs);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kSquare, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return val * val;
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kSquare);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kSquare);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kSquare);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSquare);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSquare);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kSqrt, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Sqrt<Type>(val);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kSqrt);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kSqrt);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kSqrt);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSqrt);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSqrt);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kRsqrt, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Rsqrt<Type>(val);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kRsqrt);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kRsqrt);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kExp, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Exp<Type>(val);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kExp);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kExp);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kExp);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kExp);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kExp);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kLog, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Log<Type>(val);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kLog);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kLog);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kLog1p, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return cuda::elwise::Log1p<Type>(val);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kLog1p);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kLog1p);
