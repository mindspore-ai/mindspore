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
struct UnaryFunc<ElwiseOpType::kSin, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Sin<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSin);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSin);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kCos, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Cos<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kCos);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kCos);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kTan, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Tan<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kTan);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kTan);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kSinh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Sinh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSinh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSinh);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kCosh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Cosh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kCosh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kCosh);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kTanh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Tanh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kTanh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kTanh);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kAsin, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Asin<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAsin);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAsin);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kAcos, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Acos<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAcos);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAcos);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kAtan, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Atan<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAtan);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAtan);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kAsinh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}

  DEVICE Type operator()(const Type val) const { return cuda::elwise::Asinh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAsinh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAsinh);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kAcosh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Acosh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAcosh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAcosh);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kAtanh, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return cuda::elwise::Atanh<Type>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAtanh);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAtanh);
