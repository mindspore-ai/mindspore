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
struct UnaryFunc<ElwiseOpType::kSigmoid, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return Type(1.0) / (Type(1.0) + cuda::elwise::Exp<Type>(-val)); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSigmoid);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSigmoid);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kMish, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return val * cuda::elwise::Tanh<Type>(cuda::elwise::Log1p<Type>(cuda::elwise::Exp<Type>(val)));
  }
};

template <>
struct UnaryFunc<ElwiseOpType::kMish, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return val * __float2half(tanhf(log1pf(expf(__half2float(val))))); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kMish);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kSoftsign, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return val / (Type(1.0) + cuda::elwise::Abs<Type>(val)); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSoftsign);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kSiLU, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return val / (Type(1.0) + cuda::elwise::Exp<Type>(-val)); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSiLU);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSiLU);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kReLU, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    Type zero = Type(0.0);
    return cuda::elwise::IsNan<Type>(val) || (val > zero) ? val : zero;
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kReLU);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kReLU);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kReLU);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kReLU);
