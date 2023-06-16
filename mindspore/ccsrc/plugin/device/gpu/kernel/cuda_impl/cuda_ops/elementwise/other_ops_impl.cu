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
#include <unsupported/Eigen/SpecialFunctions>
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/eltwise_ops_func.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elt_unary_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elt_binary_impl.cuh"

template <typename Type>
struct UnaryFunc<ElwiseOpType::kErfinv, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return erfinv(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErfinv, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return erfinvf(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErfinv, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return __float2half(erfinvf(__half2float(val))); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kErfinv);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kErf, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return erf(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErf, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return erff(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErf, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return __float2half(erff(__half2float(val))); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kErf);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kErfc, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return erfc(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErfc, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return erfcf(val); }
};

template <>
struct UnaryFunc<ElwiseOpType::kErfc, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return __float2half(erfcf(__half2float(val))); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kErfc);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kInvert, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return ~val; }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kInvert);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kInvert);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kInvert);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kSign, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    return val < Type(0.0) ? Type(-1.0) : (val > Type(0.0) ? Type(1.0) : Type(0.0));
  }
};

template <typename Type>
struct UnaryFunc<ElwiseOpType::kSign, Complex<Type>, Complex<Type>> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Complex<Type> operator()(const Complex<Type> val) const {
    Type sum = cuda::elwise::Sqrt<Type>(val.real() * val.real() + val.imag() * val.imag());
    if (sum != static_cast<Type>(0.0)) {
      return Complex<Type>(val.real() / sum, val.imag() / sum);
    } else {
      return Complex<Type>(0.0);
    }
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kSign);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSign);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSign);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kNeg, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return Type(-1) * val; }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kNeg);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kNeg);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kNeg);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kNeg);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kNeg);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kReciprocal, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const {
    if (val != Type(0)) {
      return Type(1.0) / val;
    }
    return std::numeric_limits<Type>::max() + Type(1);
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kReciprocal);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kReciprocal);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kReciprocal);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kReciprocal);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kReciprocal);

template <typename Type>
struct UnaryFunc<ElwiseOpType::kExpm1, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return expm1(val); }
};
template <typename Type>
struct UnaryFunc<ElwiseOpType::kExpm1, Complex<Type>, Complex<Type>> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Complex<Type> operator()(const Complex<Type> val) const { return exp(val) - Complex<Type>(1.0); }
};
template <>
struct UnaryFunc<ElwiseOpType::kExpm1, float, float> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE float operator()(const float val) const { return expm1f(val); }
};
template <>
struct UnaryFunc<ElwiseOpType::kExpm1, half, half> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE half operator()(const half val) const { return __float2half(expm1f(__half2float(val))); }
};
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kExpm1);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kExpm1);
template <typename Type>
struct UnaryFunc<ElwiseOpType::kOnesLike, Type, Type> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Type operator()(const Type val) const { return Type(1.0); }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kOnesLike);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kOnesLike);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kOnesLike);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kOnesLike);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kOnesLike);
template <>
struct UnaryFunc<ElwiseOpType::kLogicalNot, bool, bool> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE bool operator()(const bool val) const { return !val; }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kLogicalNot);

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kZeta, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t val0, const In1_t val1) const {
    return Eigen::internal::scalar_zeta_op<Out_t>()(val0, val1);
  }
};
template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<ElwiseOpType::kZeta, float, float, float>(
  const size_t num, const float *in0, const float *in1, float *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<ElwiseOpType::kZeta, double, double, double>(
  const size_t num, const double *in0, const double *in1, double *out, cudaStream_t cuda_stream);
