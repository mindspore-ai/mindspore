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

template <typename Inp_t, typename Out_t>
struct UnaryFunc<ElwiseOpType::kConj, Inp_t, Out_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Out_t operator()(const Inp_t val) const { return cuda::elwise::Conj<Out_t>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kConj);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kConj);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kConj);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kConj);
REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kConj);
template <typename Inp_t, typename Out_t>
struct UnaryFunc<ElwiseOpType::kReal, Inp_t, Out_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Out_t operator()(const Inp_t val) const { return cuda::elwise::Real<Inp_t, Out_t>(val); }
};

// Complex<T> to T
#define REGISTER_UNARY_OP_CUDA_FUNC_MIX_COMPLEX_TYPE(Op)                                \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, Complex<float>, float>(     \
    const size_t num, const Complex<float> *inp, float *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, Complex<double>, double>(   \
    const size_t num, const Complex<double> *inp, double *out, cudaStream_t cuda_stream)

REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kReal);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kReal);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kReal);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kReal);
REGISTER_UNARY_OP_CUDA_FUNC_MIX_COMPLEX_TYPE(ElwiseOpType::kReal);
template <typename Inp_t, typename Out_t>
struct UnaryFunc<ElwiseOpType::kImag, Inp_t, Out_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Out_t operator()(const Inp_t val) const { return cuda::elwise::Imag<Inp_t, Out_t>(val); }
};
REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(ElwiseOpType::kImag);
REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(ElwiseOpType::kImag);
REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(ElwiseOpType::kImag);
REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kImag);
REGISTER_UNARY_OP_CUDA_FUNC_MIX_COMPLEX_TYPE(ElwiseOpType::kImag);
template <typename Inp_t, typename Out_t>
struct UnaryFunc<ElwiseOpType::kComplexAbs, Inp_t, Out_t> {
  DEVICE_HOST UnaryFunc() {}
  DEVICE Out_t operator()(const Inp_t val) const {
    return cuda::elwise::Sqrt<Out_t>(val.real() * val.real() + val.imag() * val.imag());
  }
};
REGISTER_UNARY_OP_CUDA_FUNC_MIX_COMPLEX_TYPE(ElwiseOpType::kComplexAbs);
