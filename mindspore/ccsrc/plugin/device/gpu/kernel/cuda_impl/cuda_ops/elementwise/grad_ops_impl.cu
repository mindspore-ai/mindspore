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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elt_binary_impl.cuh"

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kAsinGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout / cuda::elwise::Conj<Out_t>(cuda::elwise::Sqrt<Out_t>(In0_t(1.0) - input * input));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAsinGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAsinGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kACosGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return -dout / cuda::elwise::Conj<Out_t>(cuda::elwise::Sqrt<Out_t>(In0_t(1.0) - input * input));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kACosGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kACosGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kAtanGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout / cuda::elwise::Conj<Out_t>(Out_t(1.0) + input * input);
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAtanGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAtanGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kAsinhGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout / cuda::elwise::Conj<Out_t>(cuda::elwise::Cosh<Out_t>(input));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAsinhGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAsinhGrad);

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kAcoshGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    float inputf = static_cast<float>(input);
    Out_t sinhy = static_cast<Out_t>(sinhf(inputf));
    return dout / sinhy;
  }
};

template <>
struct BinaryFunc<ElwiseOpType::kAcoshGrad, double, double, double> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE double operator()(const double input, const double dout) const {
    double inputf = static_cast<double>(input);
    double sinhy = static_cast<double>(sinhf(inputf));
    return dout / sinhy;
  }
};

template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kAcoshGrad, Complex<In0_t>, Complex<In1_t>, Complex<Out_t>> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Complex<Out_t> operator()(const Complex<In0_t> input, const Complex<In1_t> dout) const {
    Complex<Out_t> sinhy = sinh(input);
    sinhy = Complex<Out_t>(sinhy.real(), -sinhy.imag());
    return dout / sinhy;
  }
};

REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kAcoshGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kAcoshGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kTanhGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout * (Out_t(1.0) - input * input);
  }
};

REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kTanhGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kTanhGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kSqrtGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout / (Out_t(2.0) * cuda::elwise::Conj<Out_t>(input));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSqrtGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSqrtGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kRsqrtGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return cuda::elwise::Conj<Out_t>(Out_t(-0.5) * input * input * input * dout);
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kRsqrtGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kRsqrtGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kReciprocalGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return Out_t(-1.0) * input * input * dout;
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kReciprocalGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kReciprocalGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kSigmoidGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t input, const In1_t dout) const {
    return dout * cuda::elwise::Conj<Out_t>(input * (Out_t(1.0) - input));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSigmoidGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSigmoidGrad);
template <typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc<ElwiseOpType::kSiLUGrad, In0_t, In1_t, Out_t> {
  DEVICE_HOST BinaryFunc() {}
  DEVICE Out_t operator()(const In0_t dout, const In1_t input) const {
    Out_t one{1};
    Out_t mid = one / (one + cuda::elwise::Exp<In0_t>(-input));
    return dout * mid * (one + input * (one - mid));
  }
};
REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(ElwiseOpType::kSiLUGrad);
REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(ElwiseOpType::kSiLUGrad);
