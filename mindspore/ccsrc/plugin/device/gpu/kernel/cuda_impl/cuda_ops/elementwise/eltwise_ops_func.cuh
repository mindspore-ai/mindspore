/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_FUNC_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_FUNC_CUH_
#include <cuda_fp16.h>
#include <cmath>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
namespace cuda {
namespace elwise {
#if defined(__CUDACC__)
#define DEVICE __device__ __forceinline__
#else
#define DEVICE
#endif
template <typename T>
DEVICE T Sin(const T val) {
  return sin(val);
}
template <>
DEVICE float Sin(const float val) {
  return sinf(val);
}
template <>
DEVICE half Sin(const half val) {
  return hsin(val);
}
template <typename T>
DEVICE T Cos(const T val) {
  return cos(val);
}
template <>
DEVICE float Cos(const float val) {
  return cosf(val);
}
template <>
DEVICE half Cos(const half val) {
  return hcos(val);
}
template <typename T>
DEVICE T Tan(const T val) {
  return tan(val);
}
template <>
DEVICE float Tan(const float val) {
  return tanf(val);
}
template <>
DEVICE half Tan(const half val) {
  return __float2half(__tanf(__half2float(val)));
}
template <typename T>
DEVICE T Sinh(const T val) {
  return sinh(val);
}
template <>
DEVICE float Sinh(const float val) {
  return sinhf(val);
}
template <>
DEVICE half Sinh(const half val) {
  return half(0.5) * (hexp(val) - hexp(-val));
}
template <typename T>
DEVICE T Cosh(const T val) {
  return cosh(val);
}
template <>
DEVICE float Cosh(const float val) {
  return coshf(val);
}
template <>
DEVICE half Cosh(const half val) {
  return half(0.5) * (hexp(val) + hexp(-val));
}
template <typename T>
DEVICE T Tanh(const T val) {
  return tanh(val);
}
template <>
DEVICE float Tanh(const float val) {
  return tanhf(val);
}
template <>
DEVICE half Tanh(const half val) {
  return __float2half(tanhf(__half2float(val)));
}
template <typename T>
DEVICE T Asin(const T val) {
  return asin(val);
}
template <>
DEVICE float Asin(const float val) {
  return asinf(val);
}
template <>
DEVICE half Asin(const half val) {
  return __float2half(asinf(__half2float(val)));
}
template <typename T>
DEVICE T Acos(const T val) {
  return acos(val);
}
template <>
DEVICE float Acos(const float val) {
  return acosf(val);
}
template <>
DEVICE half Acos(const half val) {
  return __float2half(acosf(__half2float(val)));
}
template <typename T>
DEVICE T Atan(const T val) {
  return atan(val);
}
template <>
DEVICE float Atan(const float val) {
  return atanf(val);
}
template <>
DEVICE half Atan(const half val) {
  return __float2half(atanf(__half2float(val)));
}
template <typename T>
DEVICE T Asinh(const T val) {
  return asinh(val);
}
template <>
DEVICE float Asinh(const float val) {
  return asinhf(val);
}
template <>
DEVICE half Asinh(const half val) {
  return __float2half(asinhf(__half2float(val)));
}
template <typename T>
DEVICE T Acosh(const T val) {
  return acosh(val);
}
template <>
DEVICE float Acosh(const float val) {
  return acoshf(val);
}
template <>
DEVICE half Acosh(const half val) {
  return __float2half(acoshf(__half2float(val)));
}
template <typename T>
DEVICE T Atanh(const T val) {
  return atanh(val);
}
template <>
DEVICE float Atanh(const float val) {
  return atanhf(val);
}
template <>
DEVICE half Atanh(const half val) {
  return __float2half(atanhf(__half2float(val)));
}
template <typename T>
DEVICE T Abs(const T val) {
  return abs(val);
}
template <>
DEVICE float Abs(const float val) {
  return fabsf(val);
}
template <>
DEVICE half Abs(const half val) {
  return val < half(0.0) ? -val : val;
}
template <>
DEVICE bool Abs(const bool val) {
  return val;
}

template <typename T>
DEVICE T Exp(const T val) {
  return static_cast<T>(expf(static_cast<float>(val)));
}
template <>
DEVICE float Exp(const float val) {
  return expf(val);
}
template <>
DEVICE half Exp(const half val) {
  return hexp(val);
}
template <>
DEVICE double Exp(const double val) {
  return exp(val);
}
template <>
DEVICE Complex<float> Exp(const Complex<float> val) {
  return exp(val);
}
template <>
DEVICE Complex<double> Exp(const Complex<double> val) {
  return exp(val);
}
template <typename T>
DEVICE T Log1p(const T val) {
  return log1p(val);
}
template <>
DEVICE float Log1p(const float val) {
  return log1pf(val);
}
template <>
DEVICE half Log1p(const half val) {
  return __float2half(log1pf(__half2float(val)));
}
template <typename T>
DEVICE Complex<float> Log1p(const Complex<float> val) {
  return log(val + Complex<float>(1, 0));
}
template <>
DEVICE Complex<double> Log1p(const Complex<double> val) {
  return log(val + Complex<double>(1, 0));
}
template <typename T>
DEVICE T IsNan(const T val) {
  return false;
}
template <>
DEVICE float IsNan(const float val) {
  return isnan(val);
}
template <>
DEVICE double IsNan(const double val) {
  return isnan(val);
}
template <>
DEVICE half IsNan(const half val) {
  return __hisnan(val);
}
template <typename T>
DEVICE
  typename std::enable_if<!(std::is_same<T, Complex<double>>::value || std::is_same<T, Complex<float>>::value), T>::type
  Conj(const T val) {
  return val;
}
template <typename T>
DEVICE
  typename std::enable_if<std::is_same<T, Complex<double>>::value || std::is_same<T, Complex<float>>::value, T>::type
  Conj(const T val) {
  return conj(val);
}
template <typename Inp_t, typename Out_t>
DEVICE typename std::enable_if<!(std::is_same<Inp_t, Out_t>::value), Out_t>::type Real(Inp_t val) {
  return val.real();
}
template <typename Inp_t, typename Out_t>
DEVICE typename std::enable_if<!(std::is_same<Inp_t, Out_t>::value), Out_t>::type Imag(Inp_t val) {
  return val.imag();
}
template <typename Inp_t, typename Out_t>
DEVICE typename std::enable_if<std::is_same<Inp_t, Out_t>::value, Out_t>::type Real(Inp_t val) {
  return val;
}
template <typename Inp_t, typename Out_t>
DEVICE typename std::enable_if<std::is_same<Inp_t, Out_t>::value, Out_t>::type Imag(Inp_t val) {
  return Out_t(0.0);
}

template <typename T>
DEVICE T Sqrt(const T val) {
  return static_cast<T>(sqrtf(static_cast<float>(val)));
}
template <>
DEVICE float Sqrt(const float val) {
  return sqrtf(val);
}
template <>
DEVICE half Sqrt(const half val) {
  return hsqrt(val);
}
template <>
DEVICE double Sqrt(const double val) {
  return sqrt(val);
}
template <>
DEVICE Complex<float> Sqrt(const Complex<float> val) {
  return sqrt(val);
}
template <>
DEVICE Complex<double> Sqrt(const Complex<double> val) {
  return sqrt(val);
}
template <typename T>
DEVICE T Rsqrt(const T val) {
  return T(1.0) / Sqrt<T>(val);
}
template <>
DEVICE float Rsqrt(const float val) {
  return rsqrtf(val);
}
template <>
DEVICE double Rsqrt(const double val) {
  return rsqrt(val);
}
template <>
DEVICE half Rsqrt(const half val) {
  return hrsqrt(val);
}
template <typename T>
DEVICE T Log(const T val) {
  return log(val);
}
template <>
DEVICE float Log(const float val) {
  return logf(val);
}
template <>
DEVICE half Log(const half val) {
  return hlog(val);
}
}  // namespace elwise
}  // namespace cuda
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_FUNC_CUH_
