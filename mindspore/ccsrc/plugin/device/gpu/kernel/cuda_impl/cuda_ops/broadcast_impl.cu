/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include <math.h>
#include <vector>
#include "include/cuda_fp16.h"

constexpr float kFloatEplison = 1e-37;

// Basic function
template <typename T>
struct GreaterFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs > rhs; }
};

template <typename T>
struct LessFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs < rhs; }
};

template <typename T>
struct EqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs == rhs; }
};

template <>
struct EqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return abs(__half2float(lhs) - __half2float(rhs)) < 1e-9;
  }
};

template <>
struct EqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return abs(lhs - rhs) < 1e-9;
  }
};

template <typename T>
struct GreaterEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs >= rhs; }
};

template <>
struct GreaterEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return abs(__half2float(lhs) - __half2float(rhs)) < 1e-9 ? true : (__half2float(lhs) > __half2float(rhs));
  }
};

template <>
struct GreaterEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return abs(lhs - rhs) < 1e-9 ? true : (lhs > rhs);
  }
};

template <typename T>
struct LessEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs <= rhs; }
};

template <>
struct LessEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return abs(__half2float(lhs) - __half2float(rhs)) < 1e-9 ? true : (__half2float(lhs) < __half2float(rhs));
  }
};

template <>
struct LessEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) { return lhs <= rhs; }
};

template <typename T>
struct NotEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs != rhs; }
};

template <>
struct NotEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return abs(__half2float(lhs) - __half2float(rhs)) >= 1e-9;
  }
};

template <>
struct NotEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return abs(lhs - rhs) >= 1e-9;
  }
};

template <typename T>
struct LogicalAndFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs && rhs; }
};

template <typename T>
struct LogicalOrFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs || rhs; }
};

template <typename T>
struct MinimumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs < rhs ? lhs : rhs; }
};

template <typename T>
struct MaximumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs > rhs ? lhs : rhs; }
};

#ifndef _WIN32
template <typename T>
struct PowerFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return static_cast<T>(pow(static_cast<double>(lhs), static_cast<double>(rhs)));
  }
};

#else
template <typename T>
struct PowerFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return static_cast<T>(pow(static_cast<double>(lhs), static_cast<double>(rhs)));
  }
};

template <>
struct PowerFunc<float> {
  __device__ __host__ __forceinline__ float operator()(const float &lhs, const float &rhs) { return pow(lhs, rhs); }
};

template <>
struct PowerFunc<double> {
  __device__ __host__ __forceinline__ double operator()(const double &lhs, const double &rhs) { return pow(lhs, rhs); }
};
#endif

template <>
struct PowerFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct PowerFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

#define POW_INTEGER_IMPL(T)                                                         \
  template <>                                                                       \
  struct PowerFunc<T> {                                                             \
    __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {  \
      T ret = 1;                                                                    \
      T base = lhs;                                                                 \
      T exp = rhs;                                                                  \
      while (exp) {                                                                 \
        if (exp & 1) {                                                              \
          ret *= base;                                                              \
        }                                                                           \
        base *= base;                                                               \
        exp /= 2;                                                                   \
      }                                                                             \
      return ret;                                                                   \
    }                                                                               \
  };

POW_INTEGER_IMPL(uint8_t)
POW_INTEGER_IMPL(uint16_t)
POW_INTEGER_IMPL(uint32_t)
POW_INTEGER_IMPL(uint64_t)
POW_INTEGER_IMPL(int8_t)
POW_INTEGER_IMPL(int16_t)
POW_INTEGER_IMPL(int32_t)
POW_INTEGER_IMPL(int64_t)

template <typename T>
__device__ __host__ T abs_complex(const Complex<T> &x) {
  double res = 0.0;
  res = hypot(static_cast<double>(x.real()), static_cast<double>(x.imag()));
  return static_cast<T>(res);
}

template <typename T>
__device__ __host__ T arg_complex(const Complex<T> &x) {
  return atan2<T>(x.imag(), x.real());
}

template <typename T>
__device__ __host__ Complex<T> log_complex(const Complex<T> &x) {
  return Complex<T>(log(abs_complex<T>(x)), arg_complex<T>(x));
}

template <typename T>
__device__ __host__ Complex<T> exp_complex(const Complex<T> &x) {
  T imag_value = x.imag();
  if (isinf(x.real())) {
    if (x.real() < T(0)) {
      if (!isfinite(imag_value)) imag_value = T(1);
    } else if (imag_value == 0 || !isfinite(imag_value)) {
      if (isinf(imag_value)) {
        imag_value = T(NAN);
      }
      return Complex<T>(x.real(), imag_value);
    }
  } else if (isnan(x.real()) && x.imag() == 0) {
    return x;
  }
  T real_exp = exp(x.real());
  return Complex<T>(real_exp * cos(imag_value), real_exp * sin(imag_value));
}

template <>
struct PowerFunc<Complex<float>> {
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs, 0.0);
    res = exp_complex<float>(y * log_complex<float>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs, 0.0);
    Complex<float> y(rhs.real(), rhs.imag());
    res = exp_complex<float>(y * log_complex<float>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs, 0.0);
    Complex<float> y(rhs, 0.0);
    res = exp_complex<float>(y * log_complex<float>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs.real(), rhs.imag());
    res = exp_complex<float>(y * log_complex<float>(x));
    return res;
  }
};

template <>
struct PowerFunc<Complex<double>> {
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs, 0.0);
    res = exp_complex<double>(y * log_complex<double>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs, 0.0);
    Complex<double> y(rhs.real(), rhs.imag());
    res = exp_complex<double>(y * log_complex<double>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs, 0.0);
    Complex<double> y(rhs, 0.0);
    res = exp_complex<double>(y * log_complex<double>(x));
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs,
                                                                 const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs.real(), rhs.imag());
    res = exp_complex<double>(y * log_complex<double>(x));
    return res;
  }
};

template <typename T>
struct RealDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs / rhs);
  }
};

template <typename T>
struct BitwiseAnd {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs & rhs); }
};

template <>
struct BitwiseAnd<double> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) { return 0; }
};
template <>
struct BitwiseAnd<float> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseAnd<half> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseAnd<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

template <typename T>
struct BitwiseOr {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs | rhs); }
};

template <>
struct BitwiseOr<double> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) { return 0; }
};
template <>
struct BitwiseOr<float> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseOr<half> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseOr<half2> {
  // __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) { __float22half2_rn(0); }
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

template <typename T>
struct BitwiseXor {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs ^ rhs); }
};

template <>
struct BitwiseXor<double> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) { return 0; }
};
template <>
struct BitwiseXor<float> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseXor<half> {
  __device__ __host__ __forceinline__ int16_t operator()(const int16_t &lhs, const int16_t &rhs) { return 0; }
};
template <>
struct BitwiseXor<half2> {
  // __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) { __float22half2_rn(0); }
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

template <typename T>
struct ComplexFunc {
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const T &rhs) { return Complex<T>(lhs, rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return lhs; }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return rhs; }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
     return lhs;
  }
};

template <typename T>
struct DivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs / rhs);
  }
};

template <typename T>
struct MulFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs * rhs);
  }
};

template <typename T>
struct SubFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs - rhs);
  }
};

template <typename T>
struct AddFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs + rhs);
  }
};
// DivNoNan check if rhs is less than epsilon
template <typename T>
struct DivNoNanFunc {
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return rhs < kFloatEplison && rhs > -kFloatEplison ? 0.0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<int> {
  __device__ __host__ __forceinline__ int operator()(const int &lhs, const int &rhs) {
    return rhs == 0 ? 0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    if (__half2float(rhs) < (0.00007) && __half2float(rhs) > -0.00007) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct DivNoNanFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    if ((r.x < kFloatEplison && r.x > -kFloatEplison) || (r.y < kFloatEplison && r.y > -kFloatEplison)) {
      l.x = 0.0;
      l.y = 0.0;
    } else {
      l.x = l.x / r.x;
      l.y = l.y / r.y;
    }
    return __float22half2_rn(l);
  }
};

// XDivy check if lhs is less than epsilon, XDivy support half, float, double
template <typename T>
struct XDivyFunc {
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs < kFloatEplison && lhs > -kFloatEplison ? 0.0 : (lhs / rhs);
  }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    Complex<T> res(0.0, 0.0);
    Complex<T> x(lhs.real(), lhs.imag());
    Complex<T> y(rhs.real(), rhs.imag());
    res = x/y;
    return res;
  }
};

template <>
struct XDivyFunc<Complex<float>> {
__device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs, rhs);
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs, lhs);
    Complex<float> y(rhs.real(), rhs.imag());
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs, lhs);
    Complex<float> y(rhs, rhs);
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs.real(), rhs.imag());
    res = x/y;
    return res;
  }
};

template <>
struct XDivyFunc<Complex<double>> {
__device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs, rhs);
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs, lhs);
    Complex<double> y(rhs.real(), rhs.imag());
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs, lhs);
    Complex<double> y(rhs, rhs);
    res = x/y;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs,
                                                                 const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs.real(), rhs.imag());
    res = x/y;
    return res;
  }
};

template <>
struct XDivyFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    if (__half2float(lhs) < (0.00007) && __half2float(lhs) > -0.00007) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct XDivyFunc<int> {
__device__ __host__ __forceinline__ int operator()(const int &lhs, const int &rhs) {
    return lhs == 0 ? 0 : (lhs / rhs);
  }
};

template <>
struct XDivyFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    if ((l.x < kFloatEplison && l.x > -kFloatEplison) && (l.y < kFloatEplison && l.y > -kFloatEplison)) {
      res.x = 0.0;
      res.y = 0.0;
    } else if ((l.x < kFloatEplison && l.x > -kFloatEplison)) {
      res.x = 0.0;
      res.y = l.y / r.y;
    } else if (l.y < kFloatEplison && l.y > -kFloatEplison) {
      res.x = l.x / r.x;
      res.y = 0.0;
    } else {
      res.x = l.x / r.x;
      res.y = l.y / r.y;
    }
    return __float22half2_rn(res);
  }
};


// XLogy check if lhs is less than epsilon, XLogy support half, float, double
template <typename T, typename IsInteger = void>
struct XLogyFunc {
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs < kFloatEplison && lhs > -kFloatEplison ? 0.0 : (lhs * log(rhs));
  }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    Complex<T> res(0.0, 0.0);
    Complex<T> x(lhs.real(), lhs.imag());
    Complex<T> y(rhs.real(), rhs.imag());
    Complex<T> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
};

#ifdef _WIN32
template <typename T>
struct XLogyFunc<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    double tmpLhs = static_cast<double>(lhs);
    double tmpRhs = static_cast<double>(rhs);
    return tmpLhs < kFloatEplison && tmpLhs > -kFloatEplison ? 0.0 : (tmpLhs * log(tmpRhs));
  }
};

template <>
struct XLogyFunc<bool> {
  __device__ __host__ __forceinline__ bool operator()(const bool &lhs, const bool &rhs) {
    if (!lhs || !rhs) {
      return false;
    }
    return true;
  }
};
#endif

template <>
struct XLogyFunc<Complex<float>> {
__device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs, rhs);
    Complex<float> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> y(rhs.real(), rhs.imag());
    Complex<float> x(lhs, lhs);
    Complex<float> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs, lhs);
    Complex<float> y(rhs, rhs);
    Complex<float> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    Complex<float> x(lhs.real(), lhs.imag());
    Complex<float> y(rhs.real(), rhs.imag());
    Complex<float> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
};

template <>
struct XLogyFunc<Complex<double>> {
__device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs, rhs);
    Complex<double> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> y(rhs.real(), rhs.imag());
    Complex<double> x(lhs, lhs);
    Complex<double> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs, lhs);
    Complex<double> y(rhs, rhs);
    Complex<double> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs,
                                                                 const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    Complex<double> x(lhs.real(), lhs.imag());
    Complex<double> y(rhs.real(), rhs.imag());
    Complex<double> mid(0.5 * log(y.real() * y.real() + y.imag() * y.imag()), atan2(y.imag(), y.real()));
    res = x * mid;
    return res;
  }
};

template <>
struct XLogyFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half_rn(__half2float(lhs) * log(__half2float(rhs)));
  }
};

template <>
struct XLogyFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    l.x = (l.x * log(r.x));
    l.y = (l.y * log(r.y));
    return __float22half2_rn(l);
  }
};

// convert to float to fix accuracy issue
// MulNoNan
template <typename T>
struct MulNoNanFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return rhs < kFloatEplison && rhs > -kFloatEplison ? 0.0 : (lhs * rhs);
    }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    Complex<T> res(0.0, 0.0);
    if ((rhs.real() < kFloatEplison && rhs.real() > -kFloatEplison) ||
          (rhs.imag() < kFloatEplison && rhs.imag() > -kFloatEplison)) {
      return res;
    } else {
      Complex<T> x(lhs.real(), lhs.imag());
      Complex<T> y(rhs.real(), rhs.imag());
      res = x*y;
      return res;
    }
  }
};

template <>
struct MulNoNanFunc<Complex<float>> {
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    if ((lhs < 1e-15 && lhs > -1e-15) || (rhs.imag() < 1e-15 && rhs.imag() > -1e-15)) {
      return res;
    } else {
      Complex<float> x(rhs.real(), rhs.imag());
      Complex<float> y(lhs, lhs);
      res = x*y;
      return res;
    }
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    if ((rhs < 1e-15 && rhs > -1e-15) || (lhs.imag() < 1e-15 && lhs.imag() > -1e-15)) {
      return res;
    } else {
      Complex<float> x(lhs.real(), lhs.imag());
      Complex<float> y(rhs, rhs);
      res = x*y;
      return res;
    }
  }
  __device__ __host__ __forceinline__ Complex<float> operator()(const float &lhs, const float &rhs) {
    Complex<float> res(0.0, 0.0);
    if ((rhs < 1e-15 && rhs > -1e-15) || (lhs < 1e-15 && lhs > -1e-15)) {
      return res;
    } else {
      Complex<float> x(lhs, lhs);
      Complex<float> y(rhs, rhs);
      res = x*y;
      return res;
    }
  }
  // __device__ __host__ __forceinline__ float operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
  //     return kFloatEplison;
  // }
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    Complex<float> res(0.0, 0.0);
    if ((rhs.real() < kFloatEplison && rhs.real() > -kFloatEplison) ||
          (rhs.imag() < kFloatEplison && rhs.imag() > -kFloatEplison)) {
      return res;
    } else {
      Complex<float> x(lhs.real(), lhs.imag());
      Complex<float> y(rhs.real(), rhs.imag());
      res = x*y;
      return res;
    }
  }
};

template <>
struct MulNoNanFunc<Complex<double>> {
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    if ((lhs < 1e-15 && lhs > -1e-15) || (rhs.imag() < 1e-15 && rhs.imag() > -1e-15)) {
      return res;
    } else {
      Complex<double> x(rhs.real(), rhs.imag());
      Complex<double> y(lhs, lhs);
      res = x*y;
      return res;
    }
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    if ((rhs < 1e-15 && rhs > -1e-15) || (lhs.imag() < 1e-15 && lhs.imag() > -1e-15)) {
      return res;
    } else {
      Complex<double> x(lhs.real(), lhs.imag());
      Complex<double> y(rhs, rhs);
      res = x*y;
      return res;
    }
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const double &lhs, const double &rhs) {
    Complex<double> res(0.0, 0.0);
    if ((rhs < 1e-15 && rhs > -1e-15) || (lhs < 1e-15 && lhs > -1e-15)) {
      return res;
    } else {
      Complex<double> x(lhs, lhs);
      Complex<double> y(rhs, rhs);
      res = x*y;
      return res;
    }
  }
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs,
                                                                 const Complex<double> &rhs) {
    Complex<double> res(0.0, 0.0);
    if ((rhs.real() < kFloatEplison && rhs.real() > -kFloatEplison) ||
          (rhs.imag() < kFloatEplison && rhs.imag() > -kFloatEplison)) {
      return res;
    } else {
      Complex<double> x(lhs.real(), lhs.imag());
      Complex<double> y(rhs.real(), rhs.imag());
      res = x*y;
      return res;
    }
  }
};

template <>
struct MulNoNanFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    if ((r.x < kFloatEplison && r.x > -kFloatEplison) || (r.y < kFloatEplison && r.y > -kFloatEplison)) {
      l.x = 0.0;
      l.y = 0.0;
    } else {
      l.x = l.x * r.x;
      l.y = l.y * r.y;
    }
    return __float22half2_rn(l);
  }
};

template <>
struct MulNoNanFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    bool bool1 = __half2float(rhs) < (0.00001) && __half2float(rhs) > -0.00001;
    if (bool1) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) * __half2float(rhs));
  }
};

template <typename T>
struct FloorDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return floorf(static_cast<float>(lhs) / static_cast<float>(rhs));
  }
};
template <>
struct FloorDivFunc<int64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const int64_t &lhs, const int64_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<int32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<uint64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const uint64_t &lhs, const uint64_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<uint32_t> {
  __device__ __host__ __forceinline__ uint32_t operator()(const uint32_t &lhs, const uint32_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <>
struct FloorDivFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return floorf(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct FloorDivFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    l.x = floorf(l.x / r.x);
    l.y = floorf(l.y / r.y);
    return __float22half2_rn(l);
  }
};

template <typename T>
struct ModFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T data_div = lhs / rhs;
    T data_div_min = data_div < 0.0 ? data_div : 0.0;
    T data_div_max = data_div > 0.0 ? data_div : 0.0;
    T data_div_max_floor = floorf(data_div_max);
    T data_div_min_ceil = ceilf(data_div_min);
    T data_div_res = data_div_max_floor + data_div_min_ceil;
    return lhs - data_div_res * rhs;
  }
};

template <>
struct ModFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float data_div = l / r;
    float data_div_min = data_div < 0.0 ? data_div : 0.0;
    float data_div_max = data_div > 0.0 ? data_div : 0.0;
    float data_div_max_floor = floorf(data_div_max);
    float data_div_min_ceil = ceilf(data_div_min);
    float data_div_res = data_div_max_floor + data_div_min_ceil;
    return __float2half_rn(l - data_div_res * r);
  }
};

template <>
struct ModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 data_div;
    data_div.x = l.x / r.x;
    data_div.y = l.y / r.y;
    data_div.x = data_div.x < 0.0 ? ceilf(data_div.x) : floorf(data_div.x);
    data_div.y = data_div.y < 0.0 ? ceilf(data_div.y) : floorf(data_div.y);
    data_div.x = l.x - data_div.x * r.x;
    data_div.y = l.y - data_div.y * r.y;
    return __float22half2_rn(data_div);
  }
};

template <typename T>
struct FloorModFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T res = lhs - floorf(lhs / rhs) * rhs;
    res = (abs(res) > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - floorf(l / r) * r;
    res = (abs(res) > 1e-9) && ((res < 0.0) != (r < 0.0)) ? res + r : res;
    return __float2half_rn(res);
  }
};

template <>
struct FloorModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = l.x - floorf(l.x / r.x) * r.x;
    res.y = l.y - floorf(l.y / r.y) * r.y;
    res.x = (abs(res.x) > 1e-9) && ((res.x < 0.0) != (r.x < 0.0)) ? res.x + r.x : res.x;
    res.y = (abs(res.y) > 1e-9) && ((res.y < 0.0) != (r.y < 0.0)) ? res.y + r.y : res.y;
    return __float22half2_rn(res);
  }
};

// the FloorModFunc specializations for uint32_t and uint64_t are there
// because of a 'more than one instance of overloaded function "std::abs"'
// error. I realize the specializations are exactly the same, but I found
// no good alternative.
template <>
struct FloorModFunc<int32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) {
    int32_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<int64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const int64_t &lhs, const int64_t &rhs) {
    int64_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<uint32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const uint32_t &lhs, const uint32_t &rhs) {
    int32_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<uint64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const uint64_t &lhs, const uint64_t &rhs) {
    int64_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <typename T>
struct AbsGradFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T zero = 0.0;
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <>
struct AbsGradFunc<half2> {
  __device__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    half2 zero(0.0, 0.0);
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <typename T>
struct SquaredDifferenceFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T diff = lhs - rhs;
    return diff * diff;
  }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    Complex<T> diff = lhs - rhs;
    Complex<T> conj_diff(diff.real(), -diff.imag());
    return conj_diff * diff;
  }
};

template <>
struct SquaredDifferenceFunc<Complex<double>> {
  __device__ __host__ __forceinline__ Complex<double> operator()(const Complex<double> &lhs,
                                                                 const Complex<double> &rhs) {
    Complex<double> diff = lhs - rhs;
    Complex<double> conj_diff(diff.real(), -diff.imag());
    return conj_diff * diff;
  }
};

template <>
struct SquaredDifferenceFunc<Complex<float>> {
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs,
                                                                 const Complex<float> &rhs) {
    Complex<float> diff = lhs - rhs;
    Complex<float> conj_diff(diff.real(), -diff.imag());
    return conj_diff * diff;
  }
};

template <typename T>
struct TruncateDivFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    double lhs_d = static_cast<double>(lhs);
    double rhs_d = static_cast<double>(rhs);
    double res_d = trunc(lhs_d / rhs_d);
    T res = static_cast<T>(res_d);
    return res;
  }
};

template <>
struct TruncateDivFunc<half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    half res = __hdiv(lhs, rhs);
    return res;
  }
};

template <>
struct TruncateDivFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = truncf(l.x / r.x);
    res.y = truncf(l.y / r.y);
    return __float22half2_rn(res);
  }
};

template <typename T>
struct TruncateModFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T res = static_cast<T>(lhs - static_cast<int>(lhs / rhs) * rhs);
    return res;
  }
};

template <>
struct TruncateModFunc<half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - static_cast<int>(l / r) * r;
    return __float2half_rn(res);
  }
};

template <>
struct TruncateModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = l.x - static_cast<int>(l.x / r.x) * r.x;
    res.y = l.y - static_cast<int>(l.y / r.y) * r.y;
    return __float22half2_rn(res);
  }
};

template <typename T>
struct Atan2Func {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return atan2f(lhs, rhs); }
};

template <>
struct Atan2Func<double> {
  __device__ __host__ __forceinline__ double operator()(const double &lhs, const double &rhs) {
    return atan2(lhs, rhs);
  }
};

template <>
struct Atan2Func<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = atan2f(l, r);
    return __float2half_rn(res);
  }
};

template <>
struct Atan2Func<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = atan2f(l.x, r.x);
    res.y = atan2f(l.y, r.y);
    return __float22half2_rn(res);
  }
};

// Element-wise Comparison
template <typename T, typename Func>
__global__ void ElewiseCmpKernel(const int nums, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseCmp(const int &nums, enum BinaryOpType op, const T *x0, const T *x1, bool *y, cudaStream_t stream) {
  switch (op) {
    case BinaryOpType::kGreater:
      return ElewiseCmpKernel<T, GreaterFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kLess:
      return ElewiseCmpKernel<T, LessFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kEqual:
      return ElewiseCmpKernel<T, EqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kGreaterEqual:
      return ElewiseCmpKernel<T, GreaterEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kLessEqual:
      return ElewiseCmpKernel<T, LessEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kNotEqual:
      return ElewiseCmpKernel<T, NotEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kLogicalAnd:
      return ElewiseCmpKernel<T, LogicalAndFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kLogicalOr:
      return ElewiseCmpKernel<T, LogicalOrFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const double *x0, const double *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const float *x0, const float *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const half *x0, const half *x1, bool *y,
                                         cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const int *x0, const int *x1, bool *y,
                                         cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const int8_t *x0, const int8_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const uint8_t *x0, const uint8_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const int64_t *x0, const int64_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const int16_t *x0, const int16_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const uint16_t *x0, const uint16_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const uint32_t *x0, const uint32_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const uint64_t *x0, const uint64_t *x1,
                                         bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BinaryOpType op, const bool *x0, const bool *x1, bool *y,
                                         cudaStream_t stream);
// Element-wise ArithMetic
template <typename T, typename Func>
__global__ void ElewiseArithKernel(const int nums, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T1, typename T2, typename T3, typename Func>
__global__ void ElewiseArithComplexKernel(const int nums, const T1 *x0, const T2 *x1, Complex<T3> *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseArithKernel(const int &nums, enum BinaryOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  switch (op) {
    case BinaryOpType::kMinimum:
      return ElewiseArithKernel<T, MinimumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMaximum:
      return ElewiseArithKernel<T, MaximumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kPower:
      return ElewiseArithKernel<T, PowerFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kRealDiv:
      return ElewiseArithKernel<T, RealDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kBitwiseAnd:
      return ElewiseArithKernel<T, BitwiseAnd<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kBitwiseOr:
      return ElewiseArithKernel<T, BitwiseOr<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kBitwiseXor:
      return ElewiseArithKernel<T, BitwiseXor<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMul:
      return ElewiseArithKernel<T, MulFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kSub:
      return ElewiseArithKernel<T, SubFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kAdd:
      return ElewiseArithKernel<T, AddFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kFloorDiv:
      return ElewiseArithKernel<T, FloorDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kAbsGrad:
      return ElewiseArithKernel<T, AbsGradFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kDiv:
      return ElewiseArithKernel<T, DivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kDivNoNan:
      return ElewiseArithKernel<T, DivNoNanFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kSquaredDifference:
      return ElewiseArithKernel<T, SquaredDifferenceFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kTruncateDiv:
      return ElewiseArithKernel<T, TruncateDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kTruncateMod:
      return ElewiseArithKernel<T, TruncateModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMod:
      return ElewiseArithKernel<T, ModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kFloorMod:
      return ElewiseArithKernel<T, FloorModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kAtan2:
      return ElewiseArithKernel<T, Atan2Func<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kXdivy:
      return ElewiseArithKernel<T, XDivyFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMulNoNan:
      return ElewiseArithKernel<T, MulNoNanFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kXlogy:
      return ElewiseArithKernel<T, XLogyFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template <typename T1, typename T2, typename T3>
void ElewiseArithComplexKernel(const int &nums, enum BinaryOpType op, const T1 *x0, const T2 *x1, Complex<T3> *y,
                               cudaStream_t stream) {
  switch (op) {
    case BinaryOpType::kAdd:
      return ElewiseArithComplexKernel<T1, T2, T3, AddFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kSub:
      return ElewiseArithComplexKernel<T1, T2, T3, SubFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMul:
      return ElewiseArithComplexKernel<T1, T2, T3, MulFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kDiv:
      return ElewiseArithComplexKernel<T1, T2, T3, DivFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kRealDiv:
      return ElewiseArithComplexKernel<T1, T2, T3, RealDivFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kComplex:
      return ElewiseArithComplexKernel<T1, T2, T3, ComplexFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kMulNoNan:
      return ElewiseArithComplexKernel<T1, T2, T3, MulNoNanFunc<Complex<T3>>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kPower:
      return ElewiseArithComplexKernel<T1, T2, T3, PowerFunc<Complex<T3>>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kXdivy:
      return ElewiseArithComplexKernel<T1, T2, T3, XDivyFunc<Complex<T3>>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kXlogy:
      return ElewiseArithComplexKernel<T1, T2, T3, XLogyFunc<Complex<T3>>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BinaryOpType::kSquaredDifference:
      return ElewiseArithComplexKernel<T1, T2, T3, SquaredDifferenceFunc<Complex<T3>>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template <typename T>
void ElewiseArith(const int &nums, enum BinaryOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  return ElewiseArithKernel(nums, op, x0, x1, y, stream);
}

template <>
void ElewiseArith(const int &nums, enum BinaryOpType op, const half *x0, const half *x1, half *y, cudaStream_t stream) {
  // `>` return true iff both half result are true. fallback to half
  if (nums % 2 == 0 && op != BinaryOpType::kMinimum && op != BinaryOpType::kMaximum && op != BinaryOpType::kAbsGrad) {
    ElewiseArithKernel<half2>(nums / 2, op, reinterpret_cast<const half2 *>(x0), reinterpret_cast<const half2 *>(x1),
                              reinterpret_cast<half2 *>(y), stream);
  } else {
    return ElewiseArithKernel(nums, op, x0, x1, y, stream);
  }
}

template <typename T1, typename T2, typename T3>
void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const T1 *x0, const T2 *x1, Complex<T3> *y,
                         cudaStream_t stream) {
  return ElewiseArithComplexKernel(nums, op, x0, x1, y, stream);
}

template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const double *x0, const double *x1,
                                           double *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const float *x0, const float *x1,
                                           float *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const half *x0, const half *x1,
                                           half *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const int *x0, const int *x1, int *y,
                                           cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const int8_t *x0, const int8_t *x1,
                                           int8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const uint8_t *x0, const uint8_t *x1,
                                           uint8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const int64_t *x0, const int64_t *x1,
                                           int64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const int16_t *x0, const int16_t *x1,
                                           int16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const uint16_t *x0,
                                           const uint16_t *x1, uint16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const uint32_t *x0,
                                           const uint32_t *x1, uint32_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const uint64_t *x0,
                                           const uint64_t *x1, uint64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BinaryOpType op, const bool *x0, const bool *x1,
                                           bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const Complex<float> *x0,
                                                  const Complex<float> *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const Complex<float> *x0,
                                                  const float *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const float *x0,
                                                  const Complex<float> *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const Complex<double> *x0,
                                                  const Complex<double> *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const Complex<double> *x0,
                                                  const double *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const double *x0,
                                                  const Complex<double> *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const float *x0,
                                                  const float *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BinaryOpType op, const double *x0,
                                                  const double *x1, Complex<double> *y, cudaStream_t stream);

// Broadcast comparison
__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastCmpKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                   const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                   const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                   const size_t d6, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                  const std::vector<size_t> &y_dims, enum BinaryOpType op, const T *x0, const T *x1, bool *y,
                  cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }

  switch (op) {
    case BinaryOpType::kGreater:
      return BroadcastCmpKernel<T, GreaterFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kLess:
      return BroadcastCmpKernel<T, LessFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kEqual:
      return BroadcastCmpKernel<T, EqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kGreaterEqual:
      return BroadcastCmpKernel<T, GreaterEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kLessEqual:
      return BroadcastCmpKernel<T, LessEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kNotEqual:
      return BroadcastCmpKernel<T, NotEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kLogicalAnd:
      return BroadcastCmpKernel<T, LogicalAndFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kLogicalOr:
      return BroadcastCmpKernel<T, LogicalOrFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const double *x0,
                                           const double *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const float *x0,
                                           const float *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const half *x0,
                                           const half *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const int *x0,
                                           const int *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const int8_t *x0,
                                           const int8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const uint8_t *x0,
                                           const uint8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const int64_t *x0,
                                           const int64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const int16_t *x0,
                                           const int16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const uint16_t *x0,
                                           const uint16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const uint32_t *x0,
                                           const uint32_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const uint64_t *x0,
                                           const uint64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const bool *x0,
                                           const bool *x1, bool *y, cudaStream_t stream);
// Broadcast Arithmetic
template <typename T, typename Func>
__global__ void BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                     const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                     const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                     const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                     const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                     const size_t d6, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T1, typename T2, typename T3, typename Func>
__global__ void BroadcastComplexArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                            const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                            const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                            const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                            const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                            const size_t d6, const T1 *x0, const T2 *x1, Complex<T3> *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                    const std::vector<size_t> &y_dims, enum BinaryOpType op, const T *x0, const T *x1, T *y,
                    cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  switch (op) {
    case BinaryOpType::kMaximum:
      return BroadcastArithKernel<T, MaximumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMinimum:
      return BroadcastArithKernel<T, MinimumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kPower:
      return BroadcastArithKernel<T, PowerFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kRealDiv:
      return BroadcastArithKernel<T, RealDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kBitwiseAnd:
      return BroadcastArithKernel<T, BitwiseAnd<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kBitwiseOr:
      return BroadcastArithKernel<T, BitwiseOr<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kBitwiseXor:
      return BroadcastArithKernel<T, BitwiseXor<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMul:
      return BroadcastArithKernel<T, MulFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kSub:
      return BroadcastArithKernel<T, SubFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kAdd:
      return BroadcastArithKernel<T, AddFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kFloorDiv:
      return BroadcastArithKernel<T, FloorDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kAbsGrad:
      return BroadcastArithKernel<T, AbsGradFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kDiv:
      return BroadcastArithKernel<T, DivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kDivNoNan:
      return BroadcastArithKernel<T, DivNoNanFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kSquaredDifference:
      return BroadcastArithKernel<T, SquaredDifferenceFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kTruncateDiv:
      return BroadcastArithKernel<T, TruncateDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kTruncateMod:
      return BroadcastArithKernel<T, TruncateModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMod:
      return BroadcastArithKernel<T, ModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kFloorMod:
      return BroadcastArithKernel<T, FloorModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kAtan2:
      return BroadcastArithKernel<T, Atan2Func<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kXdivy:
      return BroadcastArithKernel<T, XDivyFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMulNoNan:
      return BroadcastArithKernel<T, MulNoNanFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kXlogy:
      return BroadcastArithKernel<T, XLogyFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template <typename T1, typename T2, typename T3>
void BroadcastComplexArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BinaryOpType op, const T1 *x0, const T2 *x1,
                           Complex<T3> *y, cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  switch (op) {
    case BinaryOpType::kAdd:
      return BroadcastComplexArithKernel<T1, T2, T3, AddFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kSub:
      return BroadcastComplexArithKernel<T1, T2, T3, SubFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMul:
      return BroadcastComplexArithKernel<T1, T2, T3, MulFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kDiv:
      return BroadcastComplexArithKernel<T1, T2, T3, DivFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kRealDiv:
      return BroadcastComplexArithKernel<T1, T2, T3, RealDivFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kMulNoNan:
      return BroadcastComplexArithKernel<T1, T2, T3, MulNoNanFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kPower:
      return BroadcastComplexArithKernel<T1, T2, T3, PowerFunc<Complex<T3>>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kXdivy:
      return BroadcastComplexArithKernel<T1, T2, T3, XDivyFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kXlogy:
      return BroadcastComplexArithKernel<T1, T2, T3, XLogyFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kSquaredDifference:
      return BroadcastComplexArithKernel<T1, T2, T3, SquaredDifferenceFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BinaryOpType::kComplex:
      return BroadcastComplexArithKernel<T1, T2, T3, ComplexFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const double *x0,
                                             const double *x1, double *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const float *x0,
                                             const float *x1, float *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const half *x0,
                                             const half *x1, half *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const int8_t *x0,
                                             const int8_t *x1, int8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const uint8_t *x0,
                                             const uint8_t *x1, uint8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const int16_t *x0,
                                             const int16_t *x1, int16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                             const uint16_t *x0, const uint16_t *x1, uint16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const int32_t *x0,
                                             const int32_t *x1, int32_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                             const uint32_t *x0, const uint32_t *x1, uint32_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const int64_t *x0,
                                             const int64_t *x1, int64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                             const uint64_t *x0, const uint64_t *x1, uint64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BinaryOpType op, const bool *x0,
                                             const bool *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const Complex<float> *x0, const float *x1, Complex<float> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const float *x0, const Complex<float> *x1, Complex<float> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const Complex<double> *x0, const double *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const double *x0, const Complex<double> *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const Complex<double> *x0, const Complex<double> *x1,
                                                    Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const Complex<float> *x0, const Complex<float> *x1,
                                                    Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const double *x0, const double *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BinaryOpType op,
                                                    const float *x0, const float *x1, Complex<float> *y,
                                                    cudaStream_t stream);

// BroadcastTo
template <typename T>
__global__ void BroadcastToKernel(const size_t i0, const size_t i1, const size_t i2, const size_t i3,
                                  const size_t i4, const size_t i5, const size_t i6, const size_t i7,
                                  const size_t o0, const size_t o1, const size_t o2, const size_t o3,
                                  const size_t o4, const size_t o5, const size_t o6, const size_t o7,
                                  const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3 * o4 * o5 * o6 * o7;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6 * o7) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6 * o7) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6 * o7) % o2;
    size_t l = pos / (o4 * o5 * o6 * o7) % o3;
    size_t m = pos / (o5 * o6 * o7) % o4;
    size_t n = pos / (o6 * o7) % o5;
    size_t o = pos / o7 % o6;
    size_t p = pos % o7;

    size_t input_idx = Index(i, i0) * i1 * i2 * i3 * i4 * i5 * i6 * i7
                     + Index(j, i1) * i2 * i3 * i4 * i5 * i6 * i7
                     + Index(k, i2) * i3 * i4 * i5 * i6 * i7
                     + Index(l, i3) * i4 * i5 * i6 * i7
                     + Index(m, i4) * i5 * i6 * i7
                     + Index(n, i5) * i6 * i7
                     + Index(o, i6) * i7
                     + Index(p, i7);
    output_addr[pos] = input_addr[input_idx];
  }
}

template <typename T>
void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &i4,
                 const size_t &i5, const size_t &i6, const size_t &i7, const size_t &o0, const size_t &o1,
                 const size_t &o2, const size_t &o3, const size_t &o4, const size_t &o5, const size_t &o6,
                 const size_t &o7, const T *input_addr, T *output_addr, cudaStream_t stream) {
  size_t nums = o0 * o1 * o2 * o3 * o4 * o5 * o6 * o7;
  int block_num = 256 > nums ? nums : 256;
  BroadcastToKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, nums, block_num), block_num, 0, stream>>>(
    i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7, input_addr, output_addr);
}

template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const double *input_addr, double *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const float *input_addr, float *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const half *input_addr, half *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const int8_t *input_addr, int8_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const int16_t *input_addr, int16_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const int32_t *input_addr, int32_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const int64_t *input_addr, int64_t *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const uint8_t *input_addr, uint8_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const uint16_t *input_addr, uint16_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const uint32_t *input_addr, uint32_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const uint64_t *input_addr, uint64_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const bool *input_addr, bool *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const Complex<float> *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                          const Complex<double> *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);
