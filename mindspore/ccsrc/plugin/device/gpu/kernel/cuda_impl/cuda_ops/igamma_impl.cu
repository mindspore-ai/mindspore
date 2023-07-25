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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/igamma_impl.cuh"
#include <algorithm>
#include <limits>
#include "include/cuda_runtime.h"
#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406L
enum IgammaComputationMode { VALUE, DERIVATIVE, SAMPLE_DERIVATIVE };
template <typename Scalar>
struct cephes_helper {
  __device__ static __forceinline__ Scalar machep() { return 0.0; }
  __device__ static __forceinline__ Scalar big() { return 0.0; }
  __device__ static __forceinline__ Scalar biginv() { return 0.0; }
};

template <>
struct cephes_helper<float> {
  __device__ static __forceinline__ float machep() {
    return std::numeric_limits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  __device__ static __forceinline__ float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0f / (std::numeric_limits<float>::epsilon() / 2);
  }
  __device__ static __forceinline__ float biginv() {
    // epsneg
    return machep();
  }
};

template <>
struct cephes_helper<double> {
  __device__ static __forceinline__ double machep() {
    return std::numeric_limits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  __device__ static __forceinline__ double big() { return 1.0 / std::numeric_limits<double>::epsilon(); }
  __device__ static __forceinline__ double biginv() {
    // inverse of eps
    return std::numeric_limits<double>::epsilon();
  }
};
template <typename T, IgammaComputationMode mode>
__device__ int igamma_num_iterations() {
  /* Returns the maximum number of internal iterations for igamma computation.
   */
  if (mode == VALUE) {
    return 2000;
  }

  if (std::is_same<T, float>::value) {
    return 200;
  } else if (std::is_same<T, double>::value) {
    return 500;
  } else {
    return 2000;
  }
}
template <typename T>
__forceinline__ __device__ T lgammaFunc(T a) {
  return lgamma(a);
}
template <>
__forceinline__ __device__ float lgammaFunc(float a) {
  return lgammaf(a);
}

template <typename T>
__forceinline__ __device__ T floorFunc(T a) {
  return floor(a);
}
template <>
__forceinline__ __device__ float floorFunc(float a) {
  return floorf(a);
}
template <typename T>
__forceinline__ __device__ T tanFunc(T a) {
  return tan(a);
}
template <>
__forceinline__ __device__ float tanFunc(float a) {
  return tanf(a);
}

template <typename T>
__forceinline__ __device__ T logFunc(T a) {
  return log(a);
}
template <>
__forceinline__ __device__ float logFunc(float a) {
  return logf(a);
}
template <typename T>
__forceinline__ __device__ T expFunc(T a) {
  return exp(a);
}
template <>
__forceinline__ __device__ float expFunc(float a) {
  return expf(a);
}
template <typename T>
__forceinline__ __device__ T fabsFunc(T a) {
  return fabs(a);
}
template <>
__forceinline__ __device__ float fabsFunc(float a) {
  return fabsf(a);
}

template <typename T>
static __forceinline__ __device__ T main_igamma_term(T a, T x) {
  /* Compute  x**a * exp(-x) / gamma(a)  */
  T logax = a * logFunc<T>(x) - x - lgammaFunc<T>(a);
  if (logax < -logFunc<T>(std::numeric_limits<T>::max()) || isnan(logax)) {
    return static_cast<T>(0);
  }
  return expFunc<T>(logax);
}

static __forceinline__ __device__ float ppolevl3(float x, const float coeff[]) {
  float ans = coeff[0];
  ans *= x;
  ans += coeff[1];
  ans *= x;
  ans += coeff[2];
  ans *= x;
  ans += coeff[3];
  return ans;
}

static __forceinline__ __device__ double ppolevl6(double x, const double coeff[]) {
  double ans = coeff[0];
  ans *= x;
  ans += coeff[1];
  ans *= x;
  ans += coeff[2];
  ans *= x;
  ans += coeff[3];
  ans *= x;
  ans += coeff[4];
  ans *= x;
  ans += coeff[5];
  ans *= x;
  ans += coeff[6];
  return ans;
}
static __forceinline__ __device__ float digamma_impl_maybe_poly(float s) {
  const float A[] = {-4.16666666666666666667E-3f, 3.96825396825396825397E-3f, -8.33333333333333333333E-3f,
                     8.33333333333333333333E-2f};
  float z;
  if (s < 1.0e8f) {
    z = 1.0f / (s * s);
    return z * ppolevl3(z, A);
  } else {
    return 0.0f;
  }
}
static __forceinline__ __device__ double digamma_impl_maybe_poly(double s) {
  const double A[] = {8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
                      -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                      8.33333333333333333333E-2};
  double z;
  if (s < 1.0e17) {
    z = 1.0 / (s * s);
    return z * ppolevl6(z, A);
  } else {
    return 0.0;
  }
}
template <typename T>
static __device__ T digamma_impl(T x) {
  T p, q, nz, s, w, y;
  bool negative = false;

  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T m_pi = T(PI);

  const T zero = T(0);
  const T one = T(1);
  const T half = T(0.5);
  nz = zero;

  if (x <= zero) {
    negative = true;
    q = x;
    p = floorFunc<T>(q);
    if (p == q) {
      return nan;
    }
    /* Remove the zeros of tan(m_pi x)
     * by subtracting the nearest integer from x
     */
    nz = q - p;
    if (nz != half) {
      if (nz > half) {
        p += one;
        nz = q - p;
      }
      nz = m_pi / tanFunc<T>(m_pi * nz);
    } else {
      nz = zero;
    }
    x = one - x;
  }

  /* use the recurrence psi(x+1) = psi(x) + 1/x. */
  s = x;
  w = zero;
  while (s < T(10)) {
    w += one / s;
    s += one;
  }

  y = digamma_impl_maybe_poly(s);

  y = logFunc<T>(s) - (half / s) - y - w;

  return (negative) ? y - nz : y;
}

template <typename T, IgammaComputationMode mode>
static __device__ T igammac_cf_impl(T a, T x) {
  const T zero = 0;
  const T one = 1;
  const T two = 2;
  const T machep = cephes_helper<T>::machep();
  const T big = cephes_helper<T>::big();
  const T biginv = cephes_helper<T>::biginv();

  if (isinf(x)) {
    return zero;
  }

  T ax = main_igamma_term<T>(a, x);
  // This is independent of mode. If this value is zero,
  // then the function value is zero. If the function value is zero,
  // then we are in a neighborhood where the function value evaluates to zero,
  // so the derivative is zero.
  if (ax == zero) {
    return zero;
  }

  // continued fraction
  T y = one - a;
  T z = x + y + one;
  T c = zero;
  T pkm2 = one;
  T qkm2 = x;
  T pkm1 = x + one;
  T qkm1 = z * x;
  T ans = pkm1 / qkm1;

  T dpkm2_da = zero;
  T dqkm2_da = zero;
  T dpkm1_da = zero;
  T dqkm1_da = -x;
  T dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;

  for (int i = 0; i < igamma_num_iterations<T, mode>(); i++) {
    c += one;
    y += one;
    z += two;

    T yc = y * c;
    T pk = pkm1 * z - pkm2 * yc;
    T qk = qkm1 * z - qkm2 * yc;

    T dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
    T dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;

    if (qk != zero) {
      T ans_prev = ans;
      ans = pk / qk;

      T dans_da_prev = dans_da;
      dans_da = (dpk_da - ans * dqk_da) / qk;

      if (mode == VALUE) {
        if (fabsFunc<T>(ans_prev - ans) <= machep * fabsFunc<T>(ans)) {
          break;
        }
      } else {
        if (fabsFunc<T>(dans_da - dans_da_prev) <= machep) {
          break;
        }
      }
    }

    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;

    dpkm2_da = dpkm1_da;
    dpkm1_da = dpk_da;
    dqkm2_da = dqkm1_da;
    dqkm1_da = dqk_da;

    if (fabsFunc<T>(pk) > big) {
      pkm2 *= biginv;
      pkm1 *= biginv;
      qkm2 *= biginv;
      qkm1 *= biginv;

      dpkm2_da *= biginv;
      dpkm1_da *= biginv;
      dqkm2_da *= biginv;
      dqkm1_da *= biginv;
    }
  }

  /* Compute  x**a * exp(-x) / gamma(a)  */
  T dlogax_da = logFunc<T>(x) - digamma_impl<T>(a);
  T dax_da = ax * dlogax_da;

  switch (mode) {
    case VALUE:
      return ans * ax;
    case DERIVATIVE:
      return ans * dax_da + dans_da * ax;
    case SAMPLE_DERIVATIVE:
    default:  // this is needed to suppress clang warning
      return -(dans_da + ans * dlogax_da) * x;
  }
}

template <typename T, IgammaComputationMode mode>
static __device__ T igamma_series_impl(T a, T x) {
  const T zero = 0;
  const T one = 1;
  const T machep = cephes_helper<T>::machep();

  T ax = main_igamma_term<T>(a, x);

  // This is independent of mode. If this value is zero,
  // then the function value is zero. If the function value is zero,
  // then we are in a neighborhood where the function value evaluates to zero,
  // so the derivative is zero.
  if (ax == zero) {
    return zero;
  }

  ax /= a;

  /* power series */
  T r = a;
  T c = one;
  T ans = one;

  T dc_da = zero;
  T dans_da = zero;

  for (int i = 0; i < igamma_num_iterations<T, mode>(); i++) {
    r += one;
    T term = x / r;
    T dterm_da = -x / (r * r);
    dc_da = term * dc_da + dterm_da * c;
    dans_da += dc_da;
    c *= term;
    ans += c;

    if (mode == VALUE) {
      if (c <= machep * ans) {
        break;
      }
    } else {
      if (fabsFunc<T>(dc_da) <= machep * fabsFunc<T>(dans_da)) {
        break;
      }
    }
  }

  T dlogax_da = logFunc<T>(x) - digamma_impl<T>(a + one);
  T dax_da = ax * dlogax_da;

  switch (mode) {
    case VALUE:
      return ans * ax;
    case DERIVATIVE:
      return ans * dax_da + dans_da * ax;
    case SAMPLE_DERIVATIVE:
    default:  // this is needed to suppress clang warning
      return -(dans_da + ans * dlogax_da) * x / a;
  }
}

template <typename T, IgammaComputationMode mode>
static __device__ T igamma_generic_impl(T a, T x) {
  const T zero = 0;
  const T one = 1;
  const T nan = std::numeric_limits<T>::quiet_NaN();

  if (x == zero) return zero;

  if ((x < zero) || (a <= zero)) {  // domain error
    return nan;
  }

  if (isnan(a) || isnan(x)) {  // propagate nans
    return nan;
  }

  if ((x > one) && (x > a)) {
    T ret = igammac_cf_impl<T, mode>(a, x);
    if (mode == VALUE) {
      return one - ret;
    } else {
      return -ret;
    }
  }

  return igamma_series_impl<T, mode>(a, x);
}

template <typename T>
static __device__ T IgammaSingle(T a, T x) {
  return igamma_generic_impl<T, VALUE>(a, x);
}

template <typename T>
static __device__ T IgammacSingle(T a, T x) {
  const T zero = 0;
  const T one = 1;
  const T nan = std::numeric_limits<T>::quiet_NaN();

  if ((x < zero) || (a <= zero)) {
    // domain error
    return nan;
  }

  if (isnan(a) || isnan(x)) {  // propagate nans
    return nan;
  }

  if ((x < one) || (x < a)) {
    return (one - igamma_series_impl<T, VALUE>(a, x));
  }

  return igammac_cf_impl<T, VALUE>(a, x);
}

template <typename T>
static __device__ T IgammaGradASingle(T a, T x) {
  return igamma_generic_impl<T, DERIVATIVE>(a, x);
}

template <typename T>
__global__ void Igamma(size_t size, const int64_t type, const T *a, const T *x, T *output) {
  switch (type) {
    case (kLgammaSameShape):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaSingle(a[i], x[i]);
      }
      break;
    case (kLgammaAOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaSingle(*a, x[i]);
      }
      break;
    case (kLgammaXOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaSingle(a[i], *x);
      }
  }
}

template <typename T>
__global__ void Igammac(size_t size, const int64_t type, const T *a, const T *x, T *output) {
  switch (type) {
    case (kLgammaSameShape):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammacSingle(a[i], x[i]);
      }
      break;
    case (kLgammaAOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammacSingle(*a, x[i]);
      }
      break;
    case (kLgammaXOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammacSingle(a[i], *x);
      }
  }
}

template <typename T>
__global__ void IgammaGradA(size_t size, const int64_t type, const T *a, const T *x, T *output) {
  switch (type) {
    case (kLgammaSameShape):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaGradASingle(a[i], x[i]);
      }
      break;
    case (kLgammaAOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaGradASingle(*a, x[i]);
      }
      break;
    case (kLgammaXOneElement):
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = IgammaGradASingle(a[i], *x);
      }
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void BroadcastIgamma(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                const size_t d6, const T *a, const T *x, T *output) {
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
    output[pos] = IgammaSingle(a[l_index], x[r_index]);
  }
}

template <typename T>
__global__ void BroadcastIgammac(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                 const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                 const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                 const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                 const size_t d6, const T *a, const T *x, T *output) {
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
    output[pos] = IgammacSingle(a[l_index], x[r_index]);
  }
}
template <typename T>
__global__ void BroadcastIgammaGradA(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                     const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                     const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                     const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                     const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                     const size_t d6, const T *a, const T *x, T *output) {
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
    output[pos] = IgammaGradASingle(a[l_index], x[r_index]);
  }
}

template <typename T>
cudaError_t CalIgamma(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  Igamma<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, type, a, x, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalIgammac(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                       const uint32_t &device_id, cudaStream_t cuda_stream) {
  Igammac<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, type, a, x, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalIgammaGradA(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                           const uint32_t &device_id, cudaStream_t cuda_stream) {
  int thread_num = 768 < size ? 768 : size;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((size - 1) / thread_num) + 1), max_blocks);
  IgammaGradA<<<block_num, thread_num, 0, cuda_stream>>>(size, type, a, x, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBroadcastIgammac(const std::vector<size_t> &inputa_shape, const std::vector<size_t> &inputx_shape,
                                const std::vector<size_t> &output_shape, const T *inputa, const T *inputx, T *output,
                                const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  BroadcastIgammac<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    inputa_shape[0], inputa_shape[1], inputa_shape[2], inputa_shape[3], inputa_shape[4], inputa_shape[5],
    inputa_shape[6], inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4],
    inputx_shape[5], inputx_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_shape[4], output_shape[5], output_shape[6], inputa, inputx, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBroadcastIgamma(const std::vector<size_t> &inputa_shape, const std::vector<size_t> &inputx_shape,
                               const std::vector<size_t> &output_shape, const T *inputa, const T *inputx, T *output,
                               const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  BroadcastIgamma<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    inputa_shape[0], inputa_shape[1], inputa_shape[2], inputa_shape[3], inputa_shape[4], inputa_shape[5],
    inputa_shape[6], inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4],
    inputx_shape[5], inputx_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_shape[4], output_shape[5], output_shape[6], inputa, inputx, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBroadcastIgammaGradA(const std::vector<size_t> &inputa_shape, const std::vector<size_t> &inputx_shape,
                                    const std::vector<size_t> &output_shape, const T *inputa, const T *inputx,
                                    T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  int thread_num = 768 < size ? 768 : size;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((size - 1) / thread_num) + 1), max_blocks);
  BroadcastIgammaGradA<<<block_num, thread_num, 0, cuda_stream>>>(
    inputa_shape[0], inputa_shape[1], inputa_shape[2], inputa_shape[3], inputa_shape[4], inputa_shape[5],
    inputa_shape[6], inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4],
    inputx_shape[5], inputx_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_shape[4], output_shape[5], output_shape[6], inputa, inputx, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalIgamma<float>(const size_t size, const int64_t type, const float *a,
                                                      const float *x, float *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalIgamma<double>(const size_t size, const int64_t type, const double *a,
                                                       const double *x, double *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgamma<float>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                               const std::vector<size_t> &, const float *,
                                                               const float *, float *, const uint32_t &,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgamma<double>(const std::vector<size_t> &,
                                                                const std::vector<size_t> &,
                                                                const std::vector<size_t> &, const double *,
                                                                const double *, double *, const uint32_t &,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIgammac<float>(const size_t size, const int64_t type, const float *a,
                                                       const float *x, float *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalIgammac<double>(const size_t size, const int64_t type, const double *a,
                                                        const double *x, double *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammac<float>(const std::vector<size_t> &,
                                                                const std::vector<size_t> &,
                                                                const std::vector<size_t> &, const float *,
                                                                const float *, float *, const uint32_t &,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammac<double>(const std::vector<size_t> &,
                                                                 const std::vector<size_t> &,
                                                                 const std::vector<size_t> &, const double *,
                                                                 const double *, double *, const uint32_t &,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalIgammaGradA<float>(const size_t size, const int64_t type, const float *inputa,
                                                           const float *inputx, float *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalIgammaGradA<double>(const size_t size, const int64_t type, const double *inputa,
                                                            const double *inputx, double *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammaGradA<float>(const std::vector<size_t> &,
                                                                    const std::vector<size_t> &,
                                                                    const std::vector<size_t> &, const float *,
                                                                    const float *, float *, const uint32_t &,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammaGradA<double>(const std::vector<size_t> &,
                                                                     const std::vector<size_t> &,
                                                                     const std::vector<size_t> &, const double *,
                                                                     const double *, double *, const uint32_t &,
                                                                     cudaStream_t cuda_stream);
