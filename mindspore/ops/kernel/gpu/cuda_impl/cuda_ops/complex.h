/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_COPLEX_H_
#define MINDSPORE_CCSRC_UTILS_COPLEX_H_

#ifdef ENABLE_GPU
#include <thrust/complex.h>
#include <cuda_fp16.h>
#endif
#include <complex>
#include <limits>
#include "base/float16.h"
#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace mindspore {
namespace utils {
// Implement Complex for mindspore, inspired by std::complex.
constexpr int T_SIZE = 2;
template <typename T>
struct alignas(sizeof(T) * T_SIZE) Complex {
  Complex() = default;
  ~Complex() = default;

  Complex(const Complex<T> &other) noexcept = default;
  Complex(Complex<T> &&other) noexcept = default;

  Complex &operator=(const Complex<T> &other) noexcept = default;
  Complex &operator=(Complex<T> &&other) noexcept = default;

  HOST_DEVICE inline constexpr Complex(const T &real, const T &imag = T()) : real_(real), imag_(imag) {}

  template <typename U>
  inline explicit constexpr Complex(const std::complex<U> &other) : Complex(other.real(), other.imag()) {}
  template <typename U>
  inline explicit constexpr operator std::complex<U>() const {
    return std::complex<U>(std::complex<T>(real(), imag()));
  }

  HOST_DEVICE inline explicit constexpr Complex(const float16 &real) : real_(static_cast<T>(real)), imag_(T()) {}
#if defined(__CUDACC__)
  template <typename U>
  HOST_DEVICE inline explicit Complex(const thrust::complex<U> &other) : real_(other.real()), imag_(other.imag()) {}

  template <typename U>
  HOST_DEVICE inline HOST_DEVICE explicit operator thrust::complex<U>() const {
    return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
  }
#endif
  template <typename U = T>
  HOST_DEVICE inline explicit Complex(const Complex<U> &other)
      : real_(static_cast<T>(other.real())), imag_(static_cast<T>(other.imag())) {}

  HOST_DEVICE inline explicit operator bool() const { return static_cast<bool>(real_) || static_cast<bool>(imag_); }
  HOST_DEVICE inline explicit operator signed char() const { return static_cast<signed char>(real_); }
  HOST_DEVICE inline explicit operator unsigned char() const { return static_cast<unsigned char>(real_); }
  HOST_DEVICE inline explicit operator double() const { return static_cast<double>(real_); }
  HOST_DEVICE inline explicit operator float() const { return static_cast<float>(real_); }
  HOST_DEVICE inline explicit operator int16_t() const { return static_cast<int16_t>(real_); }
  HOST_DEVICE inline explicit operator uint16_t() const { return static_cast<uint16_t>(real_); }
  HOST_DEVICE inline explicit operator int32_t() const { return static_cast<int32_t>(real_); }
  HOST_DEVICE inline explicit operator uint32_t() const { return static_cast<uint32_t>(real_); }
  HOST_DEVICE inline explicit operator int64_t() const { return static_cast<int64_t>(real_); }
  HOST_DEVICE inline explicit operator uint64_t() const { return static_cast<uint64_t>(real_); }
#if defined(__CUDACC__)
  HOST_DEVICE inline explicit operator half() const { return static_cast<half>(real_); }
#else
  inline explicit operator float16() const { return static_cast<float16>(real_); }
#endif

  HOST_DEVICE inline Complex<T> &operator=(const T &real) {
    real_ = real;
    imag_ = T();
    return *this;
  }

  HOST_DEVICE inline Complex<T> &operator+=(const T &real) {
    real_ += real;
    return *this;
  }

  HOST_DEVICE inline Complex<T> &operator-=(const T &real) {
    real_ -= real;
    return *this;
  }

  HOST_DEVICE inline Complex<T> &operator*=(const T &real) {
    real_ *= real;
    imag_ *= real;
    return *this;
  }

  // Note: check division by zero before use it.
  HOST_DEVICE inline Complex<T> &operator/=(const T &real) {
    real_ /= real;
    imag_ /= real;
    return *this;
  }

  template <typename U>
  HOST_DEVICE inline Complex<T> &operator=(const Complex<U> &z) {
    real_ = z.real();
    imag_ = z.imag();
    return *this;
  }
  template <typename U>
  HOST_DEVICE inline Complex<T> &operator+=(const Complex<U> &z) {
    real_ += z.real();
    imag_ += z.imag();
    return *this;
  }
  template <typename U>
  HOST_DEVICE inline Complex<T> &operator-=(const Complex<U> &z) {
    real_ -= z.real();
    imag_ -= z.imag();
    return *this;
  }
  template <typename U>
  HOST_DEVICE inline Complex<T> &operator*=(const Complex<U> &z);

  // Note: check division by zero before use it.
  template <typename U>
  HOST_DEVICE inline Complex<T> &operator/=(const Complex<U> &z);

  HOST_DEVICE inline constexpr T real() const { return real_; }
  HOST_DEVICE inline constexpr T imag() const { return imag_; }
  HOST_DEVICE inline void real(T val) { real_ = val; }
  HOST_DEVICE inline void imag(T val) { imag_ = val; }

 private:
  T real_;
  T imag_;
};

template <typename T>
template <typename U>
HOST_DEVICE inline Complex<T> &Complex<T>::operator*=(const Complex<U> &z) {
  const T real = real_ * z.real() - imag_ * z.imag();
  imag_ = real_ * z.imag() + imag_ * z.real();
  real_ = real;
  return *this;
}

// Note: check division by zero before use it.
template <typename T>
template <typename U>
HOST_DEVICE inline Complex<T> &Complex<T>::operator/=(const Complex<U> &z) {
  T a = real_;
  T b = imag_;
  U c = z.real();
  U d = z.imag();
  auto denominator = c * c + d * d;
  real_ = (a * c + b * d) / denominator;
  imag_ = (b * c - a * d) / denominator;
  return *this;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator+(const Complex<T> &lhs, const Complex<T> &rhs) {
  Complex<T> result = lhs;
  result += rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator+(const Complex<T> &lhs, const T &rhs) {
  Complex<T> result = lhs;
  result += rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator+(const T &lhs, const Complex<T> &rhs) {
  Complex<T> result = rhs;
  result += lhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator-(const Complex<T> &lhs, const Complex<T> &rhs) {
  Complex<T> result = lhs;
  result -= rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator-(const Complex<T> &lhs, const T &rhs) {
  Complex<T> result = lhs;
  result -= rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator-(const T &lhs, const Complex<T> &rhs) {
  Complex<T> result(lhs, -rhs.imag());
  result -= rhs.real();
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator*(const Complex<T> &lhs, const Complex<T> &rhs) {
  Complex<T> result = lhs;
  result *= rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator*(const Complex<T> &lhs, const T &rhs) {
  Complex<T> result = lhs;
  result *= rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator*(const T &lhs, const Complex<T> &rhs) {
  Complex<T> result = rhs;
  result *= lhs;
  return result;
}

// Note: check division by zero before use it.
template <typename T>
HOST_DEVICE inline Complex<T> operator/(const Complex<T> &lhs, const Complex<T> &rhs) {
  Complex<T> result = lhs;
  result /= rhs;
  return result;
}

// Note: check division by zero before use it.
template <typename T>
HOST_DEVICE inline Complex<T> operator/(const Complex<T> &lhs, const T &rhs) {
  Complex<T> result = lhs;
  result /= rhs;
  return result;
}

// Note: check division by zero before use it.
template <typename T>
HOST_DEVICE inline Complex<T> operator/(const T &lhs, const Complex<T> &rhs) {
  Complex<T> result = lhs;
  result /= rhs;
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator+(const Complex<T> &z) {
  return z;
}

template <typename T>
HOST_DEVICE inline Complex<T> operator-(const Complex<T> &z) {
  return Complex<T>(-z.real(), -z.imag());
}

template <typename T>
HOST_DEVICE inline bool operator==(const Complex<T> &lhs, const Complex<T> &rhs) {
  return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}

template <typename T>
HOST_DEVICE inline bool operator==(const T &lhs, const Complex<T> &rhs) {
  return lhs == rhs.real() && rhs.imag() == 0;
}

template <typename T>
HOST_DEVICE inline bool operator==(const Complex<T> &lhs, const T &rhs) {
  return lhs.real() == rhs && lhs.imag() == 0;
}

template <typename T>
HOST_DEVICE inline bool operator!=(const Complex<T> &lhs, const Complex<T> &rhs) {
  return !(lhs == rhs);
}

template <typename T>
HOST_DEVICE inline bool operator!=(const T &lhs, const Complex<T> &rhs) {
  return !(lhs == rhs);
}

template <typename T>
HOST_DEVICE inline bool operator!=(const Complex<T> &lhs, const T &rhs) {
  return !(lhs == rhs);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Complex<T> &v) {
  return (os << std::noshowpos << v.real() << std::showpos << v.imag() << 'j');
}

template <typename T>
HOST_DEVICE inline Complex<T> tan(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::tan(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::tan(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> sin(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::sin(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::sin(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> cos(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::cos(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::cos(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> acos(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::acos(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::acos(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> acosh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::acosh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::acosh(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> asin(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::asin(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::asin(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> asinh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::asinh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::asinh(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> atan(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::atan(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::tan(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> atanh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::atanh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::tan(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> conj(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::conj(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::conj(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> sqrt(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::sqrt(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::sqrt(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> tanh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::tanh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::tanh(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline T abs(const Complex<T> &z) {
#if defined(__CUDACC__)
  return thrust::abs(thrust::complex<T>(z));
#else
  return std::abs(std::complex<T>(z));
#endif
}

template <typename T>
HOST_DEVICE inline Complex<T> log(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::log(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::log(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> exp(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::exp(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::exp(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> cosh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::cosh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::cosh(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline Complex<T> sinh(const Complex<T> &z) {
  Complex<T> result;
#if defined(__CUDACC__)
  auto thrust_result = thrust::sinh(thrust::complex<T>(z));
  result.real(thrust_result.real());
  result.imag(thrust_result.imag());
#else
  result(std::sinh(std::complex<T>(z)));
#endif
  return result;
}

template <typename T>
HOST_DEVICE inline bool isfinite(const Complex<T> &z) {
  return std::isfinite(z.real()) || std::isfinite(z.imag());
}
}  // namespace utils
}  // namespace mindspore

template <typename T>
using Complex = mindspore::utils::Complex<T>;
namespace std {
template <typename T>
class numeric_limits<mindspore::utils::Complex<T>> : public numeric_limits<T> {};
}  // namespace std
#endif  // MINDSPORE_CCSRC_UTILS_COPLEX_H_
