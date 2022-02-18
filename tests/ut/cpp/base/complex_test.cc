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
#include <memory>

#include "common/common_test.h"
#include "utils/complex.h"

namespace mindspore {

class TestComplex : public UT::Common {
 public:
  TestComplex() {}
};

TEST_F(TestComplex, test_size) {
  ASSERT_EQ(sizeof(Complex<float>), 2 * sizeof(float));
  ASSERT_EQ(sizeof(Complex<double>), 2 * sizeof(double));
  ASSERT_EQ(alignof(Complex<float>), 2 * sizeof(float));
  ASSERT_EQ(alignof(Complex<double>), 2 * sizeof(double));
}

template <typename T>
void test_construct() {
  constexpr T real = T(1.11f);
  constexpr T imag = T(2.22f);
  ASSERT_EQ(Complex<T>().real(), T());
  ASSERT_EQ(Complex<T>().imag(), T());
  ASSERT_EQ(Complex<T>(real, imag).real(), real);
  ASSERT_EQ(Complex<T>(real, imag).imag(), imag);
  ASSERT_EQ(Complex<T>(real).real(), real);
  ASSERT_EQ(Complex<T>(real).imag(), T());
}

template <typename T1, typename T2>
void test_conver_construct() {
  ASSERT_EQ(Complex<T1>(Complex<T2>(T2(1.11f), T2(2.22f))).real(), T1(1.11f));
  ASSERT_EQ(Complex<T1>(Complex<T2>(T2(1.11f), T2(2.22f))).imag(), T1(2.22f));
}

template <typename T>
void test_conver_std_construct() {
  ASSERT_EQ(Complex<T>(std::complex<T>(T(1.11f), T(2.22f))).real(), T(1.11f));
  ASSERT_EQ(Complex<T>(std::complex<T>(T(1.11f), T(2.22f))).imag(), T(2.22f));
}

TEST_F(TestComplex, test_construct) {
  test_construct<float>();
  test_construct<double>();
  test_conver_construct<float, float>();
  test_conver_construct<double, double>();
  test_conver_construct<float, double>();
  test_conver_construct<double, float>();
  test_conver_std_construct<float>();
  test_conver_std_construct<double>();
}

template <typename T>
void test_convert_operator(T &&a) {
  ASSERT_EQ(static_cast<T>(Complex<float>(a)), a);
}

TEST_F(TestComplex, test_convert_operator) {
  test_convert_operator<bool>(true);
  test_convert_operator<signed char>(1);
  test_convert_operator<unsigned char>(1);
  ASSERT_NEAR(static_cast<double>(Complex<float>(1.11)), 1.11, 0.001);
  test_convert_operator<float>(1.11f);
  test_convert_operator<int16_t>(1);
  test_convert_operator<uint16_t>(1);
  test_convert_operator<int32_t>(1);
  test_convert_operator<uint32_t>(1);
  test_convert_operator<int64_t>(1);
  test_convert_operator<uint64_t>(1);
  float16 a(1.11f);
  ASSERT_EQ(static_cast<float16>(Complex<float>(a)), a);
}

TEST_F(TestComplex, test_assign_operator) {
  Complex<float> a = 1.11f;
  std::cout << a << std::endl;
  ASSERT_EQ(a.real(), 1.11f);
  ASSERT_EQ(a.imag(), float());
  a = Complex<double>(2.22f, 1.11f);
  ASSERT_EQ(a.real(), 2.22f);
  ASSERT_EQ(a.imag(), 1.11f);
}

template <typename T1, typename T2, typename T3>
void test_arithmetic_add(T1 lhs, T2 rhs, T3 r) {
  ASSERT_EQ(lhs + rhs, r);
  if constexpr (!(std::is_same<T1, float>::value || std::is_same<T1, double>::value)) {
    ASSERT_EQ(lhs += rhs, r);
  }
}
template <typename T1, typename T2, typename T3>
void test_arithmetic_sub(T1 lhs, T2 rhs, T3 r) {
  ASSERT_EQ(lhs - rhs, r);
  if constexpr (!(std::is_same<T1, float>::value || std::is_same<T1, double>::value)) {
    ASSERT_EQ(lhs -= rhs, r);
  }
}
template <typename T1, typename T2, typename T3>
void test_arithmetic_mul(T1 lhs, T2 rhs, T3 r) {
  ASSERT_EQ(lhs * rhs, r);
  if constexpr (!(std::is_same<T1, float>::value || std::is_same<T1, double>::value)) {
    ASSERT_EQ(lhs *= rhs, r);
  }
}
template <typename T1, typename T2, typename T3>
void test_arithmetic_div(T1 lhs, T2 rhs, T3 r) {
  ASSERT_EQ(lhs / rhs, r);
  if constexpr (!(std::is_same<T1, float>::value || std::is_same<T1, double>::value)) {
    ASSERT_EQ(lhs /= rhs, r);
  }
}

TEST_F(TestComplex, test_arithmetic) {
  test_arithmetic_add<Complex<float>, Complex<float>, Complex<float>>(
    Complex<float>(1.11, 2.22), Complex<float>(1.11, 2.22), Complex<float>(2.22, 4.44));
  test_arithmetic_add<Complex<float>, float, Complex<float>>(Complex<float>(1.11, 2.22), 1.11,
                                                             Complex<float>(2.22, 2.22));
  test_arithmetic_add<float, Complex<float>, Complex<float>>(1.11, Complex<float>(1.11, 2.22),
                                                             Complex<float>(2.22, 2.22));

  test_arithmetic_sub<Complex<float>, Complex<float>, Complex<float>>(Complex<float>(1.11, 2.22),
                                                                      Complex<float>(1.11, 2.22), Complex<float>(0, 0));
  test_arithmetic_sub<Complex<float>, float, Complex<float>>(Complex<float>(1.11, 2.22), 1.11, Complex<float>(0, 2.22));
  test_arithmetic_sub<float, Complex<float>, Complex<float>>(1.11, Complex<float>(1.11, 2.22),
                                                             Complex<float>(0, -2.22));

  test_arithmetic_mul<Complex<float>, Complex<float>, Complex<float>>(
    Complex<float>(1.11, 2.22), Complex<float>(1.11, 2.22), Complex<float>(-3.6963, 4.9284));
  test_arithmetic_mul<Complex<float>, float, Complex<float>>(Complex<float>(1.11, 2.22), 1.11,
                                                             Complex<float>(1.2321, 2.4642));
  test_arithmetic_mul<float, Complex<float>, Complex<float>>(1.11, Complex<float>(1.11, 2.22),
                                                             Complex<float>(1.2321, 2.4642));

  test_arithmetic_div<Complex<float>, Complex<float>, Complex<float>>(Complex<float>(1.11, 2.22),
                                                                      Complex<float>(1.11, 2.22), Complex<float>(1, 0));
  test_arithmetic_div<Complex<float>, float, Complex<float>>(Complex<float>(1.11, 2.22), 1.11, Complex<float>(1, 2));
  test_arithmetic_div<float, Complex<float>, Complex<float>>(1.11, Complex<float>(1.11, 2.22),
                                                             Complex<float>(0.2, -0.4));
}

}  // namespace mindspore
