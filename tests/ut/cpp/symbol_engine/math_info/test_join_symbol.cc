/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "symbol_engine/math_info/symbol_test_utils.h"
#include "ops/symbol_ops_impl/switch.h"

namespace mindspore::symshape::test {
/// Feature: JoinIntSymbol
/// Description: union symbol item of two same const symbol
/// Expectation: the output should be equal to the first input when two inputs are equal
TEST_F(TestMathInfo, join_symbol_equal_const) {
  auto s1 = IntSymbol::Make(16);
  auto s2 = IntSymbol::Make(16);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  // directly returns s1
  EXPECT_EQ(ret, s1);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of two different const symbol
/// Expectation: the divisor of output symbol is gcd of inputs
TEST_F(TestMathInfo, join_symbol_diff_const) {
  auto s1 = IntSymbol::Make(24);
  auto s2 = IntSymbol::Make(32);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret->as<IntSymbol>()->divisor(), 8);
  EXPECT_EQ(ret->as<IntSymbol>()->remainder(), 0);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of a const and a variable symbol
/// Expectation: the divisor of output symbol is gcd of inputs
TEST_F(TestMathInfo, join_symbol_const_var_1) {
  auto s1 = IntSymbol::Make(16);
  auto s2 = IntSymbol::Make();
  s2->SetDivisorRemainder(32, 0);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret->as<IntSymbol>()->divisor(), 16);
  EXPECT_EQ(ret->as<IntSymbol>()->remainder(), 0);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of a const and a variable symbol
/// Expectation: the output equals to variable symbol
TEST_F(TestMathInfo, join_symbol_const_var_2) {
  auto s1 = IntSymbol::Make(16);
  auto s2 = IntSymbol::Make();
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret, s2);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of two variable symbol
/// Expectation: the output equals to s1
TEST_F(TestMathInfo, join_symbol_var_var_1) {
  auto s1 = IntSymbol::Make();
  auto s2 = IntSymbol::Make();
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret, s1);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of two variable symbol
/// Expectation: the output equals to s1
TEST_F(TestMathInfo, join_symbol_var_var_2) {
  auto s1 = IntSymbol::Make();
  s1->SetDivisorRemainder(16, 2);
  auto s2 = IntSymbol::Make();
  s2->SetDivisorRemainder(32, 2);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret, s1);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of two variable symbol
/// Expectation: the output divisor is gcd of inputs
TEST_F(TestMathInfo, join_symbol_var_var_3) {
  auto s1 = IntSymbol::Make();
  s1->SetDivisorRemainder(48, 2);
  auto s2 = IntSymbol::Make();
  s2->SetDivisorRemainder(32, 2);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret->as<IntSymbol>()->divisor(), 16);
  EXPECT_EQ(ret->as<IntSymbol>()->remainder(), 2);
}

/// Feature: JoinIntSymbol
/// Description: union symbol item of two variable symbol
/// Expectation: the output divisor is gcd of inputs
TEST_F(TestMathInfo, join_symbol_var_var_4) {
  auto s1 = IntSymbol::Make();
  s1->SetDivisorRemainder(48, 2);
  auto s2 = IntSymbol::Make();
  s2->SetDivisorRemainder(32, 3);
  auto ret = symshape::ops::JoinIntSymbol(s1, s2);
  UT_CHECK_NULL(ret);
  EXPECT_EQ(ret->as<IntSymbol>()->divisor(), 1);
  EXPECT_EQ(ret->as<IntSymbol>()->remainder(), 0);
}
}  // namespace mindspore::symshape::test
