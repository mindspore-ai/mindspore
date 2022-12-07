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
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class TestCheckConvertUtils : public UT::Common {
 public:
  TestCheckConvertUtils() = default;
  void SetUp() {}
};

AbstractBasePtr MakeTensorAbstract(const ShapeVector &shape_vec, const TypePtr &elem_type) {
  auto shape = std::make_shared<abstract::Shape>(shape_vec);
  return std::make_shared<abstract::AbstractTensor>(elem_type, shape);
}

// Feature: CheckAbstractShapeSame.
// Description: Check function of CheckAbstractShapeSame in check_convert_utils.cc
// Expectation: Get right index of the incompatible shape.
TEST_F(TestCheckConvertUtils, TestCheckAbstractShapeSame) {
  // one abstract test
  auto abs1 = MakeTensorAbstract({1, 2, 3}, kInt32);
  auto ret = CheckAndConvertUtils::CheckAbstractShapeSame({abs1});
  ASSERT_EQ(ret, 0);
  // 3 abstracts test
  ret = CheckAndConvertUtils::CheckAbstractShapeSame({abs1, abs1, abs1});
  ASSERT_EQ(ret, 0);
  auto abs2 = MakeTensorAbstract({1, 2, 4}, kInt32);
  ret = CheckAndConvertUtils::CheckAbstractShapeSame({abs1, abs1, abs2});
  ASSERT_EQ(ret, 2);
  // Tuple shape compare with tensor shape
  auto abs3 = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({abs1}));
  ret = CheckAndConvertUtils::CheckAbstractShapeSame({abs1, abs1, abs3});
  ASSERT_EQ(ret, 2);
  // Tuple shape compare with dynamic len tuple shape.
  auto abs4 = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({abs1}));
  abs4->set_dynamic_len(true);
  abs4->set_dynamic_len_element_abs(abs1);
  ret = CheckAndConvertUtils::CheckAbstractShapeSame({abs3, abs4});
  ASSERT_EQ(ret, 1);
}

// Feature: CheckAbstractShapeSame.
// Description: Check function of TestCheckAbstractTypeSame in check_convert_utils.cc
// Expectation: Get right index of the incompatible type .
TEST_F(TestCheckConvertUtils, TestCheckAbstractTypeSame) {
  // one abstract test
  auto abs1 = MakeTensorAbstract({1, 2, 3}, kInt32);
  auto ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs1});
  ASSERT_EQ(ret, 0);
  // 3 abstracts test
  ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs1, abs1, abs1});
  ASSERT_EQ(ret, 0);
  // element type not same but type same.
  auto abs2 = MakeTensorAbstract({1, 2, 4}, kInt64);
  ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs1, abs1, abs2});
  ASSERT_EQ(ret, 2);
  // abstract type not same but element type same
  auto abs3 = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({abs1}));
  ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs1, abs1, abs3});
  ASSERT_EQ(ret, 2);
  // compare tuple type with tuple type, element not same
  auto abs4 = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({abs2}));
  ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs3, abs4});
  ASSERT_EQ(ret, 1);
  // compare tuple type with dynamic_len tuple type, element type same.
  auto abs5 = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({abs1}));
  abs5->set_dynamic_len(true);
  abs5->set_dynamic_len_element_abs(abs1);
  ret = CheckAndConvertUtils::CheckAbstractTypeSame({abs3, abs5});
  ASSERT_EQ(ret, 1);
}
}  // namespace opt
}  // namespace mindspore