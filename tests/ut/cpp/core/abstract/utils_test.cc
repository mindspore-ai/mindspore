/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "abstract/utils.h"

#include "common/common_test.h"
#include "pipeline/jit/static_analysis/static_analysis.h"

namespace mindspore {
namespace abstract {
class TestUtils : public UT::Common {
 public:
  TestUtils() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(TestUtils, test_join) {
  // AbstractScalar
  AbstractBasePtr abs_s1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr abs_s2 = FromValue(static_cast<int64_t>(2), false);
  AbstractBasePtr abs_s_anything = FromValue(static_cast<int64_t>(2), true);
  abs_s_anything->set_value(kAnyValue);

  AbstractBasePtr res_s1 = abs_s1->Join(abs_s2);
  ASSERT_EQ(*res_s1, *abs_s_anything);

  abs_s1 = FromValue(static_cast<int64_t>(1), false);

  AbstractBasePtr t1 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s1, abs_s_anything}));
  AbstractBasePtr t2 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s1, abs_s_anything}));
  AbstractBasePtr t3 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s_anything, abs_s_anything}));

  AbstractBasePtr res_t1 = t1->Join(t2);
  ASSERT_EQ(res_t1, t1);

  res_t1 = t1->Join(t3);
  ASSERT_EQ(*res_t1, *t3);

  res_t1 = t3->Join(t1);
  ASSERT_EQ(res_t1, t3);
}

void ShapeJoinCheck(const ShapeVector &shape_vec1, const ShapeVector &shape_vec2, const ShapeVector &expect_shape) {
  auto shape1 = std::make_shared<Shape>(shape_vec1);
  auto shape2 = std::make_shared<Shape>(shape_vec2);
  auto joined_shape = ShapeJoin(shape1, shape2);
  if (joined_shape == nullptr || joined_shape->shape() != expect_shape) {
    std::cout << "Join shape1:" << mindspore::ToString(shape_vec1) << " with shape2:" << mindspore::ToString(shape_vec2)
              << std::endl;
    auto joined_shape_str = joined_shape == nullptr ? "Null" : mindspore::ToString(joined_shape->shape());
    auto expect_shape_str = mindspore::ToString(expect_shape);
    std::cout << "Joined shape:" << joined_shape_str << ", expect shape:" << expect_shape_str;
  }
  ASSERT_NE(joined_shape, nullptr);
  ASSERT_EQ(joined_shape->shape(), expect_shape);
}

void ShapeJoinCheck(const ShapeVector &shape_vec1, const ShapeVector &shape_vec2, const ShapePtr &expect_shape) {
  auto shape1 = std::make_shared<Shape>(shape_vec1);
  auto shape2 = std::make_shared<Shape>(shape_vec2);
  auto joined_shape = ShapeJoin(shape1, shape2);
  if (joined_shape != expect_shape) {
    std::cout << "Join shape1:" << mindspore::ToString(shape_vec1) << " with shape2:" << mindspore::ToString(shape_vec2)
              << std::endl;
    auto joined_shape_str = joined_shape == nullptr ? "Null" : mindspore::ToString(joined_shape->shape());
    auto expect_shape_str = expect_shape == nullptr ? "Null" : mindspore::ToString(expect_shape->shape());
    std::cout << "Joined shape:" << joined_shape_str << ", expect shape:" << expect_shape_str;
  }
  ASSERT_EQ(joined_shape, expect_shape);
}

// Feature: Shape join.
// Description: Shape join test.
// Expectation: Joined shape is expect shape or nullptr.
TEST_F(TestUtils, TestShapeJoin) {
  ShapeJoinCheck({3, 4}, {3, 4}, {3, 4});
  ShapeJoinCheck({3, 5}, {3, 4}, {3, -1});
  ShapeJoinCheck({3, 4}, {3, 4, 1}, {-2});
  ShapeJoinCheck({3, -1}, {-1, -1}, {-1, -1});
  ShapeJoinCheck({3, 4, -1}, {4, -1, -1}, {-1, -1, -1});
  ShapeJoinCheck({3, -1}, {-1, 3}, {-1, -1});
  ShapeJoinCheck({3, 4}, {-1, -1}, {-1, -1});
  ShapeJoinCheck({3, 4}, {3, -1}, {3, -1});
  ShapeJoinCheck({3, -1}, {3, 4, -1}, {-2});
  ShapeJoinCheck({3, 4}, {4, -1, 5}, {-2});
  ShapeJoinCheck({3, 4}, {-2}, {-2});
  ShapeJoinCheck({-1, -1}, {-1, -1, -1}, {-2});
  ShapeJoinCheck({3, -1}, {3, -1, -1}, {-2});
  ShapeJoinCheck({-1, -1}, {-2}, {-2});
  ShapeJoinCheck({-2}, {-2}, {-2});
}

AbstractBasePtr MakeTensorAbstract(const ShapeVector &shape_vec, const TypePtr &elem_type) {
  auto shape = std::make_shared<abstract::Shape>(shape_vec);
  return std::make_shared<abstract::AbstractTensor>(elem_type, shape);
}

// Feature: AbstractBroaden.
// Description: Check function of AbstractBroaden in utils.cc
// Expectation: Scalar can be successfully broadened.
TEST_F(TestUtils, CheckScalarBroaden) {
  auto scalar_abs1 = std::make_shared<abstract::AbstractScalar>(1);
  scalar_abs1->set_is_variable(true);
  auto scalar_abs1_broaden = scalar_abs1->Broaden();

  auto scalar_abs2 = std::make_shared<abstract::AbstractScalar>(2);
  auto scalar_abs2_broaden = abstract::AbstractBroaden(scalar_abs2);
  ASSERT_TRUE(*scalar_abs1_broaden == *scalar_abs2_broaden);
}

// Feature: AbstractBroaden.
// Description: Check function of AbstractBroaden in utils.cc
// Expectation: Tensor in dynamic sequence can be successfully broadened.
TEST_F(TestUtils, CheckDynSequenceBroaden) {
  // Test tensor as element abs
  auto sequence_abs = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({}));
  auto element_abs = MakeTensorAbstract({1, 2, 3}, kFloat32);
  auto element_broaden = element_abs->Broaden();
  sequence_abs->set_dynamic_len(true);
  sequence_abs->set_dynamic_len_element_abs(element_abs);
  auto broadened_sequence_abs = abstract::AbstractBroaden(sequence_abs)->cast<abstract::AbstractSequencePtr>();
  ASSERT_TRUE(broadened_sequence_abs != nullptr);
  auto equal = *element_broaden == *broadened_sequence_abs->dynamic_len_element_abs();
  ASSERT_TRUE(equal);
  // Test scalar as element abs
  sequence_abs = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({}));
  element_abs = std::make_shared<abstract::AbstractScalar>(1);
  auto scalar_abs = std::make_shared<abstract::AbstractScalar>(2);
  auto scalar_broaden = abstract::AbstractBroaden(scalar_abs);
  sequence_abs->set_dynamic_len(true);
  sequence_abs->set_dynamic_len_element_abs(element_abs);
  broadened_sequence_abs = abstract::AbstractBroaden(sequence_abs)->cast<abstract::AbstractSequencePtr>();
  ASSERT_TRUE(broadened_sequence_abs != nullptr);
  equal = *scalar_broaden == *broadened_sequence_abs->dynamic_len_element_abs();
  ASSERT_TRUE(equal);
}

// Feature: AbstractBroaden.
// Description: Check function of AbstractBroaden in utils.cc
// Expectation: Scalar in tuple can be successfully broadened.
TEST_F(TestUtils, CheckScalarInTupleBroaden) {
  auto element_abs = std::make_shared<abstract::AbstractScalar>(1);
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(std::vector<AbstractBasePtr>({element_abs}));
  auto broadened_tuple_abs = abstract::AbstractBroaden(tuple_abs)->cast<abstract::AbstractTuplePtr>();
  ASSERT_TRUE(broadened_tuple_abs != nullptr);
  ASSERT_TRUE(broadened_tuple_abs->size() == 1);

  auto scalar_abs = std::make_shared<abstract::AbstractScalar>(2);
  auto broadened_scalar_abs = abstract::AbstractBroaden(scalar_abs);

  ASSERT_TRUE(*(broadened_tuple_abs->elements()[0]) == *broadened_scalar_abs);
}

// Feature: AbstractBroaden.
// Description: Check function of AbstractBroaden in utils.cc
// Expectation: Scalar in tuple can be successfully broadened.
TEST_F(TestUtils, CheckScalarInListBroaden) {
  auto element_abs = std::make_shared<abstract::AbstractScalar>(1);
  auto list_abs = std::make_shared<abstract::AbstractList>(std::vector<AbstractBasePtr>({element_abs}));
  auto broadened_list_abs = abstract::AbstractBroaden(list_abs)->cast<abstract::AbstractListPtr>();
  ASSERT_TRUE(broadened_list_abs != nullptr);
  ASSERT_TRUE(broadened_list_abs->size() == 1);

  auto scalar_abs = std::make_shared<abstract::AbstractScalar>(2);
  auto broadened_scalar_abs = abstract::AbstractBroaden(scalar_abs);

  ASSERT_TRUE(*(broadened_list_abs->elements()[0]) == *broadened_scalar_abs);
}
}  // namespace abstract
}  // namespace mindspore
