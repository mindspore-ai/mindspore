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
  ShapeJoinCheck({3, 5}, {3, 4}, nullptr);
  ShapeJoinCheck({3, 4}, {3, 4, 1}, nullptr);
  ShapeJoinCheck({3, -1}, {-1, -1}, {3, -1});
  ShapeJoinCheck({3, 4, -1}, {4, -1, -1}, nullptr);
  ShapeJoinCheck({3, -1}, {-1, 3}, {3, 3});
  ShapeJoinCheck({3, 4}, {-1, -1}, {3, 4});
  ShapeJoinCheck({3, 4}, {3, -1}, {3, 4});
  ShapeJoinCheck({3, -1}, {3, 4, -1}, nullptr);
  ShapeJoinCheck({3, 4}, {4, -1, 5}, nullptr);
  ShapeJoinCheck({3, 4}, {-2}, {-2});
  ShapeJoinCheck({-1, -1}, {-1, -1, -1}, {-2});
  ShapeJoinCheck({3, -1}, {3, -1, -1}, {-2});
  ShapeJoinCheck({-1, -1}, {-2}, {-2});
  ShapeJoinCheck({-2}, {-2}, {-2});
}

}  // namespace abstract
}  // namespace mindspore
