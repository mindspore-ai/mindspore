/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/gcd.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GcdShape {
  ShapeVector x1_shape;
  ShapeVector x2_shape;
  ShapeVector out_shape;
};
struct GcdType {
  TypePtr x1_type;
  TypePtr x2_type;
  TypePtr out_type;
};

class TestGcd : public TestOps, public testing::WithParamInterface<std::tuple<GcdShape, GcdType>> {};

TEST_P(TestGcd, gcd_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GcdFuncImpl gcd_func_impl;
  auto prim = std::make_shared<Primitive>("Gcd");
  auto x1 = std::make_shared<abstract::AbstractTensor>(dtype_param.x1_type, shape_param.x1_shape);
  auto x2 = std::make_shared<abstract::AbstractTensor>(dtype_param.x2_type, shape_param.x2_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);
  ASSERT_NE(x1, nullptr);
  ASSERT_NE(x2, nullptr);

  auto out_shape = gcd_func_impl.InferShape(prim, {x1, x2});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = gcd_func_impl.InferType(prim, {x1, x2});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GcdOpShapeTestCases = testing::ValuesIn({
  /* static */
  GcdShape{{2, 3}, {2, 1}, {2, 3}},
  GcdShape{{4, 4, 3, 2}, {1, 4, 3, 2}, {4, 4, 3, 2}},
  // /* dynamic shape */
  GcdShape{{-1}, {1}, {-1}},
  GcdShape{{1, -1}, {2, 3}, {2, 3}},
  GcdShape{{-1, -1}, {2, 3}, {2, 3}},
  GcdShape{{-1, -1}, {1, 3}, {-1, 3}},
  GcdShape{{1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}},
  GcdShape{{5, 3, -1}, {3, -1, -1, 2}, {3, 5, 3, 2}},
  GcdShape{{2, 1, 1, -1}, {-1, -1, 1}, {2, -1, -1, -1}},
  GcdShape{{4, 1, -1, 9}, {3, -1, 4, -1, 5, 9}, {3, -1, 4, -1, 5, 9}},
  GcdShape{{3, -1, 4, -1, 5, 9, 3, 1}, {5, -1, -1, -1}, {3, -1, 4, -1, 5, 9, 3, -1}},
  /* dynamic rank */
  GcdShape{{-2}, {3, -1}, {-2}},
  GcdShape{{3, -1}, {-2}, {-2}},
  GcdShape{{-2}, {-2}, {-2}},
});

auto GcdOpTypeTestCases = testing::ValuesIn({
  GcdType{kInt32, kInt32, kInt32},
  GcdType{kInt64, kInt64, kInt64},
});

INSTANTIATE_TEST_CASE_P(TestGcd, TestGcd, testing::Combine(GcdOpShapeTestCases, GcdOpTypeTestCases));

}  // namespace ops
}  // namespace mindspore
