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
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/greater.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GreaterShape {
  ShapeVector x_shape;
  ShapeVector y_shape;
  ShapeVector out_shape;
};
struct GreaterType {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestGreater : public TestOps, public testing::WithParamInterface<std::tuple<GreaterShape, GreaterType>> {};

TEST_P(TestGreater, greater_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GreaterFuncImpl greater_func_impl;
  auto prim = std::make_shared<Primitive>("Greater");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);

  auto out_shape = greater_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = greater_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GreaterOpShapeTestCases = testing::ValuesIn({
  /* static */
  GreaterShape{{2, 3}, {2, 1}, {2, 3}},
  GreaterShape{{4, 4, 3, 2}, {1, 4, 3, 2}, {4, 4, 3, 2}},
  // /* dynamic shape */
  GreaterShape{{-1}, {1}, {-1}},
  GreaterShape{{1, -1}, {2, 3}, {2, 3}},
  GreaterShape{{-1, -1}, {2, 3}, {2, 3}},
  GreaterShape{{-1, -1}, {1, 3}, {-1, 3}},
  GreaterShape{{1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}},
  GreaterShape{{5, 3, -1}, {3, -1, -1, 2}, {3, 5, 3, 2}},
  GreaterShape{{2, 1, 1, -1}, {-1, -1, 1}, {2, -1, -1, -1}},
  GreaterShape{{4, 1, -1, 9}, {3, -1, 4, -1, 5, 9}, {3, -1, 4, -1, 5, 9}},
  GreaterShape{{3, -1, 4, -1, 5, 9, 3, 1}, {5, -1, -1, -1}, {3, -1, 4, -1, 5, 9, 3, -1}},
  /* dynamic rank */
  GreaterShape{{-2}, {3, -1}, {-2}},
  GreaterShape{{3, -1}, {-2}, {-2}},
  GreaterShape{{-2}, {-2}, {-2}},
});

auto GreaterOpTypeTestCases = testing::ValuesIn({
  GreaterType{kInt8, kInt8, kBool},
  GreaterType{kInt16, kInt16, kBool},
  GreaterType{kInt32, kInt32, kBool},
  GreaterType{kInt64, kInt64, kBool},
  GreaterType{kUInt8, kUInt8, kBool},
  GreaterType{kUInt16, kUInt16, kBool},
  GreaterType{kUInt32, kUInt32, kBool},
  GreaterType{kUInt64, kUInt64, kBool},
  GreaterType{kFloat16, kFloat16, kBool},
  GreaterType{kFloat32, kFloat32, kBool},
  GreaterType{kFloat64, kFloat64, kBool},
  GreaterType{kBool, kBool, kBool},
});

INSTANTIATE_TEST_CASE_P(TestGreater, TestGreater, testing::Combine(GreaterOpShapeTestCases, GreaterOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
