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
#include "ops/ops_func_impl/greater_equal.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GreaterEqualShape {
  ShapeVector x_shape;
  ShapeVector y_shape;
  ShapeVector out_shape;
};
struct GreaterEqualType {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestGreaterEqual : public TestOps,
                         public testing::WithParamInterface<std::tuple<GreaterEqualShape, GreaterEqualType>> {};

TEST_P(TestGreaterEqual, greater_equal_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GreaterEqualFuncImpl greater_equal_func_impl;
  auto prim = std::make_shared<Primitive>("GreaterEqual");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);

  auto out_shape = greater_equal_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = greater_equal_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GreaterEqualOpShapeTestCases = testing::ValuesIn({
  /* static */
  GreaterEqualShape{{2, 3}, {2, 1}, {2, 3}},
  GreaterEqualShape{{4, 4, 3, 2}, {1, 4, 3, 2}, {4, 4, 3, 2}},
  // /* dynamic shape */
  GreaterEqualShape{{-1}, {1}, {-1}},
  GreaterEqualShape{{1, -1}, {2, 3}, {2, 3}},
  GreaterEqualShape{{-1, -1}, {2, 3}, {2, 3}},
  GreaterEqualShape{{-1, -1}, {1, 3}, {-1, 3}},
  GreaterEqualShape{{1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}},
  GreaterEqualShape{{5, 3, -1}, {3, -1, -1, 2}, {3, 5, 3, 2}},
  GreaterEqualShape{{2, 1, 1, -1}, {-1, -1, 1}, {2, -1, -1, -1}},
  GreaterEqualShape{{4, 1, -1, 9}, {3, -1, 4, -1, 5, 9}, {3, -1, 4, -1, 5, 9}},
  GreaterEqualShape{{3, -1, 4, -1, 5, 9, 3, 1}, {5, -1, -1, -1}, {3, -1, 4, -1, 5, 9, 3, -1}},
  /* dynamic rank */
  GreaterEqualShape{{-2}, {3, -1}, {-2}},
  GreaterEqualShape{{3, -1}, {-2}, {-2}},
  GreaterEqualShape{{-2}, {-2}, {-2}},
});

auto GreaterEqualOpTypeTestCases = testing::ValuesIn({
  GreaterEqualType{kInt8, kInt8, kBool},
  GreaterEqualType{kInt16, kInt16, kBool},
  GreaterEqualType{kInt32, kInt32, kBool},
  GreaterEqualType{kInt64, kInt64, kBool},
  GreaterEqualType{kUInt8, kUInt8, kBool},
  GreaterEqualType{kUInt16, kUInt16, kBool},
  GreaterEqualType{kUInt32, kUInt32, kBool},
  GreaterEqualType{kUInt64, kUInt64, kBool},
  GreaterEqualType{kFloat16, kFloat16, kBool},
  GreaterEqualType{kFloat32, kFloat32, kBool},
  GreaterEqualType{kFloat64, kFloat64, kBool},
  GreaterEqualType{kBool, kBool, kBool},
});

INSTANTIATE_TEST_CASE_P(TestGreaterEqual, TestGreaterEqual,
                        testing::Combine(GreaterEqualOpShapeTestCases, GreaterEqualOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
