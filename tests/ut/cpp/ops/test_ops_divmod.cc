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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/divmod.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct DivModShape {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> y_shape;
  ValuePtr rounding_mode;
  std::vector<int64_t> out_shape;
};

struct DivModType {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestDivMod : public TestOps, public testing::WithParamInterface<std::tuple<DivModShape, DivModType>> {};

TEST_P(TestDivMod, DivMod_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  DivModFuncImpl DivMod_func_impl;
  auto prim = std::make_shared<Primitive>("DivMod");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = DivMod_func_impl.InferShape(prim, {x, y, shape_param.rounding_mode->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = DivMod_func_impl.InferType(prim, {x, y, shape_param.rounding_mode->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto DivModOpShapeTestCases = testing::ValuesIn({
    /* y is number */
    DivModShape{{10}, {}, CreateScalar<int64_t>(2), {10}},
    DivModShape{{10, 1, 2}, {}, CreateScalar<int64_t>(2), {10, 1, 2}},
    DivModShape{{10, 4, 2}, {}, CreateScalar<int64_t>(2), {10, 4, 2}},
    DivModShape{{10, 1, -1}, {}, CreateScalar<int64_t>(2), {10, 1, -1}},
    DivModShape{{-2}, {}, CreateScalar<int64_t>(2), {-2}},
    /* x is number */
    DivModShape{{}, {10}, CreateScalar<int64_t>(2), {10}},
    DivModShape{{}, {10, 1, 2}, CreateScalar<int64_t>(2), {10, 1, 2}},
    DivModShape{{}, {10, 4, 2}, CreateScalar<int64_t>(2), {10, 4, 2}},
    DivModShape{{}, {10, 1, -1}, CreateScalar<int64_t>(2), {10, 1, -1}},
    DivModShape{{}, {-2}, CreateScalar<int64_t>(2), {-2}},
    /* x and y both tensor */
    DivModShape{{4, 5}, {2, 3, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    DivModShape{{2, 1, 4, 5, 6, 9}, {9}, CreateScalar<int64_t>(2), {2, 1, 4, 5, 6, 9}},
    DivModShape{{2, 3, 4, -1}, {2, 3, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    DivModShape{{2, 3, 4, -1}, {-1, -1, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    DivModShape{{2, 1, 4, -1}, {-1, -1, 4, 5}, CreateScalar<int64_t>(2), {2, -1, 4, 5}},
    DivModShape{{2, 1, 4, 5, 6, 9}, {-2}, CreateScalar<int64_t>(2), {-2}},
    DivModShape{{2, 1, 4, 5, -1, 9}, {-2}, CreateScalar<int64_t>(2), {-2}},
    DivModShape{{-2}, {2, 1, 4, 5, 6, 9}, CreateScalar<int64_t>(2), {-2}},
    DivModShape{{-2}, {2, 1, 4, 5, -1, 9}, CreateScalar<int64_t>(2), {-2}},
    DivModShape{{-2}, {-2}, CreateScalar<int64_t>(2), {-2}}
});

auto DivModOpTypeTestCases = testing::ValuesIn({
  DivModType{kFloat16, kFloat16, kFloat16},
  DivModType{kFloat32, kFloat32, kFloat32},
  DivModType{kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestDivMod, TestDivMod, testing::Combine(DivModOpShapeTestCases, DivModOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
