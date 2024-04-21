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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/equal.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"
#include "ops/ops_func_impl/flatten_ext.h"

namespace mindspore {
namespace ops {
class TestFlattenExt : public TestOps,
                       public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestFlattenExt, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  FlattenExtFuncImpl flatten_shape_impl;
  auto prim = std::make_shared<Primitive>("FlattenExt");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto start_dim = MakeValue<int64_t>(0)->ToAbstract();
  auto end_dim = MakeValue<int64_t>(-1)->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(dtype_param.out_type, shape_param.out_shape);

  auto out_shape = flatten_shape_impl.InferShape(prim, {input_x, start_dim, end_dim});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
  auto out_dtype = flatten_shape_impl.InferType(prim, {input_x, start_dim, end_dim});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
}

auto FlattenExtOpShapeCases = testing::ValuesIn({
  EltwiseOpShapeParams{{10, 2, 3, 4}, {240}},
  EltwiseOpShapeParams{{10}, {10}},
  EltwiseOpShapeParams{{-1, -1, -1}, {-1}},
  EltwiseOpShapeParams{{-1, 2, 3, 4}, {-1}},
  EltwiseOpShapeParams{{10, 2, 3, -1}, {-1}},
  EltwiseOpShapeParams{{-2}, {-1}},
});

auto FlattenExtOpTypeCases = testing::ValuesIn({
  EltwiseOpTypeParams{kBool, kBool},
  EltwiseOpTypeParams{kInt8, kInt8},
  EltwiseOpTypeParams{kInt16, kInt16},
  EltwiseOpTypeParams{kInt32, kInt32},
  EltwiseOpTypeParams{kInt64, kInt64},
  EltwiseOpTypeParams{kUInt8, kUInt8},
  EltwiseOpTypeParams{kUInt16, kUInt16},
  EltwiseOpTypeParams{kUInt32, kUInt32},
  EltwiseOpTypeParams{kUInt64, kUInt64},
  EltwiseOpTypeParams{kFloat16, kFloat16},
  EltwiseOpTypeParams{kFloat32, kFloat32},
  EltwiseOpTypeParams{kFloat64, kFloat64},
  EltwiseOpTypeParams{kComplex64, kComplex64},
  EltwiseOpTypeParams{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestFlattenExt, TestFlattenExt,
                        testing::Combine(FlattenExtOpShapeCases, FlattenExtOpTypeCases));
}  // namespace ops
}  // namespace mindspore
