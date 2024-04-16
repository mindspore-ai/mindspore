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
#include "ops/ops_func_impl/triu.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct TriuShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr diagonal;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestTriu : public TestOps, public testing::WithParamInterface<TriuShapeParams> {};

TEST_P(TestTriu, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto diagonal = param.diagonal->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  TriuFuncImpl triu_func_impl;
  auto prim = std::make_shared<Primitive>("Triu");
  auto out_dtype = triu_func_impl.InferType(prim, {x, diagonal});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = triu_func_impl.InferShape(prim, {x, diagonal});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestTriu, TestTriu,
  testing::Values(TriuShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kFloat32},
                  TriuShapeParams{{3, 4, 5}, kUInt8, CreateScalar<int64_t>(0), {3, 4, 5}, kUInt8},
                  TriuShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-2), {3, 4, 5}, kInt64},
                  TriuShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(2), {2, 3, 4, 5}, kInt32},
                  TriuShapeParams{{-1, -1, -1}, kBool, CreateScalar<int64_t>(1), {-1, -1, -1}, kBool},
                  TriuShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
