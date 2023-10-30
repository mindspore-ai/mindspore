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
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/cumprod.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct CumProdShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ValuePtr exclusive;
  ValuePtr reverse;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestCumProd : public TestOps, public testing::WithParamInterface<CumProdShapeParams> {};

TEST_P(TestCumProd, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis->ToAbstract();
  auto exclusive = param.exclusive->ToAbstract();
  auto reverse = param.reverse->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  CumProdFuncImpl cumprod_func_impl;
  auto prim = std::make_shared<Primitive>("CumProd");

  auto out_dtype = cumprod_func_impl.InferType(prim, {x, axis, exclusive, reverse});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = cumprod_func_impl.InferShape(prim, {x, axis, exclusive, reverse});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestCumProd, TestCumProd,
  testing::Values(CumProdShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<bool>(true), CreateScalar<bool>(true), {3, 4, 5}, kFloat32},
                  CumProdShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), CreateScalar<bool>(true), CreateScalar<bool>(false), {3, 4, 5}, kInt64},
                  CumProdShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3), CreateScalar<bool>(true), CreateScalar<bool>(true), {3, 4, 5}, kInt64},
                  CumProdShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2), CreateScalar<bool>(true), CreateScalar<bool>(false), {2, 3, 4, 5}, kInt32},
                  CumProdShapeParams{{-1, -1, -1}, kUInt64, CreateScalar<int64_t>(2), CreateScalar<bool>(false), CreateScalar<bool>(false), {-1, -1, -1}, kUInt64},
                  CumProdShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<bool>(true), CreateScalar<bool>(true), {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
