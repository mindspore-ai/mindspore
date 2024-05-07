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
#include "ops/ops_func_impl/cumsum_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct CumsumExtShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr dim;
  ValuePtr dtype;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestCumsumExt : public TestOps, public testing::WithParamInterface<CumsumExtShapeParams> {};

TEST_P(TestCumsumExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto dim = param.dim->ToAbstract();
  auto dtype = param.dtype->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  CumsumExtFuncImpl cumsum_func_impl;
  auto prim = std::make_shared<Primitive>("CumsumExt");

  auto out_dtype = cumsum_func_impl.InferType(prim, {x, dim, dtype});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = cumsum_func_impl.InferShape(prim, {x, dim, dtype});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestCumsumExt, TestCumsumExt,
  testing::Values(CumsumExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), CreatePyInt(kNumberTypeFloat64), {3, 4, 5}, kFloat64},
                  CumsumExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), CreatePyInt(kNumberTypeFloat64), {3, 4, 5}, kFloat64},
                  CumsumExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3), CreatePyInt(kNumberTypeInt8), {3, 4, 5}, kInt8},
                  CumsumExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2), CreatePyInt(kNumberTypeInt64), {2, 3, 4, 5}, kInt64},
                  CumsumExtShapeParams{{-1, -1, -1}, kUInt8, CreateScalar<int64_t>(2), CreatePyInt(kNumberTypeInt16), {-1, -1, -1}, kInt16},
                  CumsumExtShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), CreatePyInt(kNumberTypeFloat32), {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
