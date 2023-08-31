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

#include "ops/test_ops.h"
#include "ops/ops_func_impl/gather_d.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct GatherDParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ValuePtr dim;
  ShapeVector index_shape;
  TypePtr index_dtype;
};

class TestGatherD : public TestOps, public testing::WithParamInterface<GatherDParams> {};

TEST_P(TestGatherD, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  auto dim = param.dim->ToAbstract();
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_dtype, param.index_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.index_shape);

  GatherDFuncImpl gather_d_func_impl;
  auto prim = std::make_shared<Primitive>("GatherD");

  auto out_dtype = gather_d_func_impl.InferType(prim, {x, dim, index});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = gather_d_func_impl.InferShape(prim, {x, dim, index});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto gather_d_cases = testing::Values(
  /* static */
  GatherDParams{{2, 3, 4, 5}, kFloat64, CreatePyInt(1), {2, 10, 4, 5}, kInt64},
  GatherDParams{{2, 3, 4, 5}, kFloat16, CreatePyInt(3), {2, 3, 4, 10}, kInt32},
  GatherDParams{{2, 3, 4, 5}, kFloat32, CreateScalar(kValueAny), {2, 6, 4, 5}, kInt32},
  /* -1 */
  GatherDParams{{-1, 3, -1, 5}, kFloat32, CreateScalar(0), {2, 6, 4, 5}, kInt64},
  GatherDParams{{-1, 3, -1, 5}, kFloat16, CreateScalar(kValueAny), {-1, 6, 4, 5}, kInt32},
  /* -2 */
  GatherDParams{{-2}, kFloat64, CreatePyInt(1), {2, 10, 4, 5}, kInt64},
  GatherDParams{{-2}, kFloat64, CreatePyInt(4), {-2}, kInt64},
  GatherDParams{{2, 3, 4, 5}, kFloat64, CreateScalar(2), {-2}, kInt64});
INSTANTIATE_TEST_CASE_P(TestGatherD, TestGatherD, gather_d_cases);
}  // namespace ops
}  // namespace mindspore
