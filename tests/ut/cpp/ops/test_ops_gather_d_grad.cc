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
#include "ops/ops_func_impl/gather_d_grad_v2.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct GatherDGradV2Params {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ShapeVector index_shape;
  TypePtr index_dtype;
  ValuePtr dim;
};

class TestGatherDGradV2 : public TestOps, public testing::WithParamInterface<GatherDGradV2Params> {};

TEST_P(TestGatherDGradV2, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  auto dim = param.dim->ToAbstract();
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_dtype, param.index_shape);
  auto dout = std::make_shared<abstract::AbstractTensor>(param.index_dtype, param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);

  GatherDGradV2FuncImpl gather_d_grad_v2_func_impl;
  auto prim = std::make_shared<Primitive>("GatherDGradV2");

  auto out_dtype = gather_d_grad_v2_func_impl.InferType(prim, {x, dim, index, dout});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = gather_d_grad_v2_func_impl.InferShape(prim, {x, dim, index, dout});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto gather_d_grad_v2_cases = testing::Values(
  /* static */
  GatherDGradV2Params{{2, 3, 4, 5}, kFloat64, {2, 10, 4, 5}, kInt64, CreatePyInt(1)},
  GatherDGradV2Params{{2, 3, 4, 5}, kFloat16, {2, 3, 4, 10}, kInt32, CreatePyInt(3)},
  GatherDGradV2Params{{2, 3, 4, 5}, kFloat32, {2, 6, 4, 5}, kInt32, CreateScalar(kValueAny)},
  /* -1 */
  GatherDGradV2Params{{-1, 3, -1, 5}, kFloat32, {2, 6, 4, 5}, kInt64, CreateScalar(0)},
  GatherDGradV2Params{{-1, 3, -1, 5}, kFloat16, {-1, 6, 4, 5}, kInt32, CreateScalar(kValueAny)},
  /* -2 */
  GatherDGradV2Params{{-2}, kFloat64, {2, 10, 4, 5}, kInt64, CreatePyInt(1)},
  GatherDGradV2Params{{-2}, kFloat64, {-2}, kInt64, CreatePyInt(4)},
  GatherDGradV2Params{{2, 3, 4, 5}, kFloat64, {-2}, kInt64, CreateScalar(2)});
INSTANTIATE_TEST_CASE_P(TestGatherDGradV2, TestGatherDGradV2, gather_d_grad_v2_cases);
}  // namespace ops
}  // namespace mindspore
