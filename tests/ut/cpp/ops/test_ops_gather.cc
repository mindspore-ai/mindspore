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
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/gather.h"

namespace mindspore {
namespace ops {
struct GatherParams {
  ShapeVector input_params_shape;
  ShapeVector input_indices_shape;
  TypePtr input_indices_dtype;
  ValuePtr axis;
  ValuePtr batch_dims;
  TypePtr dtype;
  ShapeVector output_shape;
};

class TestGather : public TestOps, public testing::WithParamInterface<GatherParams> {};

TEST_P(TestGather, dyn_shape) {
  const auto &param = GetParam();
  auto input_params = std::make_shared<abstract::AbstractTensor>(param.dtype, param.input_params_shape);
  auto input_indices = std::make_shared<abstract::AbstractTensor>(param.input_indices_dtype, param.input_indices_shape);
  auto axis = param.axis->ToAbstract();
  auto batch_dims = param.batch_dims->ToAbstract();

  GatherFuncImpl gather_func_impl;
  auto prim = std::make_shared<Primitive>("Gather");
  auto expect = std::make_shared<abstract::AbstractTensor>(param.dtype, param.output_shape);

  auto out_dtype = gather_func_impl.InferType(prim, {input_params, input_indices, axis, batch_dims});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = gather_func_impl.InferShape(prim, {input_params, input_indices, axis, batch_dims});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto test_cases = testing::Values(
  /* input_params input_indices know */
  GatherParams{{5, 6, 7, 8, 9}, {10, 20}, kInt32, CreatePyInt(1), CreatePyInt(0), kFloat32, {5, 10, 20, 7, 8, 9}},
  GatherParams{{5, 6, 7, 8, 9}, {5, 10, 20}, kInt64, CreatePyInt(1), CreatePyInt(1), kInt64, {5, 10, 20, 7, 8, 9}},
  GatherParams{{5, 6, 7, 8, 9}, {5, 6, 10, 20}, kInt32, CreatePyInt(3), CreatePyInt(2), kFloat32, {5, 6, 7, 10, 20, 9}},
  GatherParams{{5, 6}, {10, 20}, kInt32, CreateScalar(kValueAny), CreateScalar(kValueAny), kInt32, {-1, -1, -1}},
  GatherParams{{5, 6, 7, 8}, {10, 20}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {5, 10, 20, 7, 8}},
  GatherParams{{5, 6, 7, 8, 9}, {5, 20}, kInt32, CreateScalar(kValueAny), CreateScalar(kValueAny), kFloat32, {-2}},
  /* input_params dyn */
  GatherParams{{5, -1, 7}, {10, 20}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {5, 10, 20, 7}},
  GatherParams{{-1, 6, 7}, {10, 20}, kInt32, CreatePyInt(1), CreatePyInt(0), kFloat32, {-1, 10, 20, 7}},
  GatherParams{{-1, 6, 7}, {10, 20}, kInt32, CreateScalar(kValueAny), CreatePyInt(1), kFloat32, {-1, -1, -1}},
  GatherParams{{-2}, {10, 20}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {-2}},
  /* input_indices dyn */
  GatherParams{{5, 6, 7}, {10, -1}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {5, 10, -1, 7}},
  GatherParams{{5, 6, 7}, {-1, 20}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {-2}},
  GatherParams{{5, 6, 7}, {-1, 20}, kInt32, CreateScalar(kValueAny), CreatePyInt(-2), kFloat32, {-1, -1, -1}},
  GatherParams{{5, 6, 7}, {-2}, kInt32, CreatePyInt(2), CreatePyInt(1), kFloat32, {-2}},
  GatherParams{{5, 6, 7}, {-1, -1}, kInt32, CreatePyInt(1), CreateScalar(kValueAny), kFloat32, {-2}},
  /* both dyn */
  GatherParams{{5, 6, -1}, {-1, 20}, kInt32, CreateScalar(kValueAny), CreatePyInt(1), kFloat32, {-1, -1, -1}},
  GatherParams{{5, 6, 7}, {-2}, kInt32, CreatePyInt(2), CreatePyInt(1), kFloat32, {-2}},
  GatherParams{{-2}, {-2}, kInt32, CreatePyInt(1), CreatePyInt(-1), kFloat32, {-2}});

INSTANTIATE_TEST_CASE_P(TestGather, TestGather, test_cases);
}  // namespace ops
}  // namespace mindspore
