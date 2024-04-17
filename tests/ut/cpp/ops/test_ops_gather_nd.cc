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
#include "ops/test_ops.h"
#include "ops/ops_func_impl/gather_nd.h"

namespace mindspore {
namespace ops {
struct GatherNdParams {
  ShapeVector input_x_shape;
  ShapeVector indices_shape;
  TypePtr dtype;
  TypePtr indices_dtype;
  ShapeVector output_shape;
};

class TestGatherNd : public TestOps, public testing::WithParamInterface<GatherNdParams> {};

TEST_P(TestGatherNd, dyn_shape) {
  const auto &param = GetParam();
  GatherNdFuncImpl gather_shape_impl;
  auto prim = std::make_shared<Primitive>("GatherNd");

  auto input_x = std::make_shared<abstract::AbstractTensor>(param.dtype, param.input_x_shape);
  auto indices = std::make_shared<abstract::AbstractTensor>(param.indices_dtype, param.indices_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.dtype, param.output_shape);

  auto out_shape = gather_shape_impl.InferShape(prim, {input_x, indices});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
  auto out_dtype = gather_shape_impl.InferType(prim, {input_x, indices});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
}

auto gather_nd_cases = testing::Values(
  /*static*/
  GatherNdParams{{5, 6, 7, 8, 9, 1, 1, 9}, {2, 3, 4}, kFloat32, kInt32, {2, 3, 9, 1, 1, 9}},
  GatherNdParams{{2, 3, 4, 5}, {}, kFloat32, kInt32, {3, 4, 5}},
  /* indices -1 */
  GatherNdParams{{5, 6, 7, 8, 9, 1, 1, 9}, {-1, -1, 4}, kFloat32, kInt32, {-1, -1, 9, 1, 1, 9}},
  GatherNdParams{{5, 6, 7, 8, 9, 1, 1, 9}, {-1, -1, -1}, kFloat32, kInt32, {-2}},
  GatherNdParams{{5, 6, 7, 8, 9, 1, 1, 9}, {2, 3, -1}, kFloat32, kInt32, {-2}},
  /* input_x -1 */
  GatherNdParams{{-1, -1, -1, -1, -1, -1}, {2, 3, 4}, kFloat32, kInt64, {2, 3, -1, -1}},
  GatherNdParams{{-1, -1, -1, 8, 9, 1, -1, 9}, {-1, -1, 4}, kFloat32, kInt64, {-1, -1, 9, 1, -1, 9}},
  GatherNdParams{{-1, -1, -1, -1, -1, -1}, {-1, -1, -1}, kFloat32, kInt64, {-2}},
  /* -2 */
  GatherNdParams{{5, 6, 7, 8, 9, 1, 1, 9}, {-2}, kComplex128, kInt64, {-2}},
  GatherNdParams{{-2}, {2, 3, 4, 5, 6}, kComplex64, kInt64, {-2}});

INSTANTIATE_TEST_CASE_P(TestGatherNd, TestGatherNd, gather_nd_cases);
}  // namespace ops
}  // namespace mindspore
