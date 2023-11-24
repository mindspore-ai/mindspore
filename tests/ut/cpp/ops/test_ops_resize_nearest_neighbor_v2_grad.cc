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
#include "ops/ops_func_impl/resize_nearest_neighbor_v2_grad.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct ResizeNearestNeighborV2GradParams {
  ShapeVector grads_shape;
  TypePtr grads_type;
  ValuePtr size;                // tuple[int*2]
  ValuePtr align_corners;       // bool
  ValuePtr half_pixel_centers;  // bool
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestResizeNearestNeighborV2Grad : public TestOps,
                                        public testing::WithParamInterface<ResizeNearestNeighborV2GradParams> {};

TEST_P(TestResizeNearestNeighborV2Grad, dyn_shape) {
  const auto &param = GetParam();
  auto grads = std::make_shared<abstract::AbstractTensor>(param.grads_type, param.grads_shape);
  ASSERT_NE(grads, nullptr);
  auto size = param.size->ToAbstract();
  ASSERT_NE(size, nullptr);
  auto align_corners = param.align_corners->ToAbstract();
  ASSERT_NE(align_corners, nullptr);
  auto half_pixel_centers = param.half_pixel_centers->ToAbstract();
  ASSERT_NE(half_pixel_centers, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ResizeNearestNeighborV2GradFuncImpl>(
    kNameResizeNearestNeighborV2Grad, {grads, size, align_corners, half_pixel_centers}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestResizeNearestNeighborV2GradBicubicGroup, TestResizeNearestNeighborV2Grad,
  testing::Values(
    ResizeNearestNeighborV2GradParams{
      {1, 3, 8, 8}, kFloat32, CreatePyIntTuple({4, 4}), CreateScalar(true), CreateScalar(true), {1, 3, 4, 4}, kFloat32},
    ResizeNearestNeighborV2GradParams{{1, 3, 8, 8},
                                      kFloat32,
                                      CreatePyIntTuple({4, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {1, 3, 4, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{{1, 3, 8, 8},
                                      kFloat32,
                                      CreatePyIntTuple({kValueAny, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {1, 3, -1, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{
      {1, 3, 8, 8}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {1, 3, -1, -1}, kFloat32},

    ResizeNearestNeighborV2GradParams{{1, 3, -1, -1},
                                      kFloat32,
                                      CreatePyIntTuple({4, 4}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {1, 3, 4, 4},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{{1, 3, -1, -1},
                                      kFloat32,
                                      CreatePyIntTuple({4, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {1, 3, 4, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{{1, 3, -1, -1},
                                      kFloat32,
                                      CreatePyIntTuple({kValueAny, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {1, 3, -1, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{
      {1, 3, -1, -1}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {1, 3, -1, -1}, kFloat32},

    ResizeNearestNeighborV2GradParams{
      {-2}, kFloat32, CreatePyIntTuple({4, 4}), CreateScalar(true), CreateScalar(true), {-1, -1, 4, 4}, kFloat32},
    ResizeNearestNeighborV2GradParams{{-2},
                                      kFloat32,
                                      CreatePyIntTuple({4, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {-1, -1, 4, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{{-2},
                                      kFloat32,
                                      CreatePyIntTuple({kValueAny, kValueAny}),
                                      CreateScalar(true),
                                      CreateScalar(true),
                                      {-1, -1, -1, -1},
                                      kFloat32},
    ResizeNearestNeighborV2GradParams{
      {-2}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {-1, -1, -1, -1}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
