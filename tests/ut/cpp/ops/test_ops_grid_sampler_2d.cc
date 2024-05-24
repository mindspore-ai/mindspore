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
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/grid_sampler_2d.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GridSampler2DShape {
  ShapeVector input_x_shape;
  ShapeVector grid_shape;
  ValuePtr interpolation_mode;
  ValuePtr padding_mode;
  ValuePtr align_corners;
  ShapeVector out_shape;
};

struct GridSampler2DDtype {
  TypePtr input_x_type;
  TypePtr grid_type;
  TypePtr out_type;
};

class TestGridSampler2D : public TestOps,
                          public testing::WithParamInterface<std::tuple<GridSampler2DShape, GridSampler2DDtype>> {};

TEST_P(TestGridSampler2D, grid_sampler_2d_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GridSampler2DFuncImpl grid_sampler_2d_func_impl;
  auto prim = std::make_shared<Primitive>("GridSampler2D");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_x_type, shape_param.input_x_shape);
  auto grid = std::make_shared<abstract::AbstractTensor>(dtype_param.grid_type, shape_param.grid_shape);
  auto interpolation_mode = shape_param.interpolation_mode->ToAbstract();
  auto padding_mode = shape_param.padding_mode->ToAbstract();
  auto align_corners = shape_param.align_corners->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape =
    grid_sampler_2d_func_impl.InferShape(prim, {input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype =
    grid_sampler_2d_func_impl.InferType(prim, {input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GridSampler2DOpShapeTestCases = testing::ValuesIn({
  /* static */
  GridSampler2DShape{
    {1, 2, 3, 4}, {1, 6, 7, 2}, MakeValue("nearest"), MakeValue("relection"), MakeValue(true), {1, 2, 6, 7}},
  /* dynamic shape */
  GridSampler2DShape{
    {-1, 2, 3, 4}, {-1, 6, 7, 2}, MakeValue("nearest"), MakeValue("zeros"), MakeValue(true), {-1, 2, 6, 7}},
  GridSampler2DShape{
    {5, 2, 3, 4}, {5, -1, 7, 2}, MakeValue("nearest"), MakeValue("relection"), MakeValue(true), {5, 2, -1, 7}},
  GridSampler2DShape{
    {-1, 2, 3, 4}, {-1, -1, 7, 2}, MakeValue("nearest"), MakeValue("relection"), MakeValue(false), {-1, 2, -1, 7}},
  GridSampler2DShape{
    {5, -1, 3, 4}, {5, 6, -1, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {5, -1, 6, -1}},
  GridSampler2DShape{
    {-1, -1, 3, 4}, {-1, 6, -1, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {-1, -1, 6, -1}},
  GridSampler2DShape{
    {-1, -1, -1, 4}, {-1, -1, -1, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {-1, -1, -1, -1}},
  GridSampler2DShape{
    {1, 2, -1, 4}, {1, 6, 7, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {1, 2, 6, 7}},
  GridSampler2DShape{
    {1, -1, -1, -1}, {1, 6, -1, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {1, -1, 6, -1}},
  GridSampler2DShape{
    {5, -1, -1, -1}, {5, -1, 1, 2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {5, -1, -1, 1}},
  /* dynamic rank */
  GridSampler2DShape{{-1, 3, 4, 5}, {-2}, MakeValue("nearest"), MakeValue("zeros"), MakeValue(false), {-1, -1, -1, -1}},
  GridSampler2DShape{
    {-2}, {5, -1, 7, 2}, MakeValue("bilinear"), MakeValue("relection"), MakeValue(false), {-1, -1, -1, -1}},
  GridSampler2DShape{{-2}, {-2}, MakeValue("bilinear"), MakeValue("zeros"), MakeValue(false), {-1, -1, -1, -1}},
});

auto GridSampler2DOpTypeTestCases = testing::ValuesIn({
  GridSampler2DDtype{kFloat16, kFloat16, kFloat16},
  GridSampler2DDtype{kFloat32, kFloat32, kFloat32},
  GridSampler2DDtype{kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestGridSampler2D, TestGridSampler2D,
                        testing::Combine(GridSampler2DOpShapeTestCases, GridSampler2DOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
