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
#include "ops/ops_func_impl/grid_sampler_3d.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GridSampler3DShape {
  ShapeVector input_x_shape;
  ShapeVector grid_shape;
  ValuePtr interpolation_mode;
  ValuePtr padding_mode;
  ValuePtr align_corners;
  ShapeVector out_shape;
};

struct GridSampler3DDtype {
  TypePtr input_x_type;
  TypePtr grid_type;
  TypePtr out_type;
};

class TestGridSampler3D : public TestOps,
                          public testing::WithParamInterface<std::tuple<GridSampler3DShape, GridSampler3DDtype>> {};

TEST_P(TestGridSampler3D, grid_sampler_3d_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GridSampler3DFuncImpl grid_sampler_3d_func_impl;
  auto prim = std::make_shared<Primitive>("GridSampler3D");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_x_type, shape_param.input_x_shape);
  auto grid = std::make_shared<abstract::AbstractTensor>(dtype_param.grid_type, shape_param.grid_shape);
  auto interpolation_mode = shape_param.interpolation_mode->ToAbstract();
  auto padding_mode = shape_param.padding_mode->ToAbstract();
  auto align_corners = shape_param.align_corners->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape =
    grid_sampler_3d_func_impl.InferShape(prim, {input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype =
    grid_sampler_3d_func_impl.InferType(prim, {input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GridSampler3DOpShapeTestCases = testing::ValuesIn({
  /* static */
  GridSampler3DShape{
    {1, 2, 3, 4, 5}, {1, 7, 8, 9, 3}, MakeValue("nearest"), MakeValue("relection"), MakeValue(true), {1, 2, 7, 8, 9}},
  /* dynamic shape */
  GridSampler3DShape{
    {1, -1, 3, 4, 5}, {1, 7, 8, 9, 3}, MakeValue("nearest"), MakeValue("zeros"), MakeValue(true), {1, -1, 7, 8, 9}},
  GridSampler3DShape{
    {6, 2, 3, 4, 5}, {6, -1, 8, 9, 3}, MakeValue("nearest"), MakeValue("relection"), MakeValue(true), {6, 2, -1, 8, 9}},
  GridSampler3DShape{{-1, 2, 3, -1, 5},
                     {-1, 7, -1, 9, 3},
                     MakeValue("nearest"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {-1, 2, 7, -1, 9}},

  GridSampler3DShape{{6, -1, 3, 4, 5},
                     {6, 7, 8, -1, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {6, -1, 7, 8, -1}},
  GridSampler3DShape{{6, -1, 3, 4, 5},
                     {6, 7, -1, -1, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {6, -1, 7, -1, -1}},
  GridSampler3DShape{{-1, -1, -1, 4, 5},
                     {-1, -1, -1, -1, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {-1, -1, -1, -1, -1}},
  GridSampler3DShape{{1, 2, 3, -1, 5},
                     {1, 7, 8, 9, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {1, 2, 7, 8, 9}},
  GridSampler3DShape{{-1, 2, -1, -1, 5},
                     {-1, 6, -1, 8, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {-1, 2, 6, -1, 8}},
  GridSampler3DShape{{1, 2, -1, -1, -1},
                     {1, 6, 7, 8, 3},
                     MakeValue("bilinear"),
                     MakeValue("relection"),
                     MakeValue(false),
                     {1, 2, 6, 7, 8}},
  /* dynamic rank */
  GridSampler3DShape{
    {-1, 2, 3, 4, 5}, {-2}, MakeValue("nearest"), MakeValue("zeros"), MakeValue(false), {-1, -1, -1, -1, -1}},
  GridSampler3DShape{{-2}, {-2}, MakeValue("bilinear"), MakeValue("relection"), MakeValue(false), {-1, -1, -1, -1, -1}},
  GridSampler3DShape{
    {-2}, {6, 7, 8, 9, 3}, MakeValue("bilinear"), MakeValue("relection"), MakeValue(false), {-1, -1, -1, -1, -1}},
});

auto GridSampler3DOpTypeTestCases = testing::ValuesIn({
  GridSampler3DDtype{kFloat16, kFloat16, kFloat16},
  GridSampler3DDtype{kFloat32, kFloat32, kFloat32},
  GridSampler3DDtype{kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestGridSampler3D, TestGridSampler3D,
                        testing::Combine(GridSampler3DOpShapeTestCases, GridSampler3DOpTypeTestCases));

}  // namespace ops
}  // namespace mindspore
