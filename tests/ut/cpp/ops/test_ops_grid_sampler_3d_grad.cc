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
#include "ops/ops_func_impl/grid_sampler_3d_grad.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GridSampler3DGradShape {
  ShapeVector grad_shape;
  ShapeVector input_x_shape;
  ShapeVector grid_shape;
  ValuePtr interpolation_mode;
  ValuePtr padding_mode;
  ValuePtr align_corners;
  ShapeVector dx_shape;
  ShapeVector dgrid_shape;
};

struct GridSampler3DGradDtype {
  TypePtr grad_type;
  TypePtr input_x_type;
  TypePtr grid_type;
  TypePtr dx_type;
  TypePtr dgrid_type;
};

class TestGridSampler3DGrad
    : public TestOps,
      public testing::WithParamInterface<std::tuple<GridSampler3DGradShape, GridSampler3DGradDtype>> {};

TEST_P(TestGridSampler3DGrad, grid_sampler_3d_grad_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GridSampler3DGradFuncImpl grid_sampler_3d_grad_func_impl;
  auto prim = std::make_shared<Primitive>("GridSampler3DGrad");

  auto grad = std::make_shared<abstract::AbstractTensor>(dtype_param.grad_type, shape_param.grad_shape);
  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_x_type, shape_param.input_x_shape);
  auto grid = std::make_shared<abstract::AbstractTensor>(dtype_param.grid_type, shape_param.grid_shape);
  auto interpolation_mode = shape_param.interpolation_mode->ToAbstract();
  auto padding_mode = shape_param.padding_mode->ToAbstract();
  auto align_corners = shape_param.align_corners->ToAbstract();

  std::vector<BaseShapePtr> shapes_list = {std::make_shared<abstract::TensorShape>(shape_param.dx_shape),
                                           std::make_shared<abstract::TensorShape>(shape_param.dgrid_shape)};
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<BaseShapePtr>{shapes_list});
  auto dx_type_out = std::make_shared<TensorType>(dtype_param.dx_type);
  auto dgrid_type_out = std::make_shared<TensorType>(dtype_param.dgrid_type);
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{dx_type_out, dgrid_type_out});

  auto out_shape = grid_sampler_3d_grad_func_impl.InferShape(
    prim, {grad, input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = grid_sampler_3d_grad_func_impl.InferType(
    prim, {grad, input_x, grid, interpolation_mode, padding_mode, align_corners});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GridSampler3DGradOpShapeTestCases = testing::ValuesIn({
  /* static */
  GridSampler3DGradShape{{1, 2, 7, 8, 9},
                         {1, 2, 3, 4, 5},
                         {1, 7, 8, 9, 3},
                         MakeValue("nearest"),
                         MakeValue("relection"),
                         MakeValue(true),
                         {1, 2, 3, 4, 5},
                         {1, 7, 8, 9, 3}},
  /* dynamic shape */
  GridSampler3DGradShape{{6, -1, 7, 8, 9},
                         {6, -1, 3, 4, 5},
                         {6, 7, 8, 9, 3},
                         MakeValue("nearest"),
                         MakeValue("zeros"),
                         MakeValue(true),
                         {6, -1, 3, 4, 5},
                         {6, 7, 8, 9, 3}},
  GridSampler3DGradShape{{6, 2, -1, 8, 9},
                         {6, 2, 3, 4, 5},
                         {6, -1, 8, 9, 3},
                         MakeValue("nearest"),
                         MakeValue("relection"),
                         MakeValue(true),
                         {6, 2, 3, 4, 5},
                         {6, -1, 8, 9, 3}},
  GridSampler3DGradShape{{-1, 2, 7, -1, 9},
                         {-1, 2, 3, -1, 5},
                         {-1, 7, -1, 9, 3},
                         MakeValue("nearest"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {-1, 2, 3, -1, 5},
                         {-1, 7, -1, 9, 3}},

  GridSampler3DGradShape{{6, -1, 7, 8, -1},
                         {6, -1, 3, 4, 5},
                         {6, 7, 8, -1, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {6, -1, 3, 4, 5},
                         {6, 7, 8, -1, 3}},
  GridSampler3DGradShape{{6, -1, 7, -1, -1},
                         {6, -1, 3, 4, 5},
                         {6, 7, -1, -1, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {6, -1, 3, 4, 5},
                         {6, 7, -1, -1, 3}},
  GridSampler3DGradShape{{-1, -1, -1, -1, -1},
                         {-1, -1, -1, 4, 5},
                         {-1, -1, -1, -1, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {-1, -1, -1, 4, 5},
                         {-1, -1, -1, -1, 3}},
  GridSampler3DGradShape{{1, 2, 7, 8, 9},
                         {1, 2, 3, -1, 5},
                         {1, 7, 8, 9, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {1, 2, 3, -1, 5},
                         {1, 7, 8, 9, 3}},
  GridSampler3DGradShape{{-1, 2, 6, -1, 8},
                         {-1, 2, -1, -1, -1},
                         {-1, 6, -1, 8, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {-1, 2, -1, -1, -1},
                         {-1, 6, -1, 8, 3}},
  GridSampler3DGradShape{{1, 2, 6, 7, 8},
                         {1, 2, -1, -1, -1},
                         {1, 6, 7, 8, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {1, 2, -1, -1, -1},
                         {1, 6, 7, 8, 3}},
  /* dynamic rank */
  GridSampler3DGradShape{{-1, 2, -1, -1, -1},
                         {-1, 2, 3, 4, 5},
                         {-2},
                         MakeValue("nearest"),
                         MakeValue("zeros"),
                         MakeValue(false),
                         {-1, -1, -1, -1, -1},
                         {-1, -1, -1, -1, 3}},
  GridSampler3DGradShape{{6, -1, 7, 8, 9},
                         {-2},
                         {6, 7, 8, 9, 3},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {-1, -1, -1, -1, -1},
                         {-1, -1, -1, -1, 3}},
  GridSampler3DGradShape{{-1, -1, -1, -1, -1},
                         {-2},
                         {-2},
                         MakeValue("bilinear"),
                         MakeValue("relection"),
                         MakeValue(false),
                         {-1, -1, -1, -1, -1},
                         {-1, -1, -1, -1, 3}},
});

auto GridSampler3DGradOpTypeTestCases = testing::ValuesIn({
  GridSampler3DGradDtype{kFloat16, kFloat16, kFloat16, kFloat16, kFloat16},
  GridSampler3DGradDtype{kFloat32, kFloat32, kFloat32, kFloat32, kFloat32},
  GridSampler3DGradDtype{kFloat64, kFloat64, kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestGridSampler3DGrad, TestGridSampler3DGrad,
                        testing::Combine(GridSampler3DGradOpShapeTestCases, GridSampler3DGradOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
