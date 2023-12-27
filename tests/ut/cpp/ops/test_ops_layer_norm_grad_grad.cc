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
#include "common/common_test.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/layer_norm_grad_grad.h"

namespace mindspore {
namespace ops {
struct LayerNormGradGradOpParams
{
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector variance_shape;
  TypePtr variance_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector d_dx_shape;
  TypePtr d_dx_type;
  ShapeVector d_dg_shape;
  TypePtr d_dg_type;
  ShapeVector d_db_shape;
  TypePtr d_db_type;
  int begin_norm_axis;
  int begin_params_axis;

  ShapeVector sopd_x_shape;
  TypePtr sopd_x_type;
  ShapeVector sopd_dy_shape;
  TypePtr sopd_dy_type;
  ShapeVector sopd_gamma_shape;
  TypePtr sopd_gamma_type;
};

class TestLayerNormGradGrad : public TestOps, public testing::WithParamInterface<LayerNormGradGradOpParams> {};

TEST_P(TestLayerNormGradGrad, layer_norm_grad_grad_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("LayerNormGradGrad");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  ASSERT_NE(dy, nullptr);
  auto variance = std::make_shared<abstract::AbstractTensor>(param.variance_type, param.variance_shape);
  ASSERT_NE(variance, nullptr);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  ASSERT_NE(mean, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto d_dx = std::make_shared<abstract::AbstractTensor>(param.d_dx_type, param.d_dx_shape);
  ASSERT_NE(d_dx, nullptr);
  auto d_dg = std::make_shared<abstract::AbstractTensor>(param.d_dg_type, param.d_dg_shape);
  ASSERT_NE(d_dg, nullptr);
  auto d_db = std::make_shared<abstract::AbstractTensor>(param.d_db_type, param.d_db_shape);
  ASSERT_NE(d_db, nullptr);
  auto begin_norm_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_norm_axis));
  ASSERT_NE(begin_norm_axis, nullptr);
  auto begin_param_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_params_axis));
  ASSERT_NE(begin_param_axis, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(dy), std::move(variance), std::move(mean),
                                                    std::move(gamma), std::move(d_dx), std::move(d_dg), std::move(d_db),
                                                    std::move(begin_norm_axis), std::move(begin_param_axis)};
  auto infer_impl = std::make_shared<LayerNormGradGradFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shapes_ptr = infer_impl->InferShape(primitive, input_args);
  std::shared_ptr<abstract::TupleShape> infer_shapes = 
                                                     std::dynamic_pointer_cast<abstract::TupleShape>(infer_shapes_ptr);
  ASSERT_NE(infer_shapes, nullptr);
  auto infer_types_ptr = infer_impl->InferType(primitive, input_args);
  std::shared_ptr<Tuple> infer_types = std::dynamic_pointer_cast<Tuple>(infer_types_ptr);
  ASSERT_NE(infer_types, nullptr);

  auto expect_sopd_x_shape = std::make_shared<abstract::TensorShape>(param.sopd_x_shape);
  ASSERT_NE(expect_sopd_x_shape, nullptr);
  auto expect_sopd_x_type = std::make_shared<TensorType>(param.sopd_x_type);
  ASSERT_NE(expect_sopd_x_type, nullptr);
  auto expect_sopd_dy_shape = std::make_shared<abstract::TensorShape>(param.sopd_dy_shape);
  ASSERT_NE(expect_sopd_dy_shape, nullptr);
  auto expect_sopd_dy_type = std::make_shared<TensorType>(param.sopd_dy_type);
  ASSERT_NE(expect_sopd_dy_type, nullptr);
  auto expect_sopd_gamma_shape = std::make_shared<abstract::TensorShape>(param.sopd_gamma_shape);
  ASSERT_NE(expect_sopd_gamma_shape, nullptr);
  auto expect_sopd_gamma_type = std::make_shared<TensorType>(param.sopd_gamma_type);
  ASSERT_NE(expect_sopd_gamma_type, nullptr);

  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_sopd_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_sopd_dy_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_sopd_gamma_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_sopd_x_type);
  ASSERT_TRUE(*((*infer_types)[1]) == *expect_sopd_dy_type);
  ASSERT_TRUE(*((*infer_types)[2]) == *expect_sopd_gamma_type);
}

INSTANTIATE_TEST_CASE_P(
  TestLayerNormGradGradGroup, TestLayerNormGradGrad,
  testing::Values(LayerNormGradGradOpParams{{1, 2, 3, 4}, kFloat32, {1, 2, 3, 4}, kFloat32, {1}, kFloat32, {1}, kFloat32,
                                        {2, 3, 4}, kFloat32, {1, 2, 3, 4}, kFloat32, {2, 3, 4}, kFloat32,
                                        {2, 3, 4}, kFloat32, 1, 1, {1, 2, 3, 4}, kFloat32, {1, 2, 3, 4}, kFloat32,
                                        {2, 3, 4}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
