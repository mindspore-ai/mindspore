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
#include "ops/ops_func_impl/layer_norm_grad.h"

namespace mindspore {
namespace ops {
struct LayerNormGradOpParams
{
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector variance_shape;
  TypePtr variance_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  int begin_norm_axis;
  int begin_params_axis;

  ShapeVector pd_x_shape;
  TypePtr pd_x_type;
  ShapeVector pd_gamma_shape;
  TypePtr pd_gamma_type;
  ShapeVector pd_beta_shape;
  TypePtr pd_beta_type;
};

class TestLayerNormGrad : public TestOps, public testing::WithParamInterface<LayerNormGradOpParams> {};

TEST_P(TestLayerNormGrad, layer_norm_grad_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("LayerNormGrad");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();

  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  ASSERT_NE(dy, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto variance = std::make_shared<abstract::AbstractTensor>(param.variance_type, param.variance_shape);
  ASSERT_NE(variance, nullptr);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  ASSERT_NE(mean, nullptr);
  auto begin_norm_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_norm_axis));
  ASSERT_NE(begin_norm_axis, nullptr);
  auto begin_param_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_params_axis));
  ASSERT_NE(begin_param_axis, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(dy), std::move(x), std::move(variance), std::move(mean),
                                                    std::move(gamma), std::move(begin_norm_axis), 
                                                    std::move(begin_param_axis)};
  auto infer_impl = std::make_shared<LayerNormGradFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shapes_ptr = infer_impl->InferShape(primitive, input_args);
  std::shared_ptr<abstract::TupleShape> infer_shapes = 
                                                     std::dynamic_pointer_cast<abstract::TupleShape>(infer_shapes_ptr);
  ASSERT_NE(infer_shapes, nullptr);
  auto infer_types_ptr = infer_impl->InferType(primitive, input_args);
  std::shared_ptr<Tuple> infer_types = std::dynamic_pointer_cast<Tuple>(infer_types_ptr);
  ASSERT_NE(infer_types, nullptr);

  auto expect_pd_x_shape = std::make_shared<abstract::TensorShape>(param.pd_x_shape);
  ASSERT_NE(expect_pd_x_shape, nullptr);
  auto expect_pd_x_type = std::make_shared<TensorType>(param.pd_x_type);
  ASSERT_NE(expect_pd_x_type, nullptr);
  auto expect_pd_gamma_shape = std::make_shared<abstract::TensorShape>(param.pd_gamma_shape);
  ASSERT_NE(expect_pd_gamma_shape, nullptr);
  auto expect_pd_gamma_type = std::make_shared<TensorType>(param.pd_gamma_type);
  ASSERT_NE(expect_pd_gamma_type, nullptr);
  auto expect_pd_beta_shape = std::make_shared<abstract::TensorShape>(param.pd_beta_shape);
  ASSERT_NE(expect_pd_beta_shape, nullptr);
  auto expect_pd_beta_type = std::make_shared<TensorType>(param.pd_beta_type);
  ASSERT_NE(expect_pd_beta_type, nullptr);

  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_pd_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_pd_gamma_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_pd_beta_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_pd_x_type);
  ASSERT_TRUE(*((*infer_types)[1]) == *expect_pd_gamma_type);
  ASSERT_TRUE(*((*infer_types)[2]) == *expect_pd_beta_type);
}

INSTANTIATE_TEST_CASE_P(
  TestLayerNormGradGroup, TestLayerNormGrad,
  testing::Values(LayerNormGradOpParams{{1, 2, 3, 4}, kFloat32, {1, 2, 3, 4}, kFloat32, {1}, kFloat32, {1}, kFloat32,
                                        {2, 3, 4}, kFloat32, 1, 1, {1, 2, 3, 4}, kFloat32, {2, 3, 4}, kFloat32,
                                        {2, 3, 4}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
