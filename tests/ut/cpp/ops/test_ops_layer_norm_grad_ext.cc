/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/layer_norm_grad_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct LayerNormGradExtOpParams {
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr normalized_shape;
  ShapeVector rstd_shape;
  TypePtr rstd_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector beta_shape;
  TypePtr beta_type;

  ShapeVector pd_x_shape;
  TypePtr pd_x_type;
  ShapeVector pd_gamma_shape;
  TypePtr pd_gamma_type;
  ShapeVector pd_beta_shape;
  TypePtr pd_beta_type;
};

class TestLayerNormGradExt : public TestOps, public testing::WithParamInterface<LayerNormGradExtOpParams> {};

TEST_P(TestLayerNormGradExt, layer_norm_grad_ext_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("LayerNormGradExt");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();

  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  ASSERT_NE(dy, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto normalized_shape = param.normalized_shape->ToAbstract();
  ASSERT_NE(normalized_shape, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto beta = std::make_shared<abstract::AbstractTensor>(param.beta_type, param.beta_shape);
  ASSERT_NE(beta, nullptr);
  auto rstd = std::make_shared<abstract::AbstractTensor>(param.rstd_type, param.rstd_shape);
  ASSERT_NE(rstd, nullptr);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  ASSERT_NE(mean, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(dy),
                                                    std::move(x),
                                                    std::move(normalized_shape),
                                                    std::move(mean),
                                                    std::move(rstd),
                                                    std::move(gamma),
                                                    std::move(beta)};
  auto infer_impl = std::make_shared<LayerNormGradExtFuncImpl>();
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

INSTANTIATE_TEST_CASE_P(TestLayerNormGradExtGroup, TestLayerNormGradExt,
                        testing::Values(LayerNormGradExtOpParams{{1, 2, 3, 4},
                                                                 kFloat32,
                                                                 {1, 2, 3, 4},
                                                                 kFloat32,
                                                                 CreateTuple({2, 3, 4}),
                                                                 {1},
                                                                 kFloat32,
                                                                 {1},
                                                                 kFloat32,
                                                                 {2, 3, 4},
                                                                 kFloat32,
                                                                 {2, 3, 4},
                                                                 kFloat32,
                                                                 {1, 2, 3, 4},
                                                                 kFloat32,
                                                                 {2, 3, 4},
                                                                 kFloat32,
                                                                 {2, 3, 4},
                                                                 kFloat32}));
}  // namespace ops
}  // namespace mindspore
