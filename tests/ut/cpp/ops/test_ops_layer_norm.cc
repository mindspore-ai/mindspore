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
#include "ops/ops_func_impl/layer_norm.h"

namespace mindspore {
namespace ops {
struct LayerNormOpParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector beta_shape;
  TypePtr beta_type;
  bool begin_norm_axis_has_value;
  int begin_norm_axis;
  bool begin_params_axis_has_value;
  int begin_params_axis;
  float epsilon;

  ShapeVector output_x_shape;
  TypePtr output_x_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector variance_shape;
  TypePtr variance_type;
};

class TestLayerNorm : public TestOps, public testing::WithParamInterface<LayerNormOpParams> {};

TEST_P(TestLayerNorm, layer_norm_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("LayerNorm");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  ASSERT_NE(input_x, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto beta = std::make_shared<abstract::AbstractTensor>(param.beta_type, param.beta_shape);
  ASSERT_NE(beta, nullptr);
  std::shared_ptr<abstract::AbstractScalar> begin_norm_axis = nullptr;
  if (param.begin_norm_axis_has_value) {
    begin_norm_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_norm_axis));
  } else {
    begin_norm_axis = std::make_shared<abstract::AbstractScalar>(std::make_shared<ValueAny>());
  }
  ASSERT_NE(begin_norm_axis, nullptr);
  std::shared_ptr<abstract::AbstractScalar> begin_param_axis = nullptr;
  if (param.begin_params_axis_has_value) {
    begin_param_axis = std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(param.begin_params_axis));
  } else {
    begin_param_axis = std::make_shared<abstract::AbstractScalar>(std::make_shared<ValueAny>());
  }
  ASSERT_NE(begin_param_axis, nullptr);
  auto epsilon = std::make_shared<abstract::AbstractScalar>(param.epsilon);
  ASSERT_NE(epsilon, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{
    std::move(input_x),          std::move(gamma),  std::move(beta), std::move(begin_norm_axis),
    std::move(begin_param_axis), std::move(epsilon)};
  auto infer_impl = std::make_shared<LayerNormFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shapes_ptr = infer_impl->InferShape(primitive, input_args);
  std::shared_ptr<abstract::TupleShape> infer_shapes =
    std::dynamic_pointer_cast<abstract::TupleShape>(infer_shapes_ptr);
  ASSERT_NE(infer_shapes, nullptr);
  auto infer_types_ptr = infer_impl->InferType(primitive, input_args);
  std::shared_ptr<Tuple> infer_types = std::dynamic_pointer_cast<Tuple>(infer_types_ptr);
  ASSERT_NE(infer_types, nullptr);
  auto expect_output_x_shape = std::make_shared<abstract::TensorShape>(param.output_x_shape);
  ASSERT_NE(expect_output_x_shape, nullptr);
  auto expect_output_x_type = std::make_shared<TensorType>(param.output_x_type);
  ASSERT_NE(expect_output_x_type, nullptr);
  auto expect_mean_shape = std::make_shared<abstract::TensorShape>(param.mean_shape);
  ASSERT_NE(expect_mean_shape, nullptr);
  auto expect_mean_type = std::make_shared<TensorType>(param.mean_type);
  ASSERT_NE(expect_mean_type, nullptr);
  auto expect_variance_shape = std::make_shared<abstract::TensorShape>(param.variance_shape);
  ASSERT_NE(expect_variance_shape, nullptr);
  auto expect_variance_type = std::make_shared<TensorType>(param.variance_type);
  ASSERT_NE(expect_variance_type, nullptr);
  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_output_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_mean_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_variance_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_output_x_type);
  ASSERT_TRUE(*((*infer_types)[1]) == *(expect_mean_type->element()));
  ASSERT_TRUE(*((*infer_types)[2]) == *(expect_variance_type->element()));
}

INSTANTIATE_TEST_CASE_P(TestLayerNormGroup, TestLayerNorm,
                        testing::Values(LayerNormOpParams{{-2},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          true,
                                                          1,
                                                          true,
                                                          1,
                                                          0.5,
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32},
                                        LayerNormOpParams{{2, 3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          true,
                                                          1,
                                                          true,
                                                          1,
                                                          0.5,
                                                          {2, 3, 4},
                                                          kFloat32,
                                                          {2, 1, 1},
                                                          kFloat32,
                                                          {2, 1, 1},
                                                          kFloat32},
                                        LayerNormOpParams{{2, 3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          false,
                                                          1,
                                                          true,
                                                          1,
                                                          0.5,
                                                          {2, 3, 4},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32,
                                                          {-2},
                                                          kFloat32},
                                        LayerNormOpParams{{2, 3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          {3, 4},
                                                          kFloat32,
                                                          true,
                                                          1,
                                                          false,
                                                          1,
                                                          0.5,
                                                          {2, 3, 4},
                                                          kFloat32,
                                                          {2, 1, 1},
                                                          kFloat32,
                                                          {2, 1, 1},
                                                          kFloat32}));
}  // namespace ops
}  // namespace mindspore
