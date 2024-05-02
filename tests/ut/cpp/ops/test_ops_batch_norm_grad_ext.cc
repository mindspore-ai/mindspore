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
 * limitations under the License.a
 */

#include "common/common_test.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/batch_norm_grad_ext.h"

namespace mindspore {
namespace ops {
struct BatchNormGradExtOpParams
{
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  ShapeVector running_mean_shape;
  TypePtr running_mean_type;
  ShapeVector running_var_shape;
  TypePtr running_var_type;
  ShapeVector saved_mean_shape;
  TypePtr saved_mean_type;
  ShapeVector saved_rstd_shape;
  TypePtr saved_rstd_type;
  ValuePtr training;
  ValuePtr eps;

  ShapeVector pd_x_shape;
  TypePtr pd_x_type;
  ShapeVector pd_weight_shape;
  TypePtr pd_weight_type;
  ShapeVector pd_bias_shape;
  TypePtr pd_bias_type;
};

class TestBatchNormGradExt : public TestOps, public testing::WithParamInterface<BatchNormGradExtOpParams> {};

TEST_P(TestBatchNormGradExt, batch_norm_grad_ext_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("BatchNormGradExt");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();

  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  ASSERT_NE(dy, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  ASSERT_NE(weight, nullptr);
  auto running_mean = std::make_shared<abstract::AbstractTensor>(param.running_mean_type, param.running_mean_shape);
  ASSERT_NE(running_mean, nullptr);
  auto running_var = std::make_shared<abstract::AbstractTensor>(param.running_var_type, param.running_var_shape);
  ASSERT_NE(running_var, nullptr);
  auto saved_mean = std::make_shared<abstract::AbstractTensor>(param.saved_mean_type, param.saved_mean_shape);
  ASSERT_NE(saved_mean, nullptr);
  auto saved_rstd = std::make_shared<abstract::AbstractTensor>(param.saved_rstd_type, param.saved_rstd_shape);
  ASSERT_NE(saved_rstd, nullptr);
  auto training = param.training->ToAbstract();
  ASSERT_NE(training, nullptr);
  auto eps = param.eps->ToAbstract();
  ASSERT_NE(eps, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{
    std::move(dy), std::move(x), std::move(weight), std::move(running_mean), std::move(running_var),
    std::move(saved_mean), std::move(saved_rstd), std::move(training), std::move(eps)};
  auto infer_impl = std::make_shared<BatchNormGradExtFuncImpl>();
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
  auto expect_pd_weight_shape = std::make_shared<abstract::TensorShape>(param.pd_weight_shape);
  ASSERT_NE(expect_pd_weight_shape, nullptr);
  auto expect_pd_weight_type = std::make_shared<TensorType>(param.pd_weight_type);
  ASSERT_NE(expect_pd_weight_type, nullptr);
  auto expect_pd_bias_shape = std::make_shared<abstract::TensorShape>(param.pd_bias_shape);
  ASSERT_NE(expect_pd_bias_shape, nullptr);
  auto expect_pd_bias_type = std::make_shared<TensorType>(param.pd_bias_type);
  ASSERT_NE(expect_pd_bias_type, nullptr);

  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_pd_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_pd_weight_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_pd_bias_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_pd_x_type);
}

INSTANTIATE_TEST_CASE_P(
  TestBatchNormGradExtGroup, TestBatchNormGradExt,
  testing::Values(BatchNormGradExtOpParams{{24, 8, 96, 96},
                                           kFloat32,
                                           {24, 8, 96, 96},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           CreateScalar<bool>(false),
                                           CreateScalar<double>(1e-5f),
                                           {24, 8, 96, 96},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32},
                  BatchNormGradExtOpParams{{-1, -1, -1, -1},
                                           kFloat32,
                                           {-1, -1, -1, -1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           CreateScalar<bool>(true),
                                           CreateScalar<double>(1e-5f),
                                           {-1, -1, -1, -1},
                                           kFloat32,
                                           {-1},
                                           kFloat32,
                                           {-1},
                                           kFloat32},
                  BatchNormGradExtOpParams{{-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           CreateScalar<bool>(true),
                                           CreateScalar<double>(1e-5f),
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32,
                                           {-2},
                                           kFloat32},
                  BatchNormGradExtOpParams{{-1, 8, 64, 64},
                                           kFloat32,
                                           {-1, 8, 64, 64},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           CreateScalar<bool>(true),
                                           CreateScalar<double>(1e-5f),
                                           {-1, 8, 64, 64},
                                           kFloat32,
                                           {8},
                                           kFloat32,
                                           {8},
                                           kFloat32}));
  }  // namespace ops
}  // namespace mindspore
