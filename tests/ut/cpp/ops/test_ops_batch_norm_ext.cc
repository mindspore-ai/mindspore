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

#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/batch_norm_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct BatchNormExtParams {
  ValuePtr training;
  ValuePtr epsilon;
  ValuePtr momentum;

  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  ShapeVector bias_shape;
  TypePtr bias_type;
  ShapeVector running_mean_shape;
  TypePtr running_mean_type;
  ShapeVector running_var_shape;
  TypePtr running_var_type;
  ShapeVector y_shape;
  ShapeVector saved_mean_shape;
};

class TestBatchNormExt : public TestOps, public testing::WithParamInterface<BatchNormExtParams> {};

TEST_P(TestBatchNormExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  auto bias = std::make_shared<abstract::AbstractTensor>(param.bias_type, param.bias_shape);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.running_mean_type, param.running_mean_shape);
  auto variance = std::make_shared<abstract::AbstractTensor>(param.running_var_type, param.running_var_shape);
  auto training = param.training->ToAbstract();
  auto epsilon = param.epsilon->ToAbstract();
  auto momentum = param.momentum->ToAbstract();

  ASSERT_NE(x, nullptr);
  ASSERT_NE(weight, nullptr);
  ASSERT_NE(bias, nullptr);
  ASSERT_NE(mean, nullptr);
  ASSERT_NE(variance, nullptr);
  ASSERT_NE(training, nullptr);
  ASSERT_NE(epsilon, nullptr);
  ASSERT_NE(momentum, nullptr);

  auto [expect_shape, expect_type] = MakeOutputTupleShapeAndType(
    {param.y_shape, param.saved_mean_shape, param.saved_mean_shape},
    {param.x_type, param.weight_type, param.weight_type});
  DoFuncImplInferAndCompare<BatchNormExtFuncImpl>(
    "BatchNormExt", {x, weight, bias, mean, variance, training, momentum, epsilon}, expect_shape,
    expect_type);
}

INSTANTIATE_TEST_CASE_P(TestBatchNormExt, TestBatchNormExt,
                        testing::Values(BatchNormExtParams{CreateScalar(true),
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(0.1),
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
                                                           {24, 8, 96, 96},
                                                           {8}},
                                        BatchNormExtParams{CreateScalar(true),
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(0.1),
                                                           {-1, 8, -1, 96},
                                                           kFloat32,
                                                           {-1},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {-1, 8, -1, 96},
                                                           {-1}},
                                        BatchNormExtParams{CreateScalar(false),
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(0.1),
                                                           {-2},
                                                           kFloat32,
                                                           {-1},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {-2},
                                                           {-1}},
                                        BatchNormExtParams{CreateScalar(false),
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(0.1),
                                                           {-1, -1, -1, -1},
                                                           kFloat32,
                                                           {-2},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {8},
                                                           kFloat32,
                                                           {-1, -1, -1, -1},
                                                           {-1}}));
}  // namespace ops
}  // namespace mindspore
