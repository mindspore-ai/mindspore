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
#include "ops/ops_func_impl/batch_norm_grad.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct BNGParams {
  ValuePtr is_training;
  ValuePtr epsilon;
  ValuePtr data_format;
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector scale_shape;
  TypePtr scale_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector variance_shape;
  TypePtr variance_type;
  ShapeVector rer_shape;
  TypePtr rer_type;
  ShapeVector dx_shape;
  ShapeVector dscale_shape;
};

class TestBatchNormGrad : public TestOps, public testing::WithParamInterface<BNGParams> {};

TEST_P(TestBatchNormGrad, dyn_shape) {
  const auto &param = GetParam();
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto scale = std::make_shared<abstract::AbstractTensor>(param.scale_type, param.scale_shape);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  auto variance = std::make_shared<abstract::AbstractTensor>(param.variance_type, param.variance_shape);
  auto reserve = std::make_shared<abstract::AbstractTensor>(param.rer_type, param.rer_shape);
  auto is_training = param.is_training->ToAbstract();
  auto epsilon = param.epsilon->ToAbstract();
  auto data_format = param.data_format->ToAbstract();
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(scale, nullptr);
  ASSERT_NE(mean, nullptr);
  ASSERT_NE(variance, nullptr);
  ASSERT_NE(reserve, nullptr);
  ASSERT_NE(is_training, nullptr);
  ASSERT_NE(epsilon, nullptr);
  ASSERT_NE(data_format, nullptr);
  auto [expect_shape, expect_type] = MakeOutputTupleShapeAndType(
    {param.dx_shape, param.dscale_shape, param.dscale_shape}, {param.x_type, param.scale_type, param.scale_type});
  DoFuncImplInferAndCompare<BatchNormGradFuncImpl>(
    "BatchNormGrad", {dy, x, scale, mean, variance, reserve, is_training, epsilon, data_format}, expect_shape,
    expect_type);
}

INSTANTIATE_TEST_CASE_P(TestBatchNormGrad, TestBatchNormGrad,
                        testing::Values(BNGParams{CreateScalar(true),
                                                  CreateScalar<double>(1e-5f),
                                                  CreateScalar<int64_t>(0),
                                                  {24, 8, 96, 96},
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
                                                  {24, 8, 96, 96},
                                                  {8}},
                                        BNGParams{CreateScalar(kValueAny),
                                                  CreateScalar<double>(1e-5f),
                                                  CreateScalar<int64_t>(0),
                                                  {-1, 8, -1, 96},
                                                  kFloat32,
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
                                        BNGParams{CreateScalar(false),
                                                  CreateScalar(kValueAny),
                                                  CreateScalar<int64_t>(0),
                                                  {-2},
                                                  kFloat32,
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
                                        BNGParams{CreateScalar(false),
                                                  CreateScalar<double>(1e-5f),
                                                  CreateScalar(kValueAny),
                                                  {-1, -1, -1, -1},
                                                  kFloat32,
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
                                                  {-2}}));
}  // namespace ops
}  // namespace mindspore
