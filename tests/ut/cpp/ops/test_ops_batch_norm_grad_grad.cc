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
#include "ops/ops_func_impl/batch_norm_grad_grad.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct BNGGParams {
  ValuePtr is_training;
  ValuePtr epsilon;
  ValuePtr data_format;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector scale_shape;
  TypePtr scale_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector variance_shape;
  TypePtr variance_type;
  ShapeVector dydx_shape;
  TypePtr dydx_type;
  ShapeVector dydscale_shape;
  TypePtr dydscale_type;
  ShapeVector dydbias_shape;
  TypePtr dydbias_type;
  ShapeVector dx_shape;
  ShapeVector ddy_shape;
  ShapeVector dscale_shape;
};

class TestBatchNormGradGrad : public TestOps, public testing::WithParamInterface<BNGGParams> {};

TEST_P(TestBatchNormGradGrad, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  auto scale = std::make_shared<abstract::AbstractTensor>(param.scale_type, param.scale_shape);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  auto variance = std::make_shared<abstract::AbstractTensor>(param.variance_type, param.variance_shape);
  auto dout_dx = std::make_shared<abstract::AbstractTensor>(param.dydx_type, param.dydx_shape);
  auto dout_dscale = std::make_shared<abstract::AbstractTensor>(param.dydscale_type, param.dydscale_shape);
  auto dout_dbias = std::make_shared<abstract::AbstractTensor>(param.dydbias_type, param.dydbias_shape);
  auto is_training = param.is_training->ToAbstract();
  auto epsilon = param.epsilon->ToAbstract();
  auto data_format = param.data_format->ToAbstract();
  auto [expect_shape, expect_type] = MakeOutputTupleShapeAndType({param.dx_shape, param.ddy_shape, param.dscale_shape},
                                                                 {param.x_type, param.dy_type, param.scale_type});
  DoFuncImplInferAndCompare<BatchNormGradGradFuncImpl>(
    "BatchNormGradGrad",
    {x, dy, scale, mean, variance, dout_dx, dout_dscale, dout_dbias, is_training, epsilon, data_format}, expect_shape,
    expect_type);
}

INSTANTIATE_TEST_CASE_P(TestBatchNormGradGrad, TestBatchNormGradGrad,
                        testing::Values(BNGGParams{CreateScalar(true),
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
                                                   {24, 8, 96, 96},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {24, 8, 96, 96},
                                                   {24, 8, 96, 96},
                                                   {8}},
                                        BNGGParams{CreateScalar(kValueAny),
                                                   CreateScalar<double>(1e-5f),
                                                   CreateScalar<int64_t>(0),
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {8},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   {24, -1, 96, -1},
                                                   {8}},
                                        BNGGParams{CreateScalar(false),
                                                   CreateScalar(kValueAny),
                                                   CreateScalar<int64_t>(0),
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   {24, -1, 96, -1},
                                                   {-1}},
                                        BNGGParams{CreateScalar(false),
                                                   CreateScalar<double>(1e-5f),
                                                   CreateScalar(kValueAny),
                                                   {-2},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-1},
                                                   kFloat32,
                                                   {-2},
                                                   {-2},
                                                   {-1}},
                                        BNGGParams{CreateScalar(true),
                                                   CreateScalar(kValueAny),
                                                   CreateScalar<int64_t>(0),
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {-2},
                                                   kFloat32,
                                                   {24, -1, 96, -1},
                                                   {24, -1, 96, -1},
                                                   {-2}}));
}  // namespace ops
}  // namespace mindspore
