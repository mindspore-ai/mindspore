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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/rms_norm.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct TestRmsNormParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector y_shape;
  TypePtr y_type;
  ShapeVector rstd_shape;
};

class TestRmsNorm : public TestOps, public testing::WithParamInterface<TestRmsNormParams> {};

TEST_P(TestRmsNorm, rms_norm_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  auto eps_val = std::make_shared<FP32Imm>(1e-6f);
  auto eps = eps_val->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(gamma, nullptr);
  ASSERT_NE(eps, nullptr);
  auto infer_impl = std::make_shared<RmsNormFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(gamma), std::move(eps)};
  auto y_shape = std::make_shared<abstract::Shape>(param.y_shape);
  auto y_type = std::make_shared<TensorType>(param.y_type);
  auto rstd_shape = std::make_shared<abstract::Shape>(param.rstd_shape);

  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_shape, rstd_shape});
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>{y_type, std::make_shared<TensorType>(kFloat32)});
  DoFuncImplInferAndCompare<RmsNormFuncImpl>(kNameRmsNorm, input_args, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestRmsNorm, TestRmsNorm,
    testing::Values(TestRmsNormParams{{-1, -1, -1}, kFloat32, {-2}, kFloat32, {-1, -1, -1}, kFloat32, {-1, -1, -1}},
                    TestRmsNormParams{{2, 3, 4}, kFloat16, {-1, -1}, kFloat16, {2, 3, 4}, kFloat16, {2, 1, 1}},
                    TestRmsNormParams{{2, 3, 4}, kFloat32, {-1, 4}, kFloat32, {2, 3, 4}, kFloat32, {2, 1, 1}},
                    TestRmsNormParams{{-2}, kFloat32, {-1, 5}, kFloat32, {-2}, kFloat32, {-2}},
                    TestRmsNormParams{{-2}, kFloat16, {-2}, kFloat16, {-2}, kFloat16, {-2}},
                    TestRmsNormParams{{2, 3, 4}, kFloat32, {}, kFloat32, {2, 3, 4}, kFloat32, {2, 3, 4}}));
}  // namespace ops
}  // namespace mindspore
