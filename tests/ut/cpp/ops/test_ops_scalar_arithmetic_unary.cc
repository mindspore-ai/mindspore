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
#include <functional>
#include "common/common_test.h"
#include "ops/ops_func_impl/scalar_bool.h"
#include "ops/ops_func_impl/scalar_log.h"
#include "ops/ops_func_impl/scalar_uadd.h"
#include "ops/ops_func_impl/scalar_usub.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ScalarArithmeticUnaryParams {
  ValuePtr x_value;
  TypePtr x_type;
};

class TestScalarArithmeticUnary : public TestOps, public testing::WithParamInterface<ScalarArithmeticUnaryParams> {};

TEST_P(TestScalarArithmeticUnary, scalar_arithmetic_unary_dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractScalar>(param.x_value, param.x_type);
  auto expect_shape = abstract::kNoShape;
  auto expect_type = param.x_type;
  DoFuncImplInferAndCompare<ScalarUaddFuncImpl>(kNameScalarUadd, {input_x}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarUsubFuncImpl>(kNameScalarUsub, {input_x}, abstract::kNoShape, expect_type);
  auto bool_expect_type = kBool;
  DoFuncImplInferAndCompare<ScalarBoolFuncImpl>(kNameScalarBool, {input_x}, abstract::kNoShape, bool_expect_type);
  auto log_expect_type = kFloat32;
  DoFuncImplInferAndCompare<ScalarLogFuncImpl>(kNameScalarLog, {input_x}, abstract::kNoShape, log_expect_type);
}

INSTANTIATE_TEST_CASE_P(
    TestScalarArithmeticUnaryGroup, TestScalarArithmeticUnary,
    testing::Values(ScalarArithmeticUnaryParams{CreateScalar<int>(1), kInt32},
                    ScalarArithmeticUnaryParams{CreateScalar(true), kBool},
                    ScalarArithmeticUnaryParams{CreateScalar<float>(1), kFloat32},
                    ScalarArithmeticUnaryParams{kValueAny, kFloat32}));
}  // namespace ops
}  // namespace mindspore
