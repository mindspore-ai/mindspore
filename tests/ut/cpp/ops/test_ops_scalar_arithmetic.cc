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
#include "ops/ops_func_impl/scalar_add.h"
#include "ops/ops_func_impl/scalar_sub.h"
#include "ops/ops_func_impl/scalar_mul.h"
#include "ops/ops_func_impl/scalar_div.h"
#include "ops/ops_func_impl/scalar_floor_div.h"
#include "ops/ops_func_impl/scalar_mod.h"
#include "ops/ops_func_impl/scalar_pow.h"
#include "ops/ops_func_impl/scalar_eq.h"
#include "ops/ops_func_impl/scalar_ge.h"
#include "ops/ops_func_impl/scalar_gt.h"
#include "ops/ops_func_impl/scalar_lt.h"
#include "ops/ops_func_impl/scalar_le.h"
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
struct ScalarArithmeticParams {
  ValuePtr x_value;
  TypePtr x_type;
  ValuePtr y_value;
  TypePtr y_type;
  TypePtr output_type;
};

class TestScalarArithmetic : public TestOps, public testing::WithParamInterface<ScalarArithmeticParams> {};

TEST_P(TestScalarArithmetic, scalar_arithmetic_dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractScalar>(param.x_value, param.x_type);
  auto input_y = std::make_shared<abstract::AbstractScalar>(param.y_value, param.y_type);
  auto expect_shape = abstract::kNoShape;
  auto expect_type = param.output_type;
  DoFuncImplInferAndCompare<ScalarAddFuncImpl>(kNameScalarAdd, {input_x, input_y}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarSubFuncImpl>(kNameScalarSub, {input_x, input_y}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarMulFuncImpl>(kNameScalarMul, {input_x, input_y}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarFloorDivFuncImpl>(kNameScalarFloorDiv, {input_x, input_y}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarModFuncImpl>(kNameScalarMod, {input_x, input_y}, abstract::kNoShape, expect_type);
  DoFuncImplInferAndCompare<ScalarPowFuncImpl>(kNameScalarPow, {input_x, input_y}, abstract::kNoShape, expect_type);
  auto div_expect_type = kFloat32;
  DoFuncImplInferAndCompare<ScalarDivFuncImpl>(kNameScalarDiv, {input_x, input_y}, abstract::kNoShape, div_expect_type);
  auto cmp_expect_type = kBool;
  DoFuncImplInferAndCompare<ScalarEqFuncImpl>(kNameScalarEq, {input_x, input_y}, abstract::kNoShape, cmp_expect_type);
  DoFuncImplInferAndCompare<ScalarGtFuncImpl>(kNameScalarGt, {input_x, input_y}, abstract::kNoShape, cmp_expect_type);
  DoFuncImplInferAndCompare<ScalarGeFuncImpl>(kNameScalarGe, {input_x, input_y}, abstract::kNoShape, cmp_expect_type);
  DoFuncImplInferAndCompare<ScalarLtFuncImpl>(kNameScalarLt, {input_x, input_y}, abstract::kNoShape, cmp_expect_type);
  DoFuncImplInferAndCompare<ScalarLtFuncImpl>(kNameScalarLe, {input_x, input_y}, abstract::kNoShape, cmp_expect_type);
}

INSTANTIATE_TEST_CASE_P(
    TestScalarArithmeticGroup, TestScalarArithmetic,
    testing::Values(ScalarArithmeticParams{CreateScalar<int>(1), kInt32, CreateScalar<float>(2), kFloat32, kFloat32},
                    ScalarArithmeticParams{CreateScalar(true), kBool, CreateScalar(false), kBool, kInt32},
                    ScalarArithmeticParams{CreateScalar<float>(1), kFloat32, CreateScalar(false), kBool, kFloat32},
                    ScalarArithmeticParams{kValueAny, kFloat32, kValueAny, kFloat32, kFloat32},
                    ScalarArithmeticParams{CreateScalar<int>(2), kInt32, kValueAny, kInt32, kInt32}));
}  // namespace ops
}  // namespace mindspore
