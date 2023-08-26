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
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/gen_ops_name.h"
#include "ops/ops_func_impl/transpose.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore::ops {
#define I64(x) (static_cast<int64_t>((x)))

struct TransParams {
  bool dynamic_len;
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr perms;
  ShapeVector out_shape;
};

class TestTranspose : public TestOps, public testing::WithParamInterface<TransParams> {};

TEST_P(TestTranspose, dyn_shape) {
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  auto perms = param.perms->ToAbstract();
  ASSERT_NE(perms, nullptr);
  if (param.dynamic_len) {
    auto tuple_abs = perms->cast<abstract::AbstractTuplePtr>();
    ASSERT_NE(tuple_abs, nullptr);
    tuple_abs->CheckAndConvertToDynamicLenSequence();
  }

  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);
  DoFuncImplInferAndCompare<TransposeFuncImpl>(kNameTranspose, {x, perms}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestTranspose, TestTranspose,
  testing::Values(
    TransParams{false, {2, 3, 4, 5}, kFloat32, CreateTuple({I64(2), I64(1), I64(-4), I64(-1)}), {4, 3, 2, 5}},
    TransParams{false, {2, 3, -1, 5}, kFloat32, CreateTuple({I64(2), I64(1), I64(-4), I64(-1)}), {-1, 3, 2, 5}},
    TransParams{true, {-2}, kFloat32, CreateTuple({kValueAny}), {-2}},
    TransParams{false, {2, -1, 4, -1}, kFloat32, CreateTuple({kValueAny, I64(0), kValueAny, I64(2)}), {-1, 2, -1, 4}},
    TransParams{false, {2, 3, 4, 5}, kFloat32, CreateTuple({I64(1), I64(0), kValueAny, kValueAny}), {3, 2, -1, -1}},
    TransParams{true, {2, 3, 4, 5}, kFloat32, CreateTuple({kValueAny}), {-1, -1, -1, -1}},
    TransParams{false, {2, 3, -1, 5}, kFloat32, CreateTuple({I64(2), kValueAny, I64(0), kValueAny}), {-1, -1, 2, -1}},
    TransParams{true, {2, 3, -1, 5}, kFloat32, CreateTuple({kValueAny}), {-1, -1, -1, -1}},
    TransParams{false, {-2}, kFloat32, CreateTuple({I64(3), I64(2), I64(1), I64(0)}), {-1, -1, -1, -1}},
    TransParams{false, {-2}, kFloat32, CreateTuple({I64(3), kValueAny, kValueAny, I64(0)}), {-1, -1, -1, -1}}));
}  // namespace mindspore::ops
