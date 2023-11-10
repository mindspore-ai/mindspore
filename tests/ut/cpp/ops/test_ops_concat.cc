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
#include "ops/ops_func_impl/concat.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
struct ConcatParams {
  bool dynamic_len;
  ShapeArray x_shapes;
  TypePtr x_type;
  ValuePtr axis;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestConcat : public TestOps, public testing::WithParamInterface<ConcatParams> {};

TEST_P(TestConcat, dyn_shape) {
  const auto &param = GetParam();

  abstract::AbstractBasePtrList inputs;
  inputs.reserve(param.x_shapes.size());
  for (auto shape : param.x_shapes) {
    auto input = std::make_shared<abstract::AbstractTensor>(param.x_type, shape);
    ASSERT_NE(input, nullptr);
    inputs.push_back(input);
  }
  auto x = std::make_shared<abstract::AbstractTuple>(inputs);
  ASSERT_NE(x, nullptr);
  if (param.dynamic_len) {
    x->CheckAndConvertToDynamicLenSequence();
  }

  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ConcatFuncImpl>("Concat", abstract::AbstractBasePtrList{x, axis}, expect_shape,
                                            expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestConcat, TestConcat,
  testing::Values(
    ConcatParams{false, {{3, 2, 4}, {3, 5, 4}}, kFloat32, CreateScalar<int64_t>(1), {3, 7, 4}, kFloat32},
    ConcatParams{false, {{3, -1, 5}, {-1, -1, -1}, {3, 4, -1}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
    ConcatParams{false, {{-2}, {-2}}, kFloat32, kValueAny, {-2}, kFloat32},
    ConcatParams{false, {{-2}, {-2}}, kFloat32, CreateScalar<int64_t>(1), {-2}, kFloat32},
    ConcatParams{true, {{2, 3, 4}}, kFloat32, CreateScalar<int64_t>(1), {2, -1, 4}, kFloat32},
    ConcatParams{true, {{2, 3, 4}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
    ConcatParams{true, {{-1, -1}}, kFloat32, CreateScalar<int64_t>(1), {-1, -1}, kFloat32},
    ConcatParams{true, {{-1, -1}}, kFloat32, kValueAny, {-1, -1}, kFloat32},
    ConcatParams{true, {{-2}}, kFloat32, CreateScalar<int64_t>(1), {-2}, kFloat32},
    ConcatParams{true, {{-2}}, kFloat32, kValueAny, {-2}, kFloat32},
    ConcatParams{false, {{3, 4, 5}, {3, 4, 5}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
    ConcatParams{false, {{3, 4, 5}, {-1, 4, 5}, {3, 4, -1}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
    ConcatParams{
      false, {{2, 3, 4}, {2, -1, -1}, {-1, -1, 5}}, kFloat32, CreateScalar<int64_t>(2), {2, 3, -1}, kFloat32},
    ConcatParams{false, {{-2}, {2, -1, -1}, {-1, 4, -1}}, kFloat32, CreateScalar<int64_t>(2), {2, 4, -1}, kFloat32},
    ConcatParams{false, {{-1, 6, 3}, {5, -1, 4}}, kFloat32, CreateScalar<int64_t>(2), {5, 6, 7}, kFloat32},
    ConcatParams{false, {{3, 4, 5}, {3, 4, 4}}, kFloat32, kValueAny, {3, 4, 9}, kFloat32}));
}  // namespace mindspore::ops
