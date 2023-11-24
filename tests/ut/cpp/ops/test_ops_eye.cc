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

#include "ops/test_ops.h"
#include "ops/ops_func_impl/eye.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct EyeParams {
  ValuePtr n;
  ValuePtr m;
  ValuePtr dtype;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestEye : public TestOps, public testing::WithParamInterface<EyeParams> {};

TEST_P(TestEye, dyn_shape) {
  const auto &param = GetParam();
  auto n = param.n->ToAbstract();
  auto m = param.m->ToAbstract();
  auto dtype = param.dtype->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);

  EyeFuncImpl eye_func_impl;
  auto prim = std::make_shared<Primitive>("Eye");

  auto out_dtype = eye_func_impl.InferType(prim, {n, m, dtype});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = eye_func_impl.InferShape(prim, {n, m, dtype});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

INSTANTIATE_TEST_CASE_P(
  TestEye, TestEye,
  testing::Values(
    EyeParams{CreatePyInt(6), CreatePyInt(8), CreatePyInt(kNumberTypeFloat64), {6, 8}, kFloat64},
    EyeParams{CreateScalar(kValueAny), CreatePyInt(8), CreatePyInt(kNumberTypeFloat16), {-1, 8}, kFloat16},
    EyeParams{CreatePyInt(6), CreateScalar(kValueAny), CreatePyInt(kNumberTypeComplex128), {6, -1}, kComplex128},
    EyeParams{CreateScalar(kValueAny), CreateScalar(kValueAny), CreatePyInt(kNumberTypeInt8), {-1, -1}, kInt8}));
}  // namespace ops
}  // namespace mindspore
