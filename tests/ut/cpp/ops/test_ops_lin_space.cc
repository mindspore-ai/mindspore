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
#include "ops/ops_func_impl/lin_space.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct LinSpaceParams {
  ShapeVector start_shape;
  TypePtr start_type;
  ShapeVector end_shape;
  TypePtr end_type;
  ValuePtr num;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestLinSpace : public TestOps, public testing::WithParamInterface<LinSpaceParams> {};

TEST_P(TestLinSpace, dyn_shape) {
  const auto &param = GetParam();
  auto start = std::make_shared<abstract::AbstractTensor>(param.start_type, param.start_shape);
  auto end = std::make_shared<abstract::AbstractTensor>(param.end_type, param.end_shape);
  auto num = param.num->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);

  LinSpaceFuncImpl lin_space_func_impl;
  auto prim = std::make_shared<Primitive>("LinSpace");

  auto out_dtype = lin_space_func_impl.InferType(prim, {start, end, num});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = lin_space_func_impl.InferShape(prim, {start, end, num});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

INSTANTIATE_TEST_CASE_P(
  TestLinSpace, TestLinSpace,
  testing::Values(LinSpaceParams{{}, kFloat64, {}, kFloat64, CreateScalar<int64_t>(3), {3}, kFloat64},
                  LinSpaceParams{{}, kFloat64, {}, kFloat64, CreateScalar(kValueAny), {-1}, kFloat64}));
}  // namespace ops
}  // namespace mindspore
