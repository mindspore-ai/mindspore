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
#include "ops/ops_func_impl/lin_space_ext.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct LinSpaceExtParams {
  ValuePtr start;
  ValuePtr end;
  ValuePtr steps;
  ValuePtr dtype;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestLinSpaceExt : public TestOps, public testing::WithParamInterface<LinSpaceExtParams> {};
TEST_P(TestLinSpaceExt, dyn_shape) {
  const auto &param = GetParam();
  auto start = param.start->ToAbstract();
  auto end = param.end->ToAbstract();
  auto steps = param.steps->ToAbstract();
  auto dtype = param.dtype->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);

  LinSpaceExtFuncImpl lin_space_ext_func_impl;
  auto prim = std::make_shared<Primitive>("LinSpaceExt");

  auto out_dtype = lin_space_ext_func_impl.InferType(prim, {start, end, steps, dtype});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = lin_space_ext_func_impl.InferShape(prim, {start, end, steps, dtype});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

INSTANTIATE_TEST_CASE_P(TestLinSpaceExt, TestLinSpaceExt,
                        testing::Values(LinSpaceExtParams{CreateScalar<int64_t>(3),
                                                          CreateScalar<int64_t>(10),
                                                          CreateScalar<int64_t>(3),
                                                          CreatePyInt(kNumberTypeFloat32),
                                                          {3},
                                                          kFloat32},
                                        LinSpaceExtParams{CreateScalar<double>(-31.414),
                                                          CreateScalar<double>(3413.54598),
                                                          CreateScalar<int64_t>(123),
                                                          CreatePyInt(kNumberTypeFloat32),
                                                          {123},
                                                          kFloat32},
                                        LinSpaceExtParams{CreateScalar<float>(3.0123),
                                                          CreateScalar<float>(-10.011),
                                                          CreateScalar(kValueAny),
                                                          CreatePyInt(kNumberTypeFloat64),
                                                          {-1},
                                                          kFloat64}));

class TestLinSpaceExtSimpleInfer : public TestOps, public testing::WithParamInterface<LinSpaceExtParams> {};
TEST_P(TestLinSpaceExtSimpleInfer, simple_infer) {
  const auto &param = GetParam();
  LinSpaceExtFuncImpl lin_space_ext_func_impl;
  auto prim = std::make_shared<Primitive>("LinSpaceExt");
  ASSERT_NE(prim, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(param.start));
  input_values.push_back(std::move(param.end));
  input_values.push_back(std::move(param.steps));
  input_values.push_back(std::move(param.dtype));

  auto expect_shape = ShapeArray{param.output_shape};
  auto expect_type = TypePtrList{param.output_type};

  auto output_shape = lin_space_ext_func_impl.InferShape(prim, input_values);
  auto output_type = lin_space_ext_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}
INSTANTIATE_TEST_CASE_P(TestLinSpaceExtSimpleInfer, TestLinSpaceExtSimpleInfer,
                        testing::Values(LinSpaceExtParams{CreateScalar<int64_t>(3),
                                                          CreateScalar<int64_t>(10),
                                                          CreateScalar<int64_t>(3),
                                                          CreatePyInt(kNumberTypeFloat32),
                                                          {3},
                                                          kFloat32},
                                        LinSpaceExtParams{CreateScalar<double>(-31.414),
                                                          CreateScalar<double>(3413.54598),
                                                          CreateScalar<int64_t>(123),
                                                          CreatePyInt(kNumberTypeFloat32),
                                                          {123},
                                                          kFloat32},
                                        LinSpaceExtParams{CreateScalar<float>(3.0123),
                                                          CreateScalar<float>(-10.011),
                                                          CreateScalar(kValueAny),
                                                          CreatePyInt(kNumberTypeFloat64),
                                                          {-1},
                                                          kFloat64}));
}  // namespace ops
}  // namespace mindspore
