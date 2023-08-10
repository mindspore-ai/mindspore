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
#include <string>
#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ops/ops_func_impl/op_func_impl.h"

#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_

namespace mindspore {
namespace ops {
void ShapeCompare(const abstract::BaseShapePtr &output, const abstract::BaseShapePtr &expect);
void TypeCompare(const TypePtr &output, const TypePtr &expect);

template <typename T, typename std::enable_if<std::is_base_of<OpFuncImpl, T>::value>::type * = nullptr>
void DoFuncImplInferAndCompare(const std::string &prim_name, const abstract::AbstractBasePtrList &input_args,
                               const abstract::BaseShapePtr &expect_shape, const TypePtr &expect_type) {
  auto infer_impl = std::make_shared<T>();
  ASSERT_NE(infer_impl, nullptr);
  auto prim = std::make_shared<Primitive>(prim_name);
  auto inferred_shape = infer_impl->InferShape(prim, input_args);
  auto inferred_type = infer_impl->InferType(prim, input_args);
  ShapeCompare(inferred_shape, expect_shape);
  TypeCompare(inferred_type, expect_type);
}

void TestOpFuncImplWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                       const EltwiseOpParams &param);
void TestOpFuncImplWithMultiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                          const MultiInputOpParams &param);
#define OP_FUNC_IMPL_TEST_DECLARE(op_name, param_name)                                           \
  class Test##op_name : public TestOps, public testing::WithParamInterface<param_name> {};       \
  TEST_P(Test##op_name, op_name##_DynamicShape) {                                                \
    TestOpFuncImplWith##param_name(std::make_shared<op_name##FuncImpl>(), #op_name, GetParam()); \
  }

#define OP_FUNC_IMPL_TEST_CASES(op_name, cases) INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, Test##op_name, cases);

static auto eltwise_op_default_cases = testing::Values(
  EltwiseOpParams{{2, 3}, kFloat16, {2, 3}, kFloat16}, EltwiseOpParams{{2, -1}, kFloat16, {2, -1}, kFloat16},
  EltwiseOpParams{{-1, -1}, kFloat16, {-1, -1}, kFloat16}, EltwiseOpParams{{-2}, kFloat16, {-2}, kFloat16});

#define ELTWISE_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, EltwiseOpParams)         \
  OP_FUNC_IMPL_TEST_CASES(op_name, eltwise_op_default_cases);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
