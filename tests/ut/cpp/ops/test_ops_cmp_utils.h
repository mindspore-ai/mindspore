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
#include "common/common_test.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/test_ops.h"
#include "utils/shape_utils.h"

#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_

namespace mindspore {
namespace ops {
void ShapeCompare(const abstract::BaseShapePtr &output, const abstract::BaseShapePtr &expect);
void ShapeCompare(const ShapeArray &output, const ShapeArray &expect);
void TypeCompare(const TypePtr &output, const TypePtr &expect);
void TypeCompare(const TypePtrList &output, const TypePtrList &expect);

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

template <typename T, typename std::enable_if<std::is_base_of<OpFuncImpl, T>::value>::type * = nullptr>
void DoFuncImplSimpleInferAndCompare(const std::string &prim_name, const ValuePtrList &input_values,
                                     const ShapeArray &expect_shape, const TypePtrList &expect_type) {
  for (const auto &value : input_values) {
    if (value->isa<tensor::BaseTensor>()) {
      const auto &in_tensor = value->cast<tensor::BaseTensorPtr>();
      ASSERT_NE(in_tensor, nullptr);
      const auto &shape = in_tensor->shape();
      if (IsDynamic(shape)) {
        return;
      }
    } else if (value->isa<ValueAny>() || (value->isa<ValueSequence>() && value->ContainsValueAny())) {
      return;
    }
  }
  auto infer_impl = std::make_shared<T>();
  ASSERT_NE(infer_impl, nullptr);
  auto prim = std::make_shared<Primitive>(prim_name);
  auto inferred_shape = infer_impl->InferShape(prim, input_values);
  auto inferred_type = infer_impl->InferType(prim, input_values);
  ShapeCompare(inferred_shape, expect_shape);
  TypeCompare(inferred_type, expect_type);
}

static inline std::pair<abstract::BaseShapePtr, TypePtr> MakeOutputTupleShapeAndType(
  const std::vector<ShapeVector> &shapes, const std::vector<TypePtr> &types) {
  std::vector<abstract::BaseShapePtr> shape_vec;
  for (const auto &shape : shapes) {
    shape_vec.emplace_back(std::move(std::make_shared<abstract::TensorShape>(shape)));
  }
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::move(shape_vec));
  std::vector<TypePtr> type_vec;
  for (const auto &type : types) {
    type_vec.emplace_back(std::move(std::make_shared<TensorType>(type)));
  }
  auto expect_type = std::make_shared<Tuple>(std::move(type_vec));
  return std::make_pair(expect_shape, expect_type);
}

void TestOpFuncImplWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                       const EltwiseOpParams &param);
void TestOpFuncImplWithMultiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                          const MultiInputOpParams &param);
void TestOpFuncImplSimpleInferWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                                  const EltwiseOpParams &param);
void TestOpFuncImplSimpleInferWithMultiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                                     const MultiInputOpParams &param);
void TestOpFuncImplInferWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                                  const EltwiseOpParams &param);
void TestOpFuncImplInferWithMultiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                                     const MultiInputOpParams &param);

#define OP_FUNC_IMPL_TEST_DECLARE(op_name, param_name)                                                      \
  class Test##op_name : public TestOps, public testing::WithParamInterface<param_name> {};                  \
  TEST_P(Test##op_name, op_name##_DynamicShape) {                                                           \
    TestOpFuncImplWith##param_name(std::make_shared<op_name##FuncImpl>(), #op_name, GetParam());            \
  }
#define OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(op_name, param_name)                                          \
  class TestSimpleInfer##op_name : public TestOps, public testing::WithParamInterface<param_name> {};       \
  TEST_P(TestSimpleInfer##op_name, op_name##_SimpleInfer) {                                                 \
    TestOpFuncImplSimpleInferWith##param_name(std::make_shared<op_name##FuncImpl>(), #op_name, GetParam()); \
  }
#define OP_FUNC_IMPL_INFER_TEST_DECLARE(op_name, param_name)                                          \
  class Test##op_name : public TestOps, public testing::WithParamInterface<param_name> {};       \
  TEST_P(Test##op_name, op_name##_Infer) {                                                 \
    TestOpFuncImplInferWith##param_name(std::make_shared<op_name##FuncImpl>(), #op_name, GetParam()); \
  }

#define OP_FUNC_IMPL_TEST_CASES(op_name, cases) INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, Test##op_name, cases);
#define OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(op_name, cases) INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestSimpleInfer##op_name, cases);
#define OP_FUNC_IMPL_INFER_TEST_CASES(op_name, cases) INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, Test##op_name, cases);

static auto eltwise_op_default_cases = testing::Values(
  EltwiseOpParams{{2, 3}, kFloat16, {2, 3}, kFloat16}, EltwiseOpParams{{2, -1}, kFloat16, {2, -1}, kFloat16},
  EltwiseOpParams{{-1, -1}, kFloat16, {-1, -1}, kFloat16}, EltwiseOpParams{{-2}, kFloat16, {-2}, kFloat16});

#define ELTWISE_OP_FUNC_IMPL_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, EltwiseOpParams)         \
  OP_FUNC_IMPL_TEST_CASES(op_name, eltwise_op_default_cases);

static auto binary_shape_equals_default_cases =
  testing::Values(MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, -1}, {2, 3}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, 3}, {2, -1}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, -1}, {-1, -1}}, {kFloat16, kFloat16}, {{2, -1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-1, -1}, {2, -1}}, {kFloat16, kFloat16}, {{2, -1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-1, -1}, {-1, -1}}, {kFloat16, kFloat16}, {{-1, -1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-2}, {-1, -1}}, {kFloat16, kFloat16}, {{-1, -1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-1, -1}, {-2}}, {kFloat16, kFloat16}, {{-1, -1}}, {kFloat16}, {}},
                  MultiInputOpParams{{{-2}, {-2}}, {kFloat16, kFloat16}, {{-2}}, {kFloat16}, {}});

#define BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(op_name) \
  OP_FUNC_IMPL_TEST_DECLARE(op_name, MultiInputOpParams)     \
  INSTANTIATE_TEST_CASE_P(Test##op_name, Test##op_name, binary_shape_equals_default_cases);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_CMP_UTILS_H_
