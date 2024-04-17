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
#include "abstract/ops/primitive_infer_map.h"

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
  auto tensors = std::make_shared<abstract::AbstractTuple>(inputs);
  ASSERT_NE(tensors, nullptr);
  if (param.dynamic_len) {
    tensors->CheckAndConvertToDynamicLenSequence();
  }

  ASSERT_NE(param.axis, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ConcatFuncImpl>("Concat", abstract::AbstractBasePtrList{tensors, axis}, expect_shape,
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

class TestConcatException : public TestOps, public testing::WithParamInterface<ConcatParams> {};

TEST_P(TestConcatException, dyn_shape_exception) {
  const auto &param = GetParam();

  abstract::AbstractBasePtrList inputs;
  inputs.reserve(param.x_shapes.size());
  for (auto shape : param.x_shapes) {
    auto input = std::make_shared<abstract::AbstractTensor>(param.x_type, shape);
    ASSERT_NE(input, nullptr);
    inputs.push_back(input);
  }
  auto tensors = std::make_shared<abstract::AbstractTuple>(inputs);
  ASSERT_NE(tensors, nullptr);
  if (param.dynamic_len) {
    tensors->CheckAndConvertToDynamicLenSequence();
  }

  ASSERT_NE(param.axis, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  auto infer_impl = std::make_shared<ConcatFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto prim = std::make_shared<Primitive>("Concat");
  EXPECT_ANY_THROW(infer_impl->InferShape(prim, abstract::AbstractBasePtrList{tensors, axis}));
}

INSTANTIATE_TEST_CASE_P(
  TestConcatException, TestConcatException,
  testing::Values(ConcatParams{true, {{2, 3, 4}}, kFloat32, CreateScalar<int64_t>(4), {2, -1, 4}, kFloat32},
                  ConcatParams{false, {{3, 2, 4}, {3, 5, 4}}, kFloat32, CreateScalar<int64_t>(4), {3, 7, 4}, kFloat32},
                  ConcatParams{false, {{3, 2, 4}, {3, 5, 4}}, kFloat32, CreateScalar<int64_t>(-4), {3, 7, 4}, kFloat32},
                  ConcatParams{
                    false, {{3, 2, 4}, {3, 5, 4}}, kFloat32, CreateScalar<int64_t>(0), {3, 7, 4}, kFloat32}));

tensor::TensorPtr CreateTensor(const ShapeVector &shape, std::vector<float> value) {
  void *data_ptr = &value[0];
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, shape, data_ptr, kNumberTypeFloat32);
  return tensor;
}

struct ConcatInferValueParams {
  std::vector<tensor::TensorPtr> input_tensors;
  ValuePtr axis;
  tensor::TensorPtr out;
};

class TestConcatInferValue : public TestOps, public testing::WithParamInterface<ConcatInferValueParams> {};

TEST_P(TestConcatInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();

  auto input_tensors = param.input_tensors;
  AbstractBasePtrList input_elements;
  for (auto tensor : input_tensors) {
    ASSERT_NE(tensor, nullptr);
    auto x = tensor->ToAbstract();
    ASSERT_NE(x, nullptr);
    input_elements.push_back(x);
  }

  auto tensors = std::make_shared<abstract::AbstractTuple>(input_elements);
  ASSERT_NE(tensors, nullptr);

  ASSERT_NE(param.axis, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  abstract::AbstractBasePtrList input_args = {tensors, axis};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimConcat, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Tile have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Tile can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestConcatInferValue, TestConcatInferValue,
  testing::Values(ConcatInferValueParams{{CreateTensor(ShapeVector{2}, std::vector<float>{1, 2}),
                                          CreateTensor(ShapeVector{2}, std::vector<float>{3, 4})},
                                         CreateScalar<int64_t>(0),
                                         CreateTensor(ShapeVector{4}, std::vector<float>{1, 2, 3, 4})},
                  ConcatInferValueParams{
                    {CreateTensor(ShapeVector{2, 2}, std::vector<float>{1, 2, 3, 4}),
                     CreateTensor(ShapeVector{2, 3}, std::vector<float>{5, 6, 7, 8, 9, 10})},
                    CreateScalar<int64_t>(1),
                    CreateTensor(ShapeVector{2, 5}, std::vector<float>{1, 2, 5, 6, 7, 3, 4, 8, 9, 10})}));
}  // namespace mindspore::ops
