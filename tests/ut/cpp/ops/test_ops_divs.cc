/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tests/ut/cpp/ops/test_binary_ops.h"
#include "ops/utils/general_infer_utils.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "ir/tensor_new.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{1, 2, 3, 4, 5}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.)},
    })
    .FeedExpectedOutput({{1, 2, 3, 4, 5}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-1, -1, -1, -1, -1, -1}, kNumberTypeInt64},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1, -1, -1}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Divs, GeneralInferTest, testing::ValuesIn(prepare_params()));

// Use REGISTER_BINARY_OP_INFER_VALUE_TEST from the framework
REGISTER_BINARY_OP_INFER_VALUE_TEST(TestDivsInferValue, InferValue);

template <typename T>
tensor::TensorPtr CreateTensorPtr(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}

INSTANTIATE_TEST_CASE_P(
  DivsInferValue, TestDivsInferValue,
  testing::Values(
    BinaryOpInferValueParams{
      "Divs",
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{2, 4, 6.8, 8})
        ->ToAbstract()
        ->BuildValue(),
      CreatePyInt(2)->ToAbstract()->BuildValue(),
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, 2, 3.4, 4})},
    BinaryOpInferValueParams{
      "Divs",
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{2, 4, 6.8, 8})
        ->ToAbstract()
        ->BuildValue(),
      CreatePyInt(2)->ToAbstract()->BuildValue(),
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{1, 2, 3.4, 4})},
    // Int/Int -> Float (True Div)
    BinaryOpInferValueParams{
      "Divs",
      CreateTensorPtr<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{5, 10, 6, 8})
        ->ToAbstract()
        ->BuildValue(),
      CreatePyInt(2)->ToAbstract()->BuildValue(),
      // Generic ConstInfer returns Float64 for integer division to preserve precision.
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{2.5, 5.0, 3.0, 4.0})}));
}  // namespace mindspore::ops
