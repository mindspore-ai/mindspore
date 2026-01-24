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
std::vector<GeneralInferParam> prepare_div_params() {
  GeneralInferParamGenerator generator;
  // Same shape, Float32
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  // Broadcast: {2, 1} / {3} -> {2, 3}
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 1}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  // True Division: Int32 / Int32 -> Float32
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4}, kNumberTypeInt32}, InferInfoParam{ShapeVector{4}, kNumberTypeInt32}})
    .FeedExpectedOutput({{4}}, {kNumberTypeFloat32});

  // Mixed Precision: Float16 / Float32 -> Float32
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{4}, kNumberTypeFloat16}, InferInfoParam{ShapeVector{4}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{4}}, {kNumberTypeFloat32});

  // Dynamic shape
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-1, 3}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Div, GeneralInferTest, testing::ValuesIn(prepare_div_params()));

// Use REGISTER_BINARY_OP_INFER_VALUE_TEST from the framework
REGISTER_BINARY_OP_INFER_VALUE_TEST(TestDivInferValue, InferValue);

// Helper
namespace {
template <typename T>
tensor::TensorPtr CreateDivTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}
}  // namespace

INSTANTIATE_TEST_CASE_P(
  DivInferValue, TestDivInferValue,
  testing::Values(
    // 6.0 / 2.0 = 3.0
    BinaryOpInferValueParams{"Div",
                             CreateDivTensor<float>(kNumberTypeFloat32, {2}, {6.0, 4.0})->ToAbstract()->BuildValue(),
                             CreateDivTensor<float>(kNumberTypeFloat32, {2}, {2.0, 4.0})->ToAbstract()->BuildValue(),
                             CreateDivTensor<float>(kNumberTypeFloat32, {2}, {3.0, 1.0})},
    // 5 / 2 -> 2.5 (True Div)
    // Note: Generic InferValue returns Float64 for integer division to preserve precision.
    BinaryOpInferValueParams{"Div",
                             CreateDivTensor<int32_t>(kNumberTypeInt32, {2}, {5, 10})->ToAbstract()->BuildValue(),
                             CreateDivTensor<int32_t>(kNumberTypeInt32, {2}, {2, 2})->ToAbstract()->BuildValue(),
                             CreateDivTensor<double>(kNumberTypeFloat64, {2}, {2.5, 5.0})},
    // Mixed: F32 / F16 -> F32
    BinaryOpInferValueParams{
      "Div", CreateDivTensor<float>(kNumberTypeFloat32, {2}, {6.0, 4.0})->ToAbstract()->BuildValue(),
      CreateDivTensor<float16>(kNumberTypeFloat16, {2}, {float16(2.0), float16(4.0)})->ToAbstract()->BuildValue(),
      CreateDivTensor<float>(kNumberTypeFloat32, {2}, {3.0, 1.0})}));

}  // namespace mindspore::ops
