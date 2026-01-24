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
#include "op_enum.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "ir/tensor_new.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_divmod_params() {
  GeneralInferParamGenerator generator;

  // Floor DivMod: 7 // 3 = 2, 7 % 3 = 1
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2}, kNumberTypeInt32}, InferInfoParam{ShapeVector{2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::FLOOR))}})
    .FeedExpectedOutput({{2}}, {kNumberTypeInt32});

  // Trunc DivMod: -7 // 3 = -2, -7 % 3 = -1
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::TRUNC))}})
    .FeedExpectedOutput({{2}}, {kNumberTypeFloat32});

  // Broadcast
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{3}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::FLOOR))}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(DivMod, GeneralInferTest, testing::ValuesIn(prepare_divmod_params()));

// Use REGISTER_TERNARY_OP_INFER_VALUE_TEST from the framework
REGISTER_TERNARY_OP_INFER_VALUE_TEST(TestDivModInferValue, InferValue);

// Helper
namespace {
template <typename T>
tensor::TensorPtr CreateDivModTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}
}  // namespace

INSTANTIATE_TEST_CASE_P(
  DivModInferValue, TestDivModInferValue,
  testing::Values(
    // Floor: 10, 20 with 3, 3 -> 3, 6 (quot); 1, 2 (rem)
    TernaryOpInferValueParams{
      "DivMod", CreateDivModTensor<int32_t>(kNumberTypeInt32, {2}, {10, 20})->ToAbstract()->BuildValue(),
      CreateDivModTensor<int32_t>(kNumberTypeInt32, {2}, {3, 3})->ToAbstract()->BuildValue(),
      CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::FLOOR))->ToAbstract()->BuildValue(),
      CreateDivModTensor<int32_t>(kNumberTypeInt32, {2}, {3, 6})},
    // Trunc: -10, -20 with 3, 3 -> -3, -6 (quot); -1, -2 (rem)
    TernaryOpInferValueParams{
      "DivMod", CreateDivModTensor<float>(kNumberTypeFloat32, {2}, {-10.0, -20.0})->ToAbstract()->BuildValue(),
      CreateDivModTensor<float>(kNumberTypeFloat32, {2}, {3.0, 3.0})->ToAbstract()->BuildValue(),
      CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::TRUNC))->ToAbstract()->BuildValue(),
      CreateDivModTensor<float>(kNumberTypeFloat32, {2}, {-3.0, -6.0})},
    // Mixed: F32 / F16 -> F32 (Trunc)
    TernaryOpInferValueParams{
      "DivMod", CreateDivModTensor<float>(kNumberTypeFloat32, {2}, {-10.0, -20.0})->ToAbstract()->BuildValue(),
      CreateDivModTensor<float16>(kNumberTypeFloat16, {2}, {float16(3.0), float16(3.0)})->ToAbstract()->BuildValue(),
      CreateScalar<int64_t>(static_cast<int64_t>(RoundingMode::TRUNC))->ToAbstract()->BuildValue(),
      CreateDivModTensor<float>(kNumberTypeFloat32, {2}, {-3.0, -6.0})}));

}  // namespace mindspore::ops
