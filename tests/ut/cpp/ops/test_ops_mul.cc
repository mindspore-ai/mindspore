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
#include "ops/test_ops.h"
#include "ops/utils/general_infer_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::ops {
namespace {
// InferShape & InferType verification using GeneralInferTest
// All scenarios migrated from test_ops_mul.cc
std::vector<GeneralInferParam> prepare_mul_params() {
  GeneralInferParamGenerator generator;
  // Cases from test_ops_mul.cc:
  // BroadcastOpParams{{2, 1}, kFloat32, {1, 1, 4}, kFloat32, {1, 2, 4}, kFloat32}
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 1}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{1, 1, 4}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{1, 2, 4}}, {kNumberTypeFloat32});

  // BroadcastOpParams{{-1, 3}, kFloat32, {-1, 1}, kFloat32, {-1, 3}, kFloat32}
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-1, 3}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{-1, 1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, 3}}, {kNumberTypeFloat32});

  // BroadcastOpParams{{-1, -1}, kFloat32, {-1, -1, -1}, kFloat32, {-1, -1, -1}, kFloat32}
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});

  // BroadcastOpParams{{-1, 1, 4}, kFloat32, {1, -1, 4}, kFloat32, {-1, -1, 4}, kFloat32}
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 1, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{1, -1, 4}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, -1, 4}}, {kNumberTypeFloat32});

  // BroadcastOpParams{{-1, 2, 3}, kFloat32, {2, -1, 3}, kFloat32, {2, 2, 3}, kFloat32}
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, -1, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 2, 3}}, {kNumberTypeFloat32});

  // BroadcastOpParams{{-2}, kFloat32, {2, 3}, kFloat32, {-2}, kFloat32}
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32}, InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Mul, GeneralInferTest, testing::ValuesIn(prepare_mul_params()));

REGISTER_BINARY_OP_INFER_VALUE_TEST(TestMulTempInferValue, InferValue);

// InferValue verification using TestMulTempInferValue
INSTANTIATE_TEST_CASE_P(
  MulTemp_InferValue, TestMulTempInferValue,
  testing::Values(
    // same type: F32
    BinaryOpInferValueParams{"Mul", CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {1.0, 2.0}),
                             CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {3.0, 4.0}),
                             CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {3.0, 8.0})},
    // same type: F16
    BinaryOpInferValueParams{"Mul",
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(1.0), float16(2.0)}),
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(3.0), float16(4.0)}),
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(3.0), float16(8.0)})},
    // same type: I32
    BinaryOpInferValueParams{"Mul", CreateBinaryOpTensor<int32_t>(kNumberTypeInt32, {2}, {1, 2}),
                             CreateBinaryOpTensor<int32_t>(kNumberTypeInt32, {2}, {3, 4}),
                             CreateBinaryOpTensor<int32_t>(kNumberTypeInt32, {2}, {3, 8})},
    // mixed type: F32 * I32 -> F32
    BinaryOpInferValueParams{"Mul", CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {1.0, 2.0}),
                             CreateBinaryOpTensor<int32_t>(kNumberTypeInt32, {2}, {3, 4}),
                             CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {3.0, 8.0})},
    // mixed type: F16 * I32 -> F16
    BinaryOpInferValueParams{"Mul",
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(1.0), float16(2.0)}),
                             CreateBinaryOpTensor<int32_t>(kNumberTypeInt32, {2}, {3, 4}),
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(3.0), float16(8.0)})},
    // mixed type: F32 * F16 -> F32
    BinaryOpInferValueParams{"Mul", CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {1.0, 2.0}),
                             CreateBinaryOpTensor<float16>(kNumberTypeFloat16, {2}, {float16(3.0), float16(4.0)}),
                             CreateBinaryOpTensor<float>(kNumberTypeFloat32, {2}, {3.0, 8.0})}));

}  // namespace mindspore::ops
