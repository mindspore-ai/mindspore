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

#include "include/mindapi/base/type_id.h"
#include "ops/utils/general_infer_utils.h"
#include "op_enum.h"
#include <vector>
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "tests/ut/cpp/ops/test_binary_ops.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::TRUNC)},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::TRUNC)},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{1, 2, 3, 4, 5, 6, 7}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::TRUNC)},
    })
    .FeedExpectedOutput({{1, 2, 3, 4, 5, 6, 7}}, {kNumberTypeInt32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::FLOOR)},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::FLOOR)},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{1, 2, 3, 4, 5, 6, 7}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(RoundingMode::FLOOR)},
    })
    .FeedExpectedOutput({{1, 2, 3, 4, 5, 6, 7}}, {kNumberTypeInt32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{1, 2, 3, 4, 5, 6, 7, 8}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      // rounding_mode
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
    })
    .FeedExpectedOutput({{1, 2, 3, 4, 5, 6, 7, 8}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(DivMods, GeneralInferTest, testing::ValuesIn(prepare_params()));

// Use REGISTER_TERNARY_OP_INFER_VALUE_TEST from the framework
REGISTER_TERNARY_OP_INFER_VALUE_TEST(TestDivmodsInferValue, dyn_shape_infer_value);

template <typename T>
tensor::TensorPtr CreateTensorPtr(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}

INSTANTIATE_TEST_CASE_P(
  TestDivmodsInferValue, TestDivmodsInferValue,
  testing::Values(
    TernaryOpInferValueParams{
      "DivMods",
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{2, -3, 2.4, 4})
        ->ToAbstract()
        ->BuildValue(),
      CreatePyInt(2)->ToAbstract()->BuildValue(), CreatePyInt(1)->ToAbstract()->BuildValue(),
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, -1, 1, 2})},
    TernaryOpInferValueParams{
      "DivMods",
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{2, -3, 2.4, 4})
        ->ToAbstract()
        ->BuildValue(),
      CreatePyInt(2)->ToAbstract()->BuildValue(), CreatePyInt(2)->ToAbstract()->BuildValue(),
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, -2, 1, 2})}));
}  // namespace mindspore::ops
