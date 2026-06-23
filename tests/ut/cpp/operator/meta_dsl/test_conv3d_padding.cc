/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <vector>

#include "abstract/abstract_value.h"
#include "common/common_test.h"
#include "ir/primitive.h"
#include "tests/ut/cpp/ops/test_value_utils.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/conv3d_padding.h"

namespace mindspore::prim {
namespace {
abstract::AbstractBasePtr MakeIntTupleAbstract(const std::vector<int64_t> &values) {
  abstract::AbstractBasePtrList elements;
  elements.reserve(values.size());
  for (const auto value : values) {
    elements.push_back(std::make_shared<abstract::AbstractScalar>(value));
  }
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(elements);
  tuple_abs->set_value(ops::CreateScalarTuple<int64_t>(values));
  return tuple_abs;
}

abstract::AbstractBasePtr MakeDynamicIntTupleAbstract(size_t size) {
  abstract::AbstractBasePtrList elements;
  elements.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    elements.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64));
  }
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(elements);
  tuple_abs->set_value(kValueAny);
  return tuple_abs;
}
}  // namespace

/// Feature: Meta DSL Conv3DPadding input check.
/// Description: Dilation value may be unknown during infer.
/// Expectation: Check function should not dereference null optional value.
TEST(TestConv3DPaddingCheck, DynamicDilationValueAnyShouldNotThrow) {
  auto primitive = std::make_shared<Primitive>("Conv3DPadding");
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1, 2, 4, 4, 4});
  auto weight_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 2, 3, 3, 3});
  auto bias_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2});
  auto stride_abs = MakeIntTupleAbstract({1, 1, 1});
  auto padding_abs =
    std::make_shared<abstract::AbstractScalar>(ops::CreateScalar<int64_t>(static_cast<int64_t>(PadMode::SAME)));
  auto dilation_abs = MakeDynamicIntTupleAbstract(3);
  auto groups_abs = std::make_shared<abstract::AbstractScalar>(int64_t{1});
  AbstractBasePtrList input_args{input_abs, weight_abs, bias_abs, stride_abs, padding_abs, dilation_abs, groups_abs};

  EXPECT_NO_THROW(CheckConv3DPaddingInputs(primitive, input_args));
}

/// Feature: Meta DSL Conv3DPadding input check.
/// Description: Dynamic padding value should not bypass static group checks.
/// Expectation: Group mismatch is still rejected when padding is unknown during infer.
TEST(TestConv3DPaddingCheck, DynamicPaddingValueAnyShouldStillCheckGroups) {
  auto primitive = std::make_shared<Primitive>("Conv3DPadding");
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1, 4, 4, 4, 4});
  auto weight_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3, 3, 3, 3});
  auto bias_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2});
  auto stride_abs = MakeIntTupleAbstract({1, 1, 1});
  auto padding_abs = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  auto dilation_abs = MakeIntTupleAbstract({1, 1, 1});
  auto groups_abs = std::make_shared<abstract::AbstractScalar>(int64_t{2});
  AbstractBasePtrList input_args{input_abs, weight_abs, bias_abs, stride_abs, padding_abs, dilation_abs, groups_abs};

  EXPECT_THROW(CheckConv3DPaddingInputs(primitive, input_args), std::exception);
}
}  // namespace mindspore::prim
