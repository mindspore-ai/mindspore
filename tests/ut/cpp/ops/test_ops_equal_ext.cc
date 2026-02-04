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
 * distributed under the License is distributed on an "AS IS" BASIS,s
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/utils/general_infer_utils.h"
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "infer/ops_func_impl/equal_ext.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 5, 6}, kNumberTypeInt8},
                    InferInfoParam{ShapeVector{2, 3, 5, 6}, kNumberTypeInt8}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeInt16},
                    InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeInt16}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{2, 3, 5}, kNumberTypeInt32}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 3}, kNumberTypeUInt8},
                    InferInfoParam{ShapeVector{2, 3, 3}, kNumberTypeUInt8}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat64}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16}})
    .FeedExpectedOutput({{1}}, {kNumberTypeBool});
  return generator.Generate();
}
}  // namespace

struct EqualExtInferValueParams {
  ShapeVector x_shape;
  TypeId x_type;
  std::vector<float> x_data;
  ShapeVector y_shape;
  TypeId y_type;
  std::vector<float> y_data;
  std::vector<bool> out_data;
};

class TestEqualExtInferValue : public TestOps, public testing::WithParamInterface<EqualExtInferValueParams> {};

TEST_P(TestEqualExtInferValue, dyn_shape_infer_value) {
  auto &param = GetParam();
  auto x_tensor = tensor::from_buffer(param.x_type, param.x_shape, (void *)&param.x_data[0], param.x_type);
  auto x = x_tensor->ToAbstract();
  ASSERT_NE(x, nullptr);
  auto y_tensor = tensor::from_buffer(param.y_type, param.y_shape, (void *)&param.y_data[0], param.y_type);
  auto y = y_tensor->ToAbstract();
  ASSERT_NE(y, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(y)};
  auto value_op = abstract::InferValueByFuncImpl(prim::kPrimEqualExt, input_args);
  ASSERT_TRUE(value_op.has_value());
  auto value = value_op.value();
  ASSERT_NE(value, nullptr);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  ASSERT_NE(value_tensor, nullptr);

  auto out = static_cast<bool *>(value_tensor->data_c());
  for (int i = 0; i < param.out_data.size(); i++) {
    ASSERT_TRUE(param.out_data[i] == out[i]);
  }
}

INSTANTIATE_TEST_CASE_P(
  TestEqualExtInferValue, TestEqualExtInferValue,
  testing::Values(EqualExtInferValueParams{ShapeVector{2, 2},
                                           kNumberTypeFloat32,
                                           {2, 2, 3, 3},
                                           ShapeVector{2, 2},
                                           kNumberTypeFloat32,
                                           {3, 3, 2, 2},
                                           {false}},
                  EqualExtInferValueParams{
                    ShapeVector{1}, kNumberTypeFloat32, {2}, ShapeVector{1}, kNumberTypeFloat32, {2}, {true}}));

INSTANTIATE_TEST_CASE_P(EqualExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
