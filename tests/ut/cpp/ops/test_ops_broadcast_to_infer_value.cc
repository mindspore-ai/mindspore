/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "ops/utils/general_infer_utils.h"
#include <vector>
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore::ops {

struct BroadcastToInferValueParams {
  tensor::TensorPtr input;
  // The target shape is provided as a Python-style list/tuple of integers.
  ValuePtr shape_list;  // ValueList or ValueTuple with ints
  tensor::TensorPtr out;
  bool expect_throw{false};
};

class TestBroadcastToInferValue : public TestOps, public testing::WithParamInterface<BroadcastToInferValueParams> {};

TEST_P(TestBroadcastToInferValue, infer_value_cases) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input_abs = param.input->ToAbstract();
  auto shape_abs = param.shape_list->ToAbstract();

  auto input_args = abstract::AbstractBasePtrList{input_abs, shape_abs};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimBroadcastTo, input_args);
  ASSERT_TRUE(value_opt.has_value());
  auto infer_out = value_opt.value();
  ASSERT_NE(infer_out, nullptr);
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

static tensor::TensorPtr CreateTensorF32(const ShapeVector &shape, const std::vector<float> &vals) {
  return CreateTensor<float>(kNumberTypeFloat32, shape, vals);
}

INSTANTIATE_TEST_CASE_P(
  TestBroadcastToInferValue, TestBroadcastToInferValue,
  testing::Values(
    // 0) Input rank is shorter than target, missing leading axis is a concrete number
    BroadcastToInferValueParams{
      CreateTensorF32({3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
      CreatePyIntTuple({2, 3, 4}),
      CreateTensorF32({2, 3, 4},
                      // Two copies along the new leading axis
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
      false},

    // 1) Aligned axes contain -1: infer from input
    BroadcastToInferValueParams{
      CreateTensorF32({3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
      CreatePyIntTuple({-1, -1}),
      CreateTensorF32({3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
      false},

    // 2) Target shape contains zero: return empty tensor
    BroadcastToInferValueParams{
      CreateTensorF32({1, 4}, {0, 1, 2, 3}),
      CreatePyIntTuple({0, 4}),
      CreateTensorF32({0, 4}, {}),
      false},

    // 3) Mix of -1 and concrete numbers: (2,3,-1) with input (3,1) -> (2,3,1)
    BroadcastToInferValueParams{
      CreateTensorF32({3, 1}, {0, 1, 2}),
      CreatePyIntTuple({2, 3, -1}),
      CreateTensorF32({2, 3, 1}, {0, 1, 2, 0, 1, 2}),
      false}
  )
);

}  // namespace mindspore::ops
