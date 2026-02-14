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

#include <vector>
#include <memory>
#include "ops/utils/general_infer_utils.h"
#include "common/common_test.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "abstract/dshape.h"
#include "ir/dtype/type.h"
#include "ir/tensor_new.h"
#include "ops/test_value_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "ops/op_def.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  // static shape
  // Case 1: Tensor(float32) < Scalar(int32), output shape should be same as Tensor
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar<int32_t>(3)}})
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeBool});

  // dynamic shape
  // Case 2: Tensor(dynamic_shape) < Scalar, output shape should be same as Tensor
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 3, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar<int32_t>(3)}})
    .FeedExpectedOutput({{-1, 3, -1}}, {kNumberTypeBool});

  // dynamic rank
  // Case 3: Tensor(dynamic_rank) < Scalar, output shape should be dynamic rank (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar<int32_t>(3)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeBool});

  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(LessScalar, GeneralInferTest, testing::ValuesIn(prepare_params()));

struct LessScalarInferValueParams {
  tensor::TensorPtr x;
  ScalarPtr y;
  tensor::TensorPtr out;
};

class TestLessScalarInferValue : public TestOps, public testing::WithParamInterface<LessScalarInferValueParams> {};

TEST_P(TestLessScalarInferValue, infer_value_cases) {
  const auto &param = GetParam();
  ASSERT_NE(param.x, nullptr);
  ASSERT_NE(param.y, nullptr);
  auto x_abs = param.x->ToAbstract();
  auto y_abs = param.y->ToAbstract();
  auto primitive = std::make_shared<Primitive>("LessScalar");
  ASSERT_NE(primitive, nullptr);

  auto input_args = abstract::AbstractBasePtrList{x_abs, y_abs};
  auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);
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

static tensor::TensorPtr CreateTensorI32(const ShapeVector &shape, const std::vector<int32_t> &vals) {
  return CreateTensor<int32_t>(kNumberTypeInt32, shape, vals);
}

static tensor::TensorPtr CreateTensorBool(const ShapeVector &shape, const std::vector<bool> &vals) {
  std::vector<uint8_t> int_vals(vals.begin(), vals.end());
  return CreateTensor<uint8_t>(kNumberTypeBool, shape, int_vals);
}

template <typename T>
static ScalarPtr CreateScalarVal(T val) {
  auto value_ptr = CreateScalar(val);
  return value_ptr->template cast<ScalarPtr>();
}

INSTANTIATE_TEST_CASE_P(TestLessScalarInferValue, TestLessScalarInferValue,
                        testing::Values(
                          // same type: Float32 < Float32(Scalar)
                          LessScalarInferValueParams{CreateTensorF32({2}, {1.0, 5.0}), CreateScalarVal<float>(3.0),
                                                     CreateTensorBool({2}, {true, false})},
                          // same type: Int32 < Int32(Scalar)
                          LessScalarInferValueParams{CreateTensorI32({2}, {1, 5}), CreateScalarVal<int32_t>(3),
                                                     CreateTensorBool({2}, {true, false})},
                          // mixed type: Float32 < Int32(Scalar)
                          LessScalarInferValueParams{CreateTensorF32({2}, {1.0, 5.0}), CreateScalarVal<int32_t>(3),
                                                     CreateTensorBool({2}, {true, false})},
                          // mixed type: Int32 < Float32(Scalar)
                          LessScalarInferValueParams{CreateTensorI32({2}, {1, 5}), CreateScalarVal<float>(3.0),
                                                     CreateTensorBool({2}, {true, false})}));

}  // namespace mindspore::ops
