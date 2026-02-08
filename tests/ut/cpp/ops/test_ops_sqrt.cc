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
#include <cmath>
#include <vector>
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "infer/ops_func_impl/sqrt.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
class TestSqrt : public TestOps,
                 public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestSqrt, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  DoFuncImplInferAndCompare<SqrtFuncImpl>(kNameSqrt, {x}, expect_shape, expect_type);
}

namespace {
auto SqrtOpTypeCases = testing::ValuesIn({
  EltwiseOpTypeParams{kInt8, kFloat32},
  EltwiseOpTypeParams{kInt16, kFloat32},
  EltwiseOpTypeParams{kInt32, kFloat32},
  EltwiseOpTypeParams{kInt64, kFloat32},
  EltwiseOpTypeParams{kUInt8, kFloat32},
  EltwiseOpTypeParams{kUInt16, kFloat32},
  EltwiseOpTypeParams{kUInt32, kFloat32},
  EltwiseOpTypeParams{kUInt64, kFloat32},
  EltwiseOpTypeParams{kFloat16, kFloat16},
  EltwiseOpTypeParams{kFloat32, kFloat32},
  EltwiseOpTypeParams{kFloat64, kFloat64},
  EltwiseOpTypeParams{kComplex64, kComplex64},
  EltwiseOpTypeParams{kComplex128, kComplex128},
  EltwiseOpTypeParams{kBool, kFloat32},
});
}

INSTANTIATE_TEST_CASE_P(TestSqrtGroup, TestSqrt, testing::Combine(EltwiseDynShapeTestCases, SqrtOpTypeCases));

struct SqrtInferValueParams {
  tensor::TensorPtr input;
  tensor::TensorPtr out;
};

class TestSqrtInferValue : public TestOps, public testing::WithParamInterface<SqrtInferValueParams> {};

TEST_P(TestSqrtInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimSqrt, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Sqrt have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Sqrt can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

#define SQRT_FP32(x) static_cast<float>(std::sqrt(static_cast<double>(x)))

tensor::TensorPtr CreateSqrtBoolTensor() {
  bool value[4] = {true, true, true, true};
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(kNumberTypeBool, ShapeVector{2, 2}, data_ptr, kNumberTypeBool);
  return tensor;
}

INSTANTIATE_TEST_CASE_P(
  TestSqrtInferValue, TestSqrtInferValue,
  testing::Values(
    SqrtInferValueParams{CreateSqrtBoolTensor(),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(1), SQRT_FP32(1), SQRT_FP32(1), SQRT_FP32(1)})},
    SqrtInferValueParams{CreateTensor<uint8_t>(kNumberTypeUInt8, ShapeVector{2, 2}, std::vector<uint8_t>{2, 2, 2, 2}),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(2), SQRT_FP32(2), SQRT_FP32(2), SQRT_FP32(2)})},
    SqrtInferValueParams{CreateTensor<int8_t>(kNumberTypeInt8, ShapeVector{2, 2}, std::vector<int8_t>{3, 3, 3, 3}),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(3), SQRT_FP32(3), SQRT_FP32(3), SQRT_FP32(3)})},
    SqrtInferValueParams{CreateTensor<int16_t>(kNumberTypeInt16, ShapeVector{2, 2}, std::vector<int16_t>{4, 4, 4, 4}),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(4), SQRT_FP32(4), SQRT_FP32(4), SQRT_FP32(4)})},
    SqrtInferValueParams{CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{5, 5, 5, 5}),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(5), SQRT_FP32(5), SQRT_FP32(5), SQRT_FP32(5)})},
    SqrtInferValueParams{CreateTensor<int64_t>(kNumberTypeInt64, ShapeVector{2, 2}, std::vector<int64_t>{6, 6, 6, 6}),
                         CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                                             std::vector<float>{SQRT_FP32(6), SQRT_FP32(6), SQRT_FP32(6), SQRT_FP32(6)})},
    SqrtInferValueParams{
      CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{7, 7, 7, 7}),
      CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                              std::vector<float>{SQRT_FP32(7), SQRT_FP32(7), SQRT_FP32(7), SQRT_FP32(7)})}));
}  // namespace ops
}  // namespace mindspore
