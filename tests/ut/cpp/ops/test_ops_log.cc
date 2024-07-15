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
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/log.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
tensor::TensorPtr CreateLogTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = std::make_shared<tensor::Tensor>(type, shape, data_ptr, type);
  return tensor;
}

tensor::TensorPtr CreateLogBoolTensor() {
  bool value[4] = {true, true, true, true};
  void *data_ptr = &value[0];
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, ShapeVector{2, 2}, data_ptr, kNumberTypeBool);
  return tensor;
}
}  // namespace
OP_FUNC_IMPL_INFER_TEST_DECLARE(Log, EltwiseOpParams);
OP_FUNC_IMPL_INFER_TEST_CASES(
  Log,
  testing::Values(
    EltwiseOpParams{{2, 3}, kBool, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kUInt8, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kInt8, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kInt16, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kInt32, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kInt64, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {}},
    EltwiseOpParams{{2, -1}, kFloat32, {2, -1}, kFloat32, {}},
    EltwiseOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32, {}},
    EltwiseOpParams{{-2}, kFloat32, {-2}, kFloat32, {}}
  ));

struct LogInferValueParams {
  tensor::TensorPtr input;
  tensor::TensorPtr out;
};

class TestLogInferValue : public TestOps, public testing::WithParamInterface<LogInferValueParams> {};

TEST_P(TestLogInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimLog, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Log have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Log can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

#define LOG_FP32(x) static_cast<float>(std::log(static_cast<double>(x)))

INSTANTIATE_TEST_CASE_P(
  TestLogInferValue, TestLogInferValue,
  testing::Values(
    LogInferValueParams{
      CreateLogBoolTensor(),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(1), LOG_FP32(1), LOG_FP32(1), LOG_FP32(1)})},
    LogInferValueParams{
      CreateLogTensor<uint8_t>(kNumberTypeUInt8, ShapeVector{2, 2}, std::vector<uint8_t>{2, 2, 2, 2}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(2), LOG_FP32(2), LOG_FP32(2), LOG_FP32(2)})},
    LogInferValueParams{
      CreateLogTensor<int8_t>(kNumberTypeInt8, ShapeVector{2, 2}, std::vector<int8_t>{3, 3, 3, 3}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(3), LOG_FP32(3), LOG_FP32(3), LOG_FP32(3)})},
    LogInferValueParams{
      CreateLogTensor<int16_t>(kNumberTypeInt16, ShapeVector{2, 2}, std::vector<int16_t>{4, 4, 4, 4}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(4), LOG_FP32(4), LOG_FP32(4), LOG_FP32(4)})},
    LogInferValueParams{
      CreateLogTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{5, 5, 5, 5}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(5), LOG_FP32(5), LOG_FP32(5), LOG_FP32(5)})},
    LogInferValueParams{
      CreateLogTensor<int64_t>(kNumberTypeInt64, ShapeVector{2, 2}, std::vector<int64_t>{6, 6, 6, 6}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(6), LOG_FP32(6), LOG_FP32(6), LOG_FP32(6)})},
    LogInferValueParams{
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{7, 7, 7, 7}),
      CreateLogTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2},
                             std::vector<float>{LOG_FP32(7), LOG_FP32(7), LOG_FP32(7), LOG_FP32(7)})}
    ));
}  // namespace ops
}  // namespace mindspore
