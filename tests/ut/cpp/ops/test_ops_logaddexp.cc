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
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "ops/test_ops_cmp_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/logaddexp.h"
#include "ops/test_value_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
class TestLogAddExp : public TestOps, public testing::WithParamInterface<BroadcastOpParams> {};

TEST_P(TestLogAddExp, logaddexp_dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto other = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(other, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(input), std::move(other)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_shape, nullptr);
  ASSERT_NE(expect_type, nullptr);
  DoFuncImplInferAndCompare<LogAddExpFuncImpl>(kNameLogAddExp, input_args, expect_shape, expect_type);

  // simple infer
  auto input_val = tensor::from_spec(param.x_type->type_id(), param.x_shape, device::DeviceType::kCPU);
  auto other_val = tensor::from_spec(param.y_type->type_id(), param.y_shape, device::DeviceType::kCPU);
  DoFuncImplSimpleInferAndCompare<LogAddExpFuncImpl>(
      kNameLogAddExp, {input_val, other_val}, {param.out_shape}, {param.out_type});
};

INSTANTIATE_TEST_CASE_P(
  TestLogAddExp, TestLogAddExp,
  testing::Values(BroadcastOpParams{{2, 1}, kFloat32, {1, 1, 4}, kFloat32, {1, 2, 4}, kFloat32},
                  BroadcastOpParams{{-1, 3}, kFloat32, {-1, 1}, kFloat32, {-1, 3}, kFloat32},
                  BroadcastOpParams{{-1, -1}, kFloat32, {-1, -1, -1}, kFloat32, {-1, -1, -1}, kFloat32},
                  BroadcastOpParams{{-1, 1, 4}, kFloat32, {1, -1, 4}, kFloat32, {-1, -1, 4}, kFloat32},
                  BroadcastOpParams{{-1, 2, 3}, kFloat32, {2, -1, 3}, kFloat32, {2, 2, 3}, kFloat32},
                  BroadcastOpParams{{-2}, kFloat32, {2, 3}, kFloat32, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
