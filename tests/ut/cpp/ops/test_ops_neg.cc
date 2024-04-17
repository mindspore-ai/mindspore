/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "abstract/ops/primitive_infer_map.h"
#include "common/common_test.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/ops_func_impl/neg.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(Neg, EltwiseOpParams);

OP_FUNC_IMPL_TEST_CASES(Neg, testing::Values(EltwiseOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32},
                                             EltwiseOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32},
                                             EltwiseOpParams{{-2}, kFloat32, {-2}, kFloat32}));

struct NegInferValueParams {
  ShapeVector x_shape;
  TypeId x_type;
  std::vector<float> x_data;
  std::vector<float> out_data;
};

class TestNegInferValue : public TestOps, public testing::WithParamInterface<NegInferValueParams> {};

TEST_P(TestNegInferValue, neg_infer_value) {
  auto &param = GetParam();
  auto x_tensor = std::make_shared<tensor::Tensor>(param.x_type, param.x_shape, (void *)&param.x_data[0], param.x_type);
  auto x = x_tensor->ToAbstract();
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
  auto value_op = abstract::InferValueByFuncImpl(prim::kPrimNeg, input_args);
  ASSERT_TRUE(value_op.has_value());
  auto value = value_op.value();
  ASSERT_NE(value, nullptr);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  ASSERT_NE(value_tensor, nullptr);

  auto out = static_cast<float *>(value_tensor->data_c());
  for (int i = 0; i < param.out_data.size(); i++) {
    ASSERT_TRUE(param.out_data[i] == out[i]);
  }
}

INSTANTIATE_TEST_CASE_P(
  TestNegInferValue, TestNegInferValue,
  testing::Values(NegInferValueParams{ShapeVector{2, 2}, kNumberTypeFloat32, {2, 2, 3, 3}, {-2, -2, -3, -3}},
                  NegInferValueParams{ShapeVector{1}, kNumberTypeFloat32, {2}, {-2}}));
}  // namespace ops
}  // namespace mindspore
