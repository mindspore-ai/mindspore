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
#include <vector>
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/ops_func_impl/not_equal.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
class TestNotEqual : public TestOps, public testing::WithParamInterface<BroadcastOpParams> {};

TEST_P(TestNotEqual, not_equal_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("NotEqual");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(y)};
  auto infer_impl = std::make_shared<NotEqualFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(TestNotEqualGroup, TestNotEqual,
                        testing::Values(BroadcastOpParams{{1, 3}, kFloat, {2, 1}, kFloat, {2, 3}, kBool},
                                        BroadcastOpParams{{-1, 3}, kFloat, {-1, 1}, kFloat, {-1, 3}, kBool},
                                        BroadcastOpParams{{-1, 1, 3}, kFloat, {1, -1, 3}, kFloat, {-1, -1, 3}, kBool},
                                        BroadcastOpParams{{-1, 2, 3}, kFloat, {2, -1, 3}, kFloat, {2, 2, 3}, kBool},
                                        BroadcastOpParams{{-2}, kFloat, {2, 3}, kFloat, {-2}, kBool}));

struct NotEqualInferValueParams {
  ShapeVector x_shape;
  TypeId x_type;
  std::vector<float> x_data;
  ShapeVector y_shape;
  TypeId y_type;
  std::vector<float> y_data;
  std::vector<bool> out_data;
};

class TestNotEqualInferValue : public TestOps, public testing::WithParamInterface<NotEqualInferValueParams> {};

TEST_P(TestNotEqualInferValue, not_equal_infer_value) {
  auto &param = GetParam();
  auto x_tensor = std::make_shared<tensor::Tensor>(param.x_type, param.x_shape, (void *)&param.x_data[0], param.x_type);
  auto x = x_tensor->ToAbstract();
  ASSERT_NE(x, nullptr);
  auto y_tensor = std::make_shared<tensor::Tensor>(param.y_type, param.y_shape, (void *)&param.y_data[0], param.y_type);
  auto y = y_tensor->ToAbstract();
  ASSERT_NE(y, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(y)};
  auto value_op = abstract::InferValueByFuncImpl(prim::kPrimNotEqual, input_args);
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

INSTANTIATE_TEST_CASE_P(TestNotEqualInferValue, TestNotEqualInferValue,
                        testing::Values(NotEqualInferValueParams{ShapeVector{2, 2},
                                                                 kNumberTypeFloat32,
                                                                 {2, 2, 3, 3},
                                                                 ShapeVector{2, 2},
                                                                 kNumberTypeFloat32,
                                                                 {3, 3, 2, 2},
                                                                 {true, true, true, true}},
                                        NotEqualInferValueParams{ShapeVector{1},
                                                                 kNumberTypeFloat32,
                                                                 {2},
                                                                 ShapeVector{1},
                                                                 kNumberTypeFloat32,
                                                                 {2},
                                                                 {false}}));
}  // namespace ops
}  // namespace mindspore
