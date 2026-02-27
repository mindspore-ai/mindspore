/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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
#include "abstract/ops/primitive_infer_map.h"
#include "common/common_test.h"
#include "ir/primitive.h"
#include "ir/tensor_new.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "infer/ops_func_impl/equal.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/common/pass_manager/helper.h"

namespace mindspore {
namespace ops {
class TestEqual : public TestOps,
                  public testing::WithParamInterface<std::tuple<BroadcastOpShapeParams, BroadcastOpTypeParams>> {};

TEST_P(TestEqual, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  EqualFuncImpl equal_infer_obj;
  auto prim = std::make_shared<Primitive>("Equal");

  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(dtype_param.out_type, shape_param.out_shape);

  auto out_shape = equal_infer_obj.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
  auto out_dtype = equal_infer_obj.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
}

auto EqualOpTypeCases = testing::ValuesIn({
  BroadcastOpTypeParams{kBool, kBool, kBool},
  BroadcastOpTypeParams{kInt8, kInt8, kBool},
  BroadcastOpTypeParams{kInt16, kInt16, kBool},
  BroadcastOpTypeParams{kInt32, kInt32, kBool},
  BroadcastOpTypeParams{kInt64, kInt64, kBool},
  BroadcastOpTypeParams{kUInt8, kUInt8, kBool},
  BroadcastOpTypeParams{kUInt16, kUInt16, kBool},
  BroadcastOpTypeParams{kUInt32, kUInt32, kBool},
  BroadcastOpTypeParams{kFloat16, kFloat16, kBool},
  BroadcastOpTypeParams{kFloat32, kFloat32, kBool},
  BroadcastOpTypeParams{kFloat64, kFloat64, kBool},
  BroadcastOpTypeParams{kComplex64, kComplex64, kBool},
  BroadcastOpTypeParams{kComplex128, kComplex128, kBool},
});

INSTANTIATE_TEST_CASE_P(TestEqual_1_TensorScalar, TestEqual,
                        testing::Combine(BroadcastOpShapeTensorScalarCases, EqualOpTypeCases));
INSTANTIATE_TEST_CASE_P(TestEqual_1_ScalarTensor, TestEqual,
                        testing::Combine(BroadcastOpShapeScalarTensorCases, EqualOpTypeCases));
INSTANTIATE_TEST_CASE_P(TestEqual_2_TensorTensor, TestEqual,
                        testing::Combine(BroadcastOpShapeTensorTensorCases, EqualOpTypeCases));

struct EqualInferValueParams {
  ShapeVector x_shape;
  TypeId x_type;
  std::vector<float> x_f32;
  std::vector<int32_t> x_i32;
  std::vector<float16> x_f16;
  ShapeVector y_shape;
  TypeId y_type;
  std::vector<float> y_f32;
  std::vector<int32_t> y_i32;
  std::vector<float16> y_f16;
  std::vector<bool> out_data;
  bool expect_null{false};
};

class TestEqualInferValue : public TestOps, public testing::WithParamInterface<EqualInferValueParams> {};

static tensor::TensorPtr MakeTensorForEqual(const EqualInferValueParams &p, bool is_x) {
  const auto &shape = is_x ? p.x_shape : p.y_shape;
  const auto type = is_x ? p.x_type : p.y_type;
  if (type == kNumberTypeFloat32) {
    auto &buf = is_x ? p.x_f32 : p.y_f32;
    return tensor::from_buffer(type, shape, (void *)&buf[0], type);
  }
  if (type == kNumberTypeInt32) {
    auto &buf = is_x ? p.x_i32 : p.y_i32;
    return tensor::from_buffer(type, shape, (void *)&buf[0], type);
  }
  if (type == kNumberTypeFloat16) {
    auto &buf = is_x ? p.x_f16 : p.y_f16;
    return tensor::from_buffer(type, shape, (void *)&buf[0], type);
  }
  return nullptr;
}

TEST_P(TestEqualInferValue, infer_value_cases) {
  const auto &p = GetParam();
  auto x_tensor = MakeTensorForEqual(p, true);
  auto y_tensor = MakeTensorForEqual(p, false);
  ASSERT_NE(x_tensor, nullptr);
  ASSERT_NE(y_tensor, nullptr);

  auto x_abs = x_tensor->ToAbstract();
  auto y_abs = y_tensor->ToAbstract();
  ASSERT_NE(x_abs, nullptr);
  ASSERT_NE(y_abs, nullptr);

  auto primitive = std::make_shared<Primitive>("Equal");
  ASSERT_NE(primitive, nullptr);
  auto input_args = abstract::AbstractBasePtrList{x_abs, y_abs};
  auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);
  ASSERT_TRUE(value_opt.has_value());
  auto value = value_opt.value();
  if (p.expect_null) {
    ASSERT_EQ(value, nullptr);
    return;
  }
  ASSERT_NE(value, nullptr);
  auto out_tensor = value->cast<tensor::TensorPtr>();
  ASSERT_NE(out_tensor, nullptr);

  auto *out = static_cast<bool *>(out_tensor->data_c());
  for (size_t i = 0; i < p.out_data.size(); ++i) {
    ASSERT_EQ(out[i], p.out_data[i]);
  }
}

INSTANTIATE_TEST_CASE_P(
  TestEqualInferValue, TestEqualInferValue,
  testing::Values(
    // Mixed dtype: int32 vs float32 -> promote to float32 then compare
    EqualInferValueParams{ShapeVector{4},
                          kNumberTypeInt32,
                          {},
                          {1, 2, 3, 4},
                          {},
                          ShapeVector{4},
                          kNumberTypeFloat32,
                          {1.0f, 0.0f, 3.0f, 5.0f},
                          {},
                          {},
                          {true, false, true, false}},
    // Mixed dtype: float16 vs float32 -> promote then compare
    EqualInferValueParams{ShapeVector{2},
                          kNumberTypeFloat16,
                          {},
                          {},
                          {float16(1.0f), float16(2.0f)},
                          ShapeVector{2},
                          kNumberTypeFloat32,
                          {1.0f, 3.0f},
                          {},
                          {},
                          {true, false}},
    // Shape mismatch -> should not fold
    EqualInferValueParams{ShapeVector{2},
                          kNumberTypeInt32,
                          {},
                          {1, 2},
                          {},
                          ShapeVector{3},
                          kNumberTypeInt32,
                          {},
                          {1, 2, 3},
                          {},
                          {},
                          true}));
}  // namespace ops
}  // namespace mindspore
