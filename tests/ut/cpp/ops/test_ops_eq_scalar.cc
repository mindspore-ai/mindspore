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
#include <memory>
#include <tuple>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "common/common_test.h"
#include "infer/ops_func_impl/eq_scalar.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor_new.h"
#include "ops/test_value_utils.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"

namespace mindspore {
namespace ops {
class TestEqScalar : public TestOps,
                     public testing::WithParamInterface<std::tuple<BroadcastOpShapeParams, BroadcastOpTypeParams>> {};

TEST_P(TestEqScalar, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto prim = std::make_shared<Primitive>("EqScalar");
  ASSERT_NE(prim, nullptr);

  // x is tensor
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto x_info = std::make_unique<AbstractInferInfoAdapter>(x, "EqScalar", "input");

  // other is scalar (number)
  auto y = std::make_shared<abstract::AbstractScalar>(kValueAny, dtype_param.y_type);
  ASSERT_NE(y, nullptr);
  auto y_info = std::make_unique<AbstractInferInfoAdapter>(y, "EqScalar", "other");

  InferInfoPtrList input_infos;
  input_infos.push_back(std::move(x_info));
  input_infos.push_back(std::move(y_info));

  auto infer_impl = std::make_shared<EqScalarFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = std::make_shared<abstract::Shape>(infer_impl->InferShape(prim, input_infos)[0]);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = TypeIdToType(infer_impl->InferType(prim, input_infos)[0]);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(infer_type == dtype_param.out_type);
}

auto EqScalarOpTypeCases = testing::ValuesIn({
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
});

INSTANTIATE_TEST_CASE_P(TestEqScalar, TestEqScalar,
                        testing::Combine(BroadcastOpShapeTensorScalarCases, EqScalarOpTypeCases));

struct EqScalarInferValueParams {
  ShapeVector x_shape;
  TypeId x_type;
  std::vector<float> x_f32;
  std::vector<int32_t> x_i32;
  ShapeVector y_shape;  // should be scalar shape: {}
  TypeId y_type;
  ValuePtr y_value;
  std::vector<bool> out_data;
  bool expect_null{false};
};

class TestEqScalarInferValue : public TestOps, public testing::WithParamInterface<EqScalarInferValueParams> {};

static tensor::TensorPtr MakeTensorForEqScalar(const EqScalarInferValueParams &p) {
  if (p.x_type == kNumberTypeFloat32) {
    return tensor::from_buffer(p.x_type, p.x_shape, (void *)&p.x_f32[0], p.x_type);
  }
  if (p.x_type == kNumberTypeInt32) {
    return tensor::from_buffer(p.x_type, p.x_shape, (void *)&p.x_i32[0], p.x_type);
  }
  return nullptr;
}

TEST_P(TestEqScalarInferValue, infer_value_cases) {
  const auto &p = GetParam();
  auto x_tensor = MakeTensorForEqScalar(p);
  ASSERT_NE(x_tensor, nullptr);
  auto x_abs = x_tensor->ToAbstract();
  ASSERT_NE(x_abs, nullptr);

  auto y_abs = std::make_shared<abstract::AbstractScalar>(p.y_value, TypeIdToType(p.y_type));
  ASSERT_NE(y_abs, nullptr);

  auto primitive = std::make_shared<Primitive>("EqScalar");
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
  TestEqScalarInferValue, TestEqScalarInferValue,
  testing::Values(
    // int32 tensor == float scalar (mixed): compare under promoted/common type
    EqScalarInferValueParams{ShapeVector{4},
                             kNumberTypeInt32,
                             {},
                             {1, 2, 3, 4},
                             ShapeVector{},
                             kNumberTypeFloat32,
                             CreateScalar<float>(2.0f),
                             {false, true, false, false}},
    // float32 tensor == int scalar (mixed)
    EqScalarInferValueParams{ShapeVector{3},
                             kNumberTypeFloat32,
                             {1.0f, 2.0f, 2.0f},
                             {},
                             ShapeVector{},
                             kNumberTypeInt32,
                             CreateScalar<int32_t>(2),
                             {false, true, true}},
    // scalar value unknown -> should not fold
    EqScalarInferValueParams{ShapeVector{2},
                             kNumberTypeInt32,
                             {},
                             {1, 2},
                             ShapeVector{},
                             kNumberTypeInt32,
                             kValueAny,
                             {},
                             true}));
}  // namespace ops
}  // namespace mindspore
