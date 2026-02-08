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
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/tril_ext.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct TrilExtShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr diagonal;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestTrilExt : public TestOps, public testing::WithParamInterface<TrilExtShapeParams> {};

TEST_P(TestTrilExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto diagonal = param.diagonal->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  TrilExtFuncImpl tril_ext_func_impl;
  auto prim = std::make_shared<Primitive>("TrilExt");
  auto out_dtype = tril_ext_func_impl.InferType(prim, {x, diagonal});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = tril_ext_func_impl.InferShape(prim, {x, diagonal});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestTrilExt, TestTrilExt,
  testing::Values(TrilExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kFloat32},
                  TrilExtShapeParams{{3, 4, 5}, kUInt8, CreateScalar<int64_t>(0), {3, 4, 5}, kUInt8},
                  TrilExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-2), {3, 4, 5}, kInt64},
                  TrilExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(2), {2, 3, 4, 5}, kInt32},
                  TrilExtShapeParams{{-1, -1, -1}, kBool, CreateScalar<int64_t>(1), {-1, -1, -1}, kBool},
                  TrilExtShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), {-2}, kFloat32}));

class TestTrilExtSimpleInfer : public TestOps, public testing::WithParamInterface<TrilExtShapeParams> {};
TEST_P(TestTrilExtSimpleInfer, simple_infer) {
  const auto &param = GetParam();
  auto x = tensor::from_spec(param.x_type->type_id(), param.x_shape, device::DeviceType::kCPU);
  TrilExtFuncImpl tril_ext_func_impl;
  auto prim = std::make_shared<Primitive>("TrilExt");
  ASSERT_NE(prim, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.diagonal));

  auto expect_shape = ShapeArray{param.out_shape};
  auto expect_type = TypePtrList{param.out_type};

  auto output_shape = tril_ext_func_impl.InferShape(prim, input_values);
  auto output_type = tril_ext_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestTrilExtSimpleInfer, TestTrilExtSimpleInfer,
  testing::Values(TrilExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kFloat32},
                  TrilExtShapeParams{{3, 4, 5}, kUInt8, CreateScalar<int64_t>(0), {3, 4, 5}, kUInt8},
                  TrilExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-2), {3, 4, 5}, kInt64},
                  TrilExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(2), {2, 3, 4, 5}, kInt32}));

}  // namespace ops
}  // namespace mindspore
