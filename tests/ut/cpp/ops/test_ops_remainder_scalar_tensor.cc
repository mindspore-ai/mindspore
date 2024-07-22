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
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/remainder_scalar_tensor.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {

struct RemainderScalarTensorParam {
  ValuePtr x;
  std::vector<int64_t> y_shape;
  TypePtr y_type;
  std::vector<int64_t> out_shape;
  TypePtr out_type;
};

class TestRemainderScalarTensor :
  public TestOps,
  public testing::WithParamInterface<std::tuple<RemainderScalarTensorParam>> {};

TEST_P(TestRemainderScalarTensor, dyn_shape) {
  const auto &param = std::get<0>(GetParam());

  RemainderScalarTensorFuncImpl remainder_scalar_tensor_func_impl;
  auto prim = std::make_shared<Primitive>("RemainderScalarTensor");
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  auto x = param.x->ToAbstract();
  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.out_type);

  auto out_shape = remainder_scalar_tensor_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = remainder_scalar_tensor_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

class TestRemainderScalarTensorSimpleInfer :
  public TestOps,
  public testing::WithParamInterface<std::tuple<RemainderScalarTensorParam>> {};

TEST_P(TestRemainderScalarTensorSimpleInfer, simple_infer) {
  const auto &param = std::get<0>(GetParam());
  RemainderScalarTensorFuncImpl remainder_scalar_tensor_func_impl;

  auto prim = std::make_shared<Primitive>("RemainderScalarTensor");
  ASSERT_NE(prim, nullptr);

  auto y = std::make_shared<tensor::BaseTensor>(param.y_type->type_id(), param.y_shape);
  ASSERT_NE(y, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(param.x));
  input_values.push_back(std::move(y));

  auto expect_shape = ShapeArray{param.out_shape};
  auto expect_type = TypePtrList{param.out_type};

  auto output_shape = remainder_scalar_tensor_func_impl.InferShape(prim, input_values);
  auto output_type = remainder_scalar_tensor_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto RemainderScalarTensorOpTestCases =
  testing::ValuesIn({RemainderScalarTensorParam{CreateScalar<bool>(true), {10}, kFloat32, {10}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<bool>(true), {10, 1, 2}, kInt64, {10, 1, 2}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<float>(2.0), {10, 4, 2}, kInt64, {10, 4, 2}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<int>(2), {10, 1, -1}, kInt64, {10, 1, -1}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<int>(2), {10, 1, -1}, kFloat32, {10, 1, -1}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<bool>(false), {-2}, kInt64, {-2}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<float>(2.0), {}, kFloat32, {}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<bool>(true), {}, kInt64, {}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<int>(2), {}, kInt64, {}, kInt64}});

auto RemainderScalarTensorOpSimpleInferTestCases =
  testing::ValuesIn({RemainderScalarTensorParam{CreateScalar<bool>(true), {10}, kFloat32, {10}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<bool>(true), {10, 1, 2}, kInt64, {10, 1, 2}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<float>(2.0), {10, 4, 2}, kInt64, {10, 4, 2}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<float>(2.0), {}, kFloat32, {}, kFloat32},
                     RemainderScalarTensorParam{CreateScalar<bool>(true), {}, kInt64, {}, kInt64},
                     RemainderScalarTensorParam{CreateScalar<int>(2), {}, kInt64, {}, kInt64}});

INSTANTIATE_TEST_CASE_P(TestRemainderScalarTensor,
                        TestRemainderScalarTensor,
                        testing::Combine(RemainderScalarTensorOpTestCases));

INSTANTIATE_TEST_CASE_P(TestRemainderScalarTensorSimpleInfer,
                        TestRemainderScalarTensorSimpleInfer,
                        testing::Combine(RemainderScalarTensorOpSimpleInferTestCases));
}  // namespace ops
}  // namespace mindspore
