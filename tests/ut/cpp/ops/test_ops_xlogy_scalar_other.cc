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
#include "ops/ops_func_impl/xlogy_scalar_other.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {

struct XlogyScalarOtherParam {
  ValuePtr y;
  std::vector<int64_t> x_shape;
  TypePtr x_type;
  std::vector<int64_t> out_shape;
  TypePtr out_type;
};

class TestXlogyScalarOther : public TestOps, public testing::WithParamInterface<std::tuple<XlogyScalarOtherParam>> {};

TEST_P(TestXlogyScalarOther, xlogy_scalar_self_dyn_shape) {
  const auto &param = std::get<0>(GetParam());

  XLogYScalarOtherFuncImpl xlogy_scalar_other_func_impl;
  auto prim = std::make_shared<Primitive>("XLogYScalarOther");
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = param.y->ToAbstract();
  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.out_type);

  auto out_shape = xlogy_scalar_other_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = xlogy_scalar_other_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

class TestXlogyScalarOtherSimpleInfer : public TestOps,
                                        public testing::WithParamInterface<std::tuple<XlogyScalarOtherParam>> {};

TEST_P(TestXlogyScalarOtherSimpleInfer, simple_infer) {
  const auto &param = std::get<0>(GetParam());
  XLogYScalarOtherFuncImpl xlogy_scalar_other_func_impl;

  auto prim = std::make_shared<Primitive>("XLogYScalarOther");
  ASSERT_NE(prim, nullptr);

  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);
  ASSERT_NE(x, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.y));

  auto expect_shape = ShapeArray{param.out_shape};
  auto expect_type = TypePtrList{param.out_type};

  auto output_shape = xlogy_scalar_other_func_impl.InferShape(prim, input_values);
  auto output_type = xlogy_scalar_other_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto XlogyScalarOtherOpShapeTestCases =
  testing::ValuesIn({XlogyScalarOtherParam{CreateScalar<bool>(true), {10}, kFloat32, {10}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<bool>(true), {10, 1, 2}, kInt64, {10, 1, 2}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<float>(2.0), {10, 4, 2}, kInt64, {10, 4, 2}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<int>(2), {10, 1, -1}, kInt64, {10, 1, -1}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<int>(2), {10, 1, -1}, kFloat32, {10, 1, -1}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<bool>(false), {-2}, kInt64, {-2}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<float>(2.0), {}, kFloat32, {}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<bool>(true), {}, kInt64, {}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<int>(2), {}, kInt64, {}, kFloat32}});

auto XlogyScalarOtherOpSimpleInferShapeTestCases =
  testing::ValuesIn({XlogyScalarOtherParam{CreateScalar<bool>(true), {10}, kFloat32, {10}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<bool>(true), {10, 1, 2}, kInt64, {10, 1, 2}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<float>(2.0), {10, 4, 2}, kInt64, {10, 4, 2}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<float>(2.0), {}, kFloat32, {}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<bool>(true), {}, kInt64, {}, kFloat32},
                     XlogyScalarOtherParam{CreateScalar<int>(2), {}, kInt64, {}, kFloat32}});

INSTANTIATE_TEST_CASE_P(TestXlogyScalarOther, TestXlogyScalarOther, testing::Combine(XlogyScalarOtherOpShapeTestCases));
INSTANTIATE_TEST_CASE_P(TestXlogyScalarOtherSimpleInfer, TestXlogyScalarOtherSimpleInfer,
                        testing::Combine(XlogyScalarOtherOpSimpleInferShapeTestCases));
}  // namespace ops
}  // namespace mindspore
