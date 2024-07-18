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
#include "ops/ops_func_impl/unique2.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct Unique2Params {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr sorted;
  ValuePtr return_inverse;
  ValuePtr return_counts;
  ShapeVector output_shape;
  TypePtr output_type;
  ShapeVector inverse_indices_shape;
  TypePtr inverse_indices_type;
  ShapeVector counts_shape;
  TypePtr counts_type;
};

class TestUnique2 : public TestOps, public testing::WithParamInterface<Unique2Params> {};

TEST_P(TestUnique2, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto sorted = param.sorted->ToAbstract();
  auto return_inverse = param.return_inverse->ToAbstract();
  auto return_counts = param.return_counts->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{std::make_shared<abstract::Shape>(param.output_shape),
                                        std::make_shared<abstract::Shape>(param.inverse_indices_shape),
                                        std::make_shared<abstract::Shape>(param.counts_shape)});
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(param.output_type),
                                                                   param.inverse_indices_type, param.counts_type});

  Unique2FuncImpl unique2_func_impl;
  auto prim = std::make_shared<Primitive>("Unique2");
  auto out_dtype = unique2_func_impl.InferType(prim, {x, sorted, return_inverse, return_counts});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = unique2_func_impl.InferShape(prim, {x, sorted, return_inverse, return_counts});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

class TestUnique2SimpleInfer : public TestOps, public testing::WithParamInterface<Unique2Params> {};

TEST_P(TestUnique2SimpleInfer, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.sorted));
  input_values.push_back(std::move(param.return_inverse));
  input_values.push_back(std::move(param.return_counts));

  Unique2FuncImpl unique2_func_impl;
  auto prim = std::make_shared<Primitive>("Unique2");

  auto expect_shape = ShapeArray{param.output_shape, param.inverse_indices_shape, param.counts_shape};
  auto expect_type = TypePtrList{param.output_type, param.inverse_indices_type, param.counts_type};

  auto output_shape = unique2_func_impl.InferShape(prim, input_values);
  auto output_type = unique2_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestUnique2, TestUnique2,
                        testing::Values(Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(false),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(false),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(false),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {},
                                                      kInt64}));

INSTANTIATE_TEST_CASE_P(TestUnique2SimpleInfer, TestUnique2SimpleInfer,
                        testing::Values(Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(false),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(false),
                                                      CreateScalar<bool>(true),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64},
                                        Unique2Params{{3, 4, 5},
                                                      kFloat32,
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(true),
                                                      CreateScalar<bool>(false),
                                                      {60},
                                                      kFloat32,
                                                      {3, 4, 5},
                                                      kInt64,
                                                      {60},
                                                      kInt64}));
}  // namespace ops
}  // namespace mindspore
