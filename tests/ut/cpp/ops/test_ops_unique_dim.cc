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
#include "ops/ops_func_impl/unique_dim.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct UniqueDimParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr sorted;
  ValuePtr return_inverse;
  ValuePtr dim;
  ShapeVector output_shape;
  TypePtr output_type;
  ShapeVector inverse_indices_shape;
  TypePtr inverse_indices_type;
  ShapeVector counts_shape;
  TypePtr counts_type;
};

class TestUniqueDim : public TestOps, public testing::WithParamInterface<UniqueDimParams> {};

TEST_P(TestUniqueDim, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto sorted = param.sorted->ToAbstract();
  auto return_inverse = param.return_inverse->ToAbstract();
  auto dim = param.dim->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{std::make_shared<abstract::Shape>(param.output_shape),
                                        std::make_shared<abstract::Shape>(param.inverse_indices_shape),
                                        std::make_shared<abstract::Shape>(param.counts_shape)});
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(param.output_type),
                                                                   param.inverse_indices_type, param.counts_type});

  UniqueDimFuncImpl unique_dim_func_impl;
  auto prim = std::make_shared<Primitive>("UniqueDim");
  auto out_dtype = unique_dim_func_impl.InferType(prim, {x, sorted, return_inverse, dim});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = unique_dim_func_impl.InferShape(prim, {x, sorted, return_inverse, dim});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestUniqueDim, TestUniqueDim,
                        testing::Values(UniqueDimParams{{3, 4, 5},
                                                        kFloat32,
                                                        CreateScalar<bool>(true),
                                                        CreateScalar<bool>(true),
                                                        CreateScalar<int64_t>(0),
                                                        {3, 4, 5},
                                                        kFloat32,
                                                        {3},
                                                        kInt64,
                                                        {3},
                                                        kInt64},
                                        UniqueDimParams{{3, 4, 5},
                                                        kFloat32,
                                                        CreateScalar<bool>(false),
                                                        CreateScalar<bool>(true),
                                                        CreateScalar<int64_t>(1),
                                                        {3, 4, 5},
                                                        kFloat32,
                                                        {4},
                                                        kInt64,
                                                        {4},
                                                        kInt64},
                                        UniqueDimParams{{3, 4, 5},
                                                        kFloat32,
                                                        CreateScalar<bool>(true),
                                                        CreateScalar<bool>(false),
                                                        CreateScalar<int64_t>(2),
                                                        {3, 4, 5},
                                                        kFloat32,
                                                        {5},
                                                        kInt64,
                                                        {5},
                                                        kInt64}));
}  // namespace ops
}  // namespace mindspore
