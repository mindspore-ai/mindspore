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
#include "ops/ops_func_impl/sort_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct SortExtShape {
  std::vector<int64_t> input_shape;
  ValuePtr dim;
  ValuePtr descending;
  ValuePtr stable;
  std::vector<int64_t> out_values_shape;
  std::vector<int64_t> out_indices_shape;
};

struct SortExtType {
  TypePtr input_type;
  TypePtr dim_type;
  TypePtr descending_type;
  TypePtr stable_type;
  TypePtr out_values_type;
  TypePtr out_indices_type;
};

class TestSortExt : public TestOps, public testing::WithParamInterface<std::tuple<SortExtShape, SortExtType>> {};

TEST_P(TestSortExt, SortExt_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  SortExtFuncImpl SortExt_func_impl;
  auto prim = std::make_shared<Primitive>("SortExt");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_type, shape_param.input_shape);
  std::vector<int64_t> empty_shape = {};
  auto dim = std::make_shared<abstract::AbstractTensor>(dtype_param.dim_type, empty_shape);
  auto descending = std::make_shared<abstract::AbstractTensor>(dtype_param.descending_type, empty_shape);
  auto stable = std::make_shared<abstract::AbstractTensor>(dtype_param.stable_type, empty_shape);
  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{std::make_shared<abstract::Shape>(shape_param.out_values_shape),
                                        std::make_shared<abstract::Shape>(shape_param.out_indices_shape)});
  auto expect_dtype = std::make_shared<Tuple>(
    std::vector<TypePtr>{std::make_shared<TensorType>(dtype_param.out_values_type), dtype_param.out_indices_type});

  auto out_shape =
    SortExt_func_impl.InferShape(prim, {x, shape_param.dim->ToAbstract(), shape_param.descending->ToAbstract(),
                                        shape_param.stable->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = SortExt_func_impl.InferType(prim, {x, dim, descending, stable});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto SortExtOpShapeTestCases = testing::ValuesIn({
  SortExtShape{{10},
               CreateScalar<int64_t>(-1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10},
               {10}},
  SortExtShape{{10, 8, 5},
               CreateScalar<int64_t>(0),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 8, 5},
               {10, 8, 5}},
  SortExtShape{{10, 8, 5},
               CreateScalar<int64_t>(1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 8, 5},
               {10, 8, 5}},
  SortExtShape{{10, 8, 5},
               CreateScalar<int64_t>(2),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 8, 5},
               {10, 8, 5}},
});

auto SortExtOpTypeTestCases = testing::ValuesIn({
  SortExtType{kFloat16, kInt64, kBool, kBool, kFloat16, kInt64},
  SortExtType{kFloat32, kInt64, kBool, kBool, kFloat32, kInt64},
});

INSTANTIATE_TEST_CASE_P(TestSortExt, TestSortExt, testing::Combine(SortExtOpShapeTestCases, SortExtOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
