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
#include "ops/ops_func_impl/topk_ext.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {

struct TopkExtShape {
  std::vector<int64_t> input_shape;
  ValuePtr k;
  ValuePtr dim;
  ValuePtr largest;
  ValuePtr sorted;
  std::vector<int64_t> out_values_shape;
  std::vector<int64_t> out_indices_shape;
};

struct TopkExtType {
  TypePtr input_type;
  TypePtr k_type;
  TypePtr dim_type;
  TypePtr largest_type;
  TypePtr sorted_type;
  TypePtr out_values_type;
  TypePtr out_indices_type;
};

class TestTopkExt : public TestOps, public testing::WithParamInterface<std::tuple<TopkExtShape, TopkExtType>> {};

TEST_P(TestTopkExt, TopkExt_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  TopkExtFuncImpl TopkExt_func_impl;
  auto prim = std::make_shared<Primitive>("TopkExt");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_type, shape_param.input_shape);
  std::vector<int64_t> empty_shape = {};
  auto k = std::make_shared<abstract::AbstractTensor>(dtype_param.k_type, empty_shape);
  auto dim = std::make_shared<abstract::AbstractTensor>(dtype_param.dim_type, empty_shape);
  auto largest = std::make_shared<abstract::AbstractTensor>(dtype_param.largest_type, empty_shape);
  auto sorted = std::make_shared<abstract::AbstractTensor>(dtype_param.sorted_type, empty_shape);
  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{std::make_shared<abstract::Shape>(shape_param.out_values_shape),
                                        std::make_shared<abstract::Shape>(shape_param.out_indices_shape)});
  auto expect_dtype = std::make_shared<Tuple>(
    std::vector<TypePtr>{std::make_shared<TensorType>(dtype_param.out_values_type), dtype_param.out_indices_type});

  auto out_shape =
    TopkExt_func_impl.InferShape(prim, {x, shape_param.k->ToAbstract(), shape_param.dim->ToAbstract(),
                                        shape_param.largest->ToAbstract(), shape_param.sorted->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = TopkExt_func_impl.InferType(prim, {x, k, dim, largest, sorted});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

class TestTopkExtSimpleInfer : public TestOps,
                               public testing::WithParamInterface<std::tuple<TopkExtShape, TopkExtType>> {};

TEST_P(TestTopkExtSimpleInfer, TopkExt_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  std::vector<int64_t> empty_shape = {};
  auto x = std::make_shared<tensor::BaseTensor>(dtype_param.input_type->type_id(), shape_param.input_shape);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(shape_param.k));
  input_values.push_back(std::move(shape_param.dim));
  input_values.push_back(std::move(shape_param.largest));
  input_values.push_back(std::move(shape_param.sorted));

  TopkExtFuncImpl TopkExt_func_impl;
  auto prim = std::make_shared<Primitive>("TopkExt");

  auto expect_shape = ShapeArray{shape_param.out_values_shape, shape_param.out_indices_shape};
  auto expect_type = TypePtrList{dtype_param.out_values_type, dtype_param.out_indices_type};

  auto output_shape = TopkExt_func_impl.InferShape(prim, input_values);
  auto output_type = TopkExt_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto TopkExtOpShapeTestCases = testing::ValuesIn({
  TopkExtShape{{10},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(-1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {3},
               {3}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(0),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {3, 8, 5},
               {3, 8, 5}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 3, 5},
               {10, 3, 5}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(2),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 8, 3},
               {10, 8, 3}},
});

auto TopkExtOpShapeSimpleInferTestCases = testing::ValuesIn({
  TopkExtShape{{10},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(-1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {3},
               {3}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(0),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {3, 8, 5},
               {3, 8, 5}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(1),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 3, 5},
               {10, 3, 5}},
  TopkExtShape{{10, 8, 5},
               CreateScalar<int64_t>(3),
               CreateScalar<int64_t>(2),
               CreateScalar<bool>(true),
               CreateScalar<bool>(true),
               {10, 8, 3},
               {10, 8, 3}},
});

auto TopkExtOpTypeTestCases = testing::ValuesIn({
  TopkExtType{kFloat16, kInt64, kInt64, kBool, kBool, kFloat16, kInt64},
  TopkExtType{kFloat32, kInt64, kInt64, kBool, kBool, kFloat32, kInt64},
  TopkExtType{kFloat64, kInt64, kInt64, kBool, kBool, kFloat64, kInt64},
});

INSTANTIATE_TEST_CASE_P(TestTopkExt, TestTopkExt, testing::Combine(TopkExtOpShapeTestCases, TopkExtOpTypeTestCases));
INSTANTIATE_TEST_CASE_P(TestTopkExtSimpleInfer, TestTopkExtSimpleInfer,
                        testing::Combine(TopkExtOpShapeSimpleInferTestCases, TopkExtOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
