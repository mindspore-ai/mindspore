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
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/narrow.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct NarrowShape {
  std::vector<int64_t> input_shape;
  ValuePtr dim;
  ValuePtr start;
  ValuePtr length;
  std::vector<int64_t> out_shape;
};

struct NarrowType {
  TypePtr input_type;
  TypePtr dim_type;
  TypePtr start_type;
  TypePtr length_type;
  TypePtr out_type;
};

class TestNarrow : public TestOps, public testing::WithParamInterface<std::tuple<NarrowShape, NarrowType>> {};

TEST_P(TestNarrow, Narrow_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  NarrowFuncImpl Narrow_func_impl;
  auto prim = std::make_shared<Primitive>("Narrow");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_type, shape_param.input_shape);
  std::vector<int64_t> empty_shape = {};
  auto dim = std::make_shared<abstract::AbstractTensor>(dtype_param.dim_type, empty_shape);
  auto start = std::make_shared<abstract::AbstractTensor>(dtype_param.start_type, empty_shape);
  auto length = std::make_shared<abstract::AbstractTensor>(dtype_param.length_type, empty_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = Narrow_func_impl.InferShape(
    prim, {x, shape_param.dim->ToAbstract(), shape_param.start->ToAbstract(), shape_param.length->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = Narrow_func_impl.InferType(prim, {x, dim, start, length});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto NarrowOpShapeTestCases = testing::ValuesIn({
  NarrowShape{{10},
              CreateScalar<int64_t>(0),
              CreateScalar<int64_t>(-3),
              CreateScalar<int64_t>(3),
              {3}},
  NarrowShape{{10, 8, 5},
              CreateScalar<int64_t>(1),
              CreateScalar<int64_t>(0),
              CreateScalar<int64_t>(8),
              {10, 8, 5}},
  NarrowShape{{10, 8, 5},
              CreateScalar<int64_t>(2),
              CreateScalar<int64_t>(1),
              CreateScalar<int64_t>(3),
              {10, 8, 3}},
});

auto NarrowOpTypeTestCases = testing::ValuesIn({
  NarrowType{kFloat16, kInt64, kInt64, kInt64, kFloat16},
  NarrowType{kFloat32, kInt64, kInt64, kInt64, kFloat32},
  NarrowType{kBFloat16, kInt64, kInt64, kInt64, kBFloat16},
});

INSTANTIATE_TEST_CASE_P(TestNarrow, TestNarrow, testing::Combine(NarrowOpShapeTestCases, NarrowOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
