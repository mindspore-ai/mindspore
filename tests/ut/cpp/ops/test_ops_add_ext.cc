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
#include "ops/ops_func_impl/add_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct AddExtShape {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> y_shape;
  ValuePtr alpha;
  std::vector<int64_t> out_shape;
};

struct AddExtType {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestAddExt : public TestOps, public testing::WithParamInterface<std::tuple<AddExtShape, AddExtType>> {};

TEST_P(TestAddExt, AddExt_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  AddExtFuncImpl AddExt_func_impl;
  auto prim = std::make_shared<Primitive>("AddExt");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = AddExt_func_impl.InferShape(prim, {x, y, shape_param.alpha->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = AddExt_func_impl.InferType(prim, {x, y, shape_param.alpha->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto AddExtOpShapeTestCases = testing::ValuesIn({
    /* y is number */
    AddExtShape{{10}, {}, CreateScalar<int64_t>(2), {10}}, AddExtShape{{10, 1, 2}, {}, CreateScalar<int64_t>(2), {10, 1, 2}},
    AddExtShape{{10, 4, 2}, {}, CreateScalar<int64_t>(2), {10, 4, 2}}, AddExtShape{{10, 1, -1}, {}, CreateScalar<int64_t>(2), {10, 1, -1}},
    AddExtShape{{-2}, {}, CreateScalar<int64_t>(2), {-2}},
    /* x is number */
    AddExtShape{{}, {10}, CreateScalar<int64_t>(2), {10}}, AddExtShape{{}, {10, 1, 2}, CreateScalar<int64_t>(2), {10, 1, 2}},
    AddExtShape{{}, {10, 4, 2}, CreateScalar<int64_t>(2), {10, 4, 2}}, AddExtShape{{}, {10, 1, -1}, CreateScalar<int64_t>(2), {10, 1, -1}},
    AddExtShape{{}, {-2}, CreateScalar<int64_t>(2), {-2}},
    /* x and y both tensor */
    AddExtShape{{4, 5}, {2, 3, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    AddExtShape{{2, 1, 4, 5, 6, 9}, {9}, CreateScalar<int64_t>(2), {2, 1, 4, 5, 6, 9}},
    AddExtShape{{2, 3, 4, -1}, {2, 3, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    AddExtShape{{2, 3, 4, -1}, {-1, -1, 4, 5}, CreateScalar<int64_t>(2), {2, 3, 4, 5}},
    AddExtShape{{2, 1, 4, -1}, {-1, -1, 4, 5}, CreateScalar<int64_t>(2), {2, -1, 4, 5}},
    AddExtShape{{2, 1, 4, 5, 6, 9}, {-2}, CreateScalar<int64_t>(2), {-2}}, AddExtShape{{2, 1, 4, 5, -1, 9}, {-2}, CreateScalar<int64_t>(2), {-2}},
    AddExtShape{{-2}, {2, 1, 4, 5, 6, 9}, CreateScalar<int64_t>(2), {-2}}, AddExtShape{{-2}, {2, 1, 4, 5, -1, 9}, CreateScalar<int64_t>(2), {-2}},
    AddExtShape{{-2}, {-2}, CreateScalar<int64_t>(2), {-2}}
});

auto AddExtOpTypeTestCases = testing::ValuesIn({
  AddExtType{kFloat16, kFloat16, kFloat16},
  AddExtType{kFloat32, kFloat32, kFloat32},
  AddExtType{kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestAddExt, TestAddExt, testing::Combine(AddExtOpShapeTestCases, AddExtOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
