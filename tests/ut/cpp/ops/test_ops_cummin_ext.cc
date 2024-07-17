/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/cummin_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct CumminExtShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ShapeVector values_shape;
  TypePtr values_type;
  ShapeVector indices_shape;
  TypePtr indices_type;
};

class TestCumminExt : public TestOps, public testing::WithParamInterface<CumminExtShapeParams> {};

TEST_P(TestCumminExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis->ToAbstract();

  auto values_shape = std::make_shared<abstract::Shape>(param.values_shape);
  auto indices_shape = std::make_shared<abstract::Shape>(param.indices_shape);
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{values_shape, indices_shape});
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(param.values_type), param.indices_type});

  CumminExtFuncImpl cummin_ext_func_impl;
  auto prim = std::make_shared<Primitive>("CumminExt");

  auto out_dtype = cummin_ext_func_impl.InferType(prim, {x, axis});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = cummin_ext_func_impl.InferShape(prim, {x, axis});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestCumminExt, TestCumminExt,
  testing::Values(CumminExtShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kFloat32, {3, 4, 5}, kInt64},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), {3, 4, 5}, kInt64, {3, 4, 5}, kInt64},
                  CumminExtShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3), {3, 4, 5}, kInt64, {3, 4, 5}, kInt64},
                  CumminExtShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2), {2, 3, 4, 5}, kInt32, {2, 3, 4, 5}, kInt64},
                  CumminExtShapeParams{{-1, -1, -1}, kUInt64, CreateScalar<int64_t>(2), {-1, -1, -1}, kUInt64, {-1, -1, -1}, kInt64},
                  CumminExtShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), {-2}, kFloat32, {-2}, kInt64}));
}  // namespace ops
}  // namespace mindspore
