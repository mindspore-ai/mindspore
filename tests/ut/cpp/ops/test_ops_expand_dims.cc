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
#include "ops/ops_func_impl/expand_dims.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ExpandDimsShapeParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr axis;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestExpandDims : public TestOps, public testing::WithParamInterface<ExpandDimsShapeParams> {};

TEST_P(TestExpandDims, dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  ExpandDimsFuncImpl expand_dims_func_impl;
  auto prim = std::make_shared<Primitive>("ExpandDims");

  auto out_dtype = expand_dims_func_impl.InferType(prim, {input_x, param.axis->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = expand_dims_func_impl.InferShape(prim, {input_x, param.axis->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestExpandDims, TestExpandDims,
  testing::Values(ExpandDimsShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 1, 5}, kFloat32},
                  ExpandDimsShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(0), {1, 3, 4, 5}, kInt64},
                  ExpandDimsShapeParams{{3, 4, 5}, kInt64, CreateScalar<int64_t>(-3), {3, 1, 4, 5}, kInt64},
                  ExpandDimsShapeParams{{2, 3, 4, 5}, kInt32, CreateScalar<int64_t>(-2), {2, 3, 4, 1, 5}, kInt32},
                  ExpandDimsShapeParams{{-1, -1, -1}, kUInt64, CreateScalar<int64_t>(2), {-1, -1, 1, -1}, kUInt64},
                  ExpandDimsShapeParams{{-1, -1, -1}, kFloat32, CreateScalar(kValueAny), {-1, -1, -1, -1}, kFloat32},
                  ExpandDimsShapeParams{{-2}, kFloat32, CreateScalar<int64_t>(2), {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
