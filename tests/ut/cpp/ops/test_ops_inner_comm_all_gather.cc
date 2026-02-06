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
#include <memory>
#include "common/common_test.h"
#include "infer/ops_func_impl/communication/inner_comm_all_gather.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct AllGatherShapeParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr rank_size;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestAllGather : public TestOps, public testing::WithParamInterface<AllGatherShapeParams> {};

TEST_P(TestAllGather, dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  InnerCommAllGatherFuncImpl all_gather_func_impl;
  auto prim = std::make_shared<Primitive>("InnerCommAllGather");

  auto out_dtype = all_gather_func_impl.InferType(prim, {input_x, param.rank_size->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = all_gather_func_impl.InferShape(prim, {input_x, param.rank_size->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestAllGather, TestAllGather,
  testing::Values(AllGatherShapeParams{{3, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {6, 4, 5}, kFloat32},
                  AllGatherShapeParams{{3, 4, 5}, kFloat16, CreateScalar<int64_t>(4), {12, 4, 5}, kFloat16},
                  AllGatherShapeParams{{-1, 4, 5}, kInt64, CreateScalar<int64_t>(2), {-1, 4, 5}, kInt64},
                  AllGatherShapeParams{{3, -1, -1}, kInt64, CreateScalar<int64_t>(2), {6, -1, -1}, kInt64}));
}  // namespace ops
}  // namespace mindspore
