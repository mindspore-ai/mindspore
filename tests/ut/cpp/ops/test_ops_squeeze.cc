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
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/squeeze.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))

struct SqueezeShapeParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr axis;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestSqueeze : public TestOps, public testing::WithParamInterface<SqueezeShapeParams> {};

TEST_P(TestSqueeze, dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  SqueezeFuncImpl squeeze_func_impl;
  auto prim = std::make_shared<Primitive>("Squeeze");

  auto out_dtype = squeeze_func_impl.InferType(prim, {input_x, param.axis->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = squeeze_func_impl.InferShape(prim, {input_x, param.axis->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestSqueeze, TestSqueeze,
  testing::Values(SqueezeShapeParams{{3, 4, 1, 5}, kFloat32, CreateTuple({I64(2)}), {3, 4, 5}, kFloat32},
                  SqueezeShapeParams{{3, 4, 1, 5, 1}, kFloat32, CreateTuple({I64(2), I64(4)}), {3, 4, 5}, kFloat32},
                  SqueezeShapeParams{{1, 3, 4, 5}, kInt64, CreateTuple({I64(0)}), {3, 4, 5}, kInt64},
                  SqueezeShapeParams{{3, 4, 1, 5, 1}, kInt64, CreateTuple({I64(-3), I64(-1)}), {3, 4, 5}, kInt64},
                  SqueezeShapeParams{{3, 1, 4, 5}, kInt64, CreateTuple({I64(-3)}), {3, 4, 5}, kInt64},
                  SqueezeShapeParams{{2, 3, 4, 1, 5}, kInt32, CreateTuple({I64(-2)}), {2, 3, 4, 5}, kInt32},
                  SqueezeShapeParams{{-1, -1, -1, -1}, kUInt64, CreateTuple({I64(2)}), {-2}, kUInt64},
                  SqueezeShapeParams{
                    {-1, -1, -1, -1}, kFloat32, CreateTuple({I64(0), I64(1), I64(-1)}), {-2}, kFloat32},
                  SqueezeShapeParams{{-2}, kFloat32, CreateTuple({I64(2)}), {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
