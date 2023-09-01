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
#include "ops/ops_func_impl/reshape.h"
#include "ops/op_name.h"
#include "ops/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct ReshapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr shape;  // shape is tuple[int]
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestReshape : public TestOps, public testing::WithParamInterface<ReshapeParams> {};

TEST_P(TestReshape, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto shape = param.shape->ToAbstract();
  ASSERT_NE(shape, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ReshapeFuncImpl>(kNameReshape, {x, shape}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestReshapeGroup, TestReshape,
  testing::Values(ReshapeParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({4, 3, 2}), {4, 3, 2}, kFloat32},
                  ReshapeParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({-1, 3, 2}), {4, 3, 2}, kFloat32},
                  ReshapeParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({4, -1}), {4, 6}, kFloat32},
                  ReshapeParams{{2, 3}, kFloat32, CreatePyIntTuple({6}), {6}, kFloat32},
                  ReshapeParams{{2, 3}, kFloat32, CreatePyIntTuple({kValueAny, 2}), {-1, 2}, kFloat32},
                  ReshapeParams{{2, 3}, kFloat32, CreatePyIntTuple({kValueAny, kValueAny}), {-1, -1}, kFloat32},
                  ReshapeParams{{2, 3}, kFloat32, CreatePyIntTuple({kValueAny}), {-1}, kFloat32},
                  ReshapeParams{{2, 3}, kFloat32, kValueAny, {-2}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, CreatePyIntTuple({4, 3, 2}), {4, 3, 2}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, CreatePyIntTuple({-1, 3, 2}), {-1, 3, 2}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, CreatePyIntTuple({kValueAny, 2}), {-1, 2}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, CreatePyIntTuple({kValueAny, kValueAny}), {-1, -1}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, CreatePyIntTuple({kValueAny}), {-1}, kFloat32},
                  ReshapeParams{{-1, -1}, kFloat32, kValueAny, {-2}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, CreatePyIntTuple({4, 3, 2}), {4, 3, 2}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, CreatePyIntTuple({-1, 3, 2}), {-1, 3, 2}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, CreatePyIntTuple({kValueAny, 2}), {-1, 2}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, CreatePyIntTuple({kValueAny, kValueAny}), {-1, -1}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, CreatePyIntTuple({kValueAny}), {-1}, kFloat32},
                  ReshapeParams{{-2}, kFloat32, kValueAny, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
