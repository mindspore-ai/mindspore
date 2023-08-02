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
#include "ops/reduce_mean.h"
#include "ops/reduce_sum.h"
#include "ops/op_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ReduceParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;  // axis is tuple[int]/list[int]/int
  bool keep_dims;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestReduce : public TestOps, public testing::WithParamInterface<ReduceParams> {};

TEST_P(TestReduce, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(axis, nullptr);

  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  auto prim = std::make_shared<Primitive>(kNameReduceMean);
  prim->AddAttr(kKeepDims, MakeValue<bool>(param.keep_dims));
  auto out_abstract = ReduceArithmeticInferFunc(nullptr, prim, {x, axis});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestReduce, TestReduce,
  testing::Values(ReduceParams{{2, 3}, kFloat32, CreateScalar(1), true, {2, 1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateScalar(1), false, {2}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateScalar(kValueAny), true, {-1, -1}, kFloat32},
                  ReduceParams{{-1, 3}, kFloat32, CreateScalar(0), true, {1, 3}, kFloat32},
                  ReduceParams{{-1, 3}, kFloat32, CreateScalar(0), false, {3}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({1}), true, {2, 1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({1}), false, {2}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({0, 1}), true, {1, 1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({}), false, {}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({kValueAny}), false, {-2}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateTuple({1, kValueAny}), true, {-1, -1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({1}), true, {2, 1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({1}), false, {2}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({0, 1}), true, {1, 1}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({}), false, {}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({kValueAny}), false, {-2}, kFloat32},
                  ReduceParams{{2, 3}, kFloat32, CreateList({1, kValueAny}), true, {-1, -1}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
