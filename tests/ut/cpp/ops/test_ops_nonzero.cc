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
#include "ops/non_zero.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct NonZeroOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestNonZero : public TestOps, public testing::WithParamInterface<NonZeroOpParams> {};

TEST_P(TestNonZero, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(x, nullptr);
  auto prim = std::make_shared<Primitive>(kNameNonZero);
  auto out_abstract = opt::CppInferShapeAndType(prim, {x});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestNonZeroGroup, TestNonZero,
                        testing::Values(NonZeroOpParams{{2, 3}, kFloat, {-1, 2}, kInt64},
                                        NonZeroOpParams{{-1, -1}, kFloat, {-1, 2}, kInt64},
                                        NonZeroOpParams{{-2}, kFloat, {-1, -1}, kInt64}));
}  // namespace ops
}  // namespace mindspore
