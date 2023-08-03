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
#include "ops/polar.h"
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
struct PolarParams {
  ShapeVector abs_shape;
  TypePtr abs_type;
  ShapeVector angle_shape;
  TypePtr angle_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestPolar : public TestOps, public testing::WithParamInterface<PolarParams> {};

TEST_P(TestPolar, dyn_shape) {
  const auto &param = GetParam();
  auto abs = std::make_shared<abstract::AbstractTensor>(param.abs_type, param.abs_shape);
  auto angle = std::make_shared<abstract::AbstractTensor>(param.angle_type, param.angle_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(abs, nullptr);
  ASSERT_NE(angle, nullptr);
  auto prim = std::make_shared<Primitive>(kNamePolar);
  auto out_abstract = opt::CppInferShapeAndType(prim, {abs, angle});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestPolarGroup, TestPolar,
                        testing::Values(
                          PolarParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {2, 3}, kComplex64},
                          PolarParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32, {-1, -1}, kComplex64},
                          PolarParams{{-2}, kFloat32, {-2}, kFloat32, {-2}, kComplex64}));
}  // namespace ops
}  // namespace mindspore
