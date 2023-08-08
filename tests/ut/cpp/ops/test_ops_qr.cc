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
#include "ops/qr.h"
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
struct QRParams {
  ShapeVector x_shape;
  TypePtr x_type;
  bool full_matrices;
  ShapeVector q_shape;
  TypePtr q_type;
  ShapeVector r_shape;
  TypePtr r_type;
};

class TestQR : public TestOps, public testing::WithParamInterface<QRParams> {};

TEST_P(TestQR, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto q = std::make_shared<abstract::AbstractTensor>(param.q_type, param.q_shape);
  auto r = std::make_shared<abstract::AbstractTensor>(param.r_type, param.r_shape);
  AbstractBasePtrList abstract_list{q, r};
  auto expect = std::make_shared<abstract::AbstractTuple>(abstract_list);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(expect, nullptr);
  auto prim = std::make_shared<Primitive>(kNameQr);
  prim->set_attr("full_matrices", MakeValue<bool>(param.full_matrices));
  auto out_abstract = opt::CppInferShapeAndType(prim, {x});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestQRGroup, TestQR,
  testing::Values(
    QRParams{{5, 4}, kFloat32, true, {5, 5}, kFloat32, {5, 4}, kFloat32},
    QRParams{{5, 4}, kFloat32, false, {5, 4}, kFloat32, {4, 4}, kFloat32},
    QRParams{{-1, -1 , -1}, kFloat32, true, {-1, -1 , -1}, kFloat32, {-1, -1 , -1}, kFloat32},
    QRParams{{-1, -1 , -1}, kFloat32, false, {-1, -1 , -1}, kFloat32, {-1, -1 , -1}, kFloat32},
    QRParams{{-2}, kFloat32, true, {-2}, kFloat32, {-2}, kFloat32},
    QRParams{{-2}, kFloat32, false, {-2}, kFloat32, {-2}, kFloat32}));
}  // namespace opsout_shape
}  // namespace mindsporeclear

