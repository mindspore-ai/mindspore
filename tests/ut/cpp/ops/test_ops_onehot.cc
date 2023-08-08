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
#include "ops/one_hot.h"
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
struct OneHotOpParams {
  ShapeVector indices_shape;
  TypePtr indices_type;
  ShapeVector depth_shape;
  TypePtr depth_type;
  ValuePtr depth_value;
  ShapeVector on_value_shape;
  TypePtr on_value_type;
  ShapeVector off_value_shape;
  TypePtr off_value_type;
  int64_t axis;
  ShapeVector out_shape;
  TypePtr out_type;
};
class TestOneHot : public TestOps, public testing::WithParamInterface<OneHotOpParams> {};

TEST_P(TestOneHot, dyn_shape) {
  const auto &param = GetParam();
  auto indices = std::make_shared<abstract::AbstractTensor>(param.indices_type, param.indices_shape);
  auto depth = std::make_shared<abstract::AbstractTensor>(param.depth_type, param.depth_shape);
  depth->set_value(param.depth_value);
  auto on_value = std::make_shared<abstract::AbstractTensor>(param.on_value_type, param.on_value_shape);
  auto off_value = std::make_shared<abstract::AbstractTensor>(param.off_value_type, param.off_value_shape);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(depth, nullptr);
  ASSERT_NE(on_value, nullptr);
  ASSERT_NE(off_value, nullptr);

  auto prim = std::make_shared<Primitive>(kNameOneHot);
  prim->set_attr("axis", MakeValue<int64_t>(param.axis));

  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  auto out_abstract = opt::CppInferShapeAndType(prim, {indices, depth, on_value, off_value});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestOneHotGroup, TestOneHot,
  testing::Values(
    OneHotOpParams{{-2}, kInt32, {1}, kInt32, kValueAny, {1}, kInt32, {0}, kInt32, 0, {-2}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, {1}, kInt32, kValueAny, {1}, kInt32, {0}, kInt32, -1, {2, 2, 3, -1}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, {1}, kInt32, kValueAny, {1}, kInt32, {0}, kInt32, 0, {-1, 2, 2, 3}, kInt32},
    OneHotOpParams{{2, 2, -1}, kInt32, {1}, kInt32, kValueAny, {1}, kInt32, {0}, kInt32, -1, {2, 2, -1, -1}, kInt32}));
}  // namespace ops
}  // namespace mindspore
