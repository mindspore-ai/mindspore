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
#include "ops/pad.h"
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
struct PadParams {
  ShapeVector input_shape;
  TypePtr input_type;
  std::vector<std::vector<int64_t>> paddings;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestPad : public TestOps, public testing::WithParamInterface<PadParams> {};

TEST_P(TestPad, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  ASSERT_NE(input, nullptr);

  auto prim = std::make_shared<Primitive>(kNamePad);
  prim->set_attr("paddings", MakeValue<std::vector<std::vector<int64_t>>>(param.paddings));

  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto out_abstract = opt::CppInferShapeAndType(prim, {input});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestPadGroup, TestPad,
  testing::Values(PadParams{{1, -1, -1, -1}, kFloat32, {{1, 1}, {1, 1}, {1, 1}, {1, 1}}, {3, -1, -1, -1}, kFloat32},
                  PadParams{{1, 2, 3, 4}, kFloat32, {{1, 1}, {1, 1}, {1, 1}, {1, 1}}, {3, 4, 5, 6}, kFloat32},
                  PadParams{{-2}, kFloat32, {{1, 1}, {1, 1}, {1, 1}, {1, 1}}, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
