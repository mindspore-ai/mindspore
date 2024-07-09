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

#include <string>

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
struct ElemwiseBinOpDynShape {
  std::string opname;
  ShapeVector shape1;
  ShapeVector shape2;
  ShapeVector out_status;  // 0: output new symbol. 1: equals to shape1. 2: equals to shape2. 3: equals to the both.
};

class TestElemwiseBinOp : public TestSymbolEngine, public testing::WithParamInterface<ElemwiseBinOpDynShape> {};

TEST_P(TestElemwiseBinOp, dynshape) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto a = cg.NewTensorInput("a", kFloat32, param.shape1);
  auto b = cg.NewTensorInput("b", kFloat32, param.shape2);
  auto node = cg.NewCNode(param.opname, {a, b}, {});
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  auto shape1 = a->abstract()->GetSymbolicShape();
  UT_CHECK_NULL(shape1);
  auto shape2 = b->abstract()->GetSymbolicShape();
  UT_CHECK_NULL(shape2);
  ASSERT_EQ(out_shape->size(), param.out_status.size());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  auto n = param.out_status.size();
  for (size_t i = n; i > 0; i--) {
    auto out = out_shape->item_as<IntSymbol>(n - i);
    if (param.out_status[n - i] == 0) {
      ASSERT_FALSE(i <= shape1->size() && out->EqualsTo(shape1->item(shape1->size() - i)));
      ASSERT_FALSE(i <= shape2->size() && out->EqualsTo(shape2->item(shape2->size() - i)));
    } else if (param.out_status[n - i] == 1) {
      ASSERT_TRUE(out->EqualsTo(shape1->item(shape1->size() - i)));
      ASSERT_FALSE(i <= shape2->size() && out->EqualsTo(shape2->item(shape2->size() - i)));
    } else if (param.out_status[n - i] == 2) {
      ASSERT_FALSE(i <= shape1->size() && out->EqualsTo(shape1->item(shape1->size() - i)));
      ASSERT_TRUE(out->EqualsTo(shape2->item(shape2->size() - i)));
    } else {  // param.out_status[n - i] == 3
      ASSERT_TRUE(out->EqualsTo(shape1->item(shape1->size() - i)));
      ASSERT_TRUE(out->EqualsTo(shape2->item(shape2->size() - i)));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestElemwiseBinOp,
                        testing::Values(ElemwiseBinOpDynShape{"Add", {-1, 32, -1}, {1, -1, -1}, {1, 1, 0}},
                                        ElemwiseBinOpDynShape{"Sub", {1, 32, -1}, {16, -1, -1}, {2, 1, 0}},
                                        ElemwiseBinOpDynShape{"Mul", {-1, 8}, {-1, -1, 8}, {2, 0, 3}},
                                        ElemwiseBinOpDynShape{"RealDiv", {-1, 32, -1}, {-1, -1}, {1, 1, 0}}));
}  // namespace mindspore::symshape::test
