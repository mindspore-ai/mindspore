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
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore::symshape::test {
struct MatMulDynShape {
  std::string opname;
  ShapeVector shape1;
  ShapeVector shape2;
  bool trans_a = false;
  bool trans_b = false;
};

class TestMatMul : public TestSymbolEngine, public testing::WithParamInterface<MatMulDynShape> {};

#define MATMUL_EXPECT_EQ_K(k1, k2)                  \
  do {                                              \
    if (!((k1)->HasData()) && !((k2)->HasData())) { \
      EXPECT_TRUE((k1)->EqualsTo(k2));              \
    }                                               \
  } while (0)

TEST_P(TestMatMul, dynshape) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto a = cg.NewTensorInput("a", kFloat32, param.shape1);
  auto b = cg.NewTensorInput("b", kFloat32, param.shape2);
  CNodePtr node;
  if (param.opname.find("Ext") == std::string::npos) {
    auto trans_a = cg.NewValueNode(MakeValue(param.trans_a));
    auto trans_b = cg.NewValueNode(MakeValue(param.trans_b));
    node = cg.NewCNode(param.opname, {a, b, trans_a, trans_b}, {});
  } else {
    node = cg.NewCNode(param.opname, {a, b}, {});
  }
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  auto out_rank = out_shape->size();
  ASSERT_TRUE(helper_->SupportInfer());
  if (!node->abstract()->GetShape()->IsDynamic()) {
    return;
  }

  auto shape1 = a->abstract()->GetSymbolicShape();
  UT_CHECK_NULL(shape1);
  size_t rank1 = shape1->size();
  auto shape2 = b->abstract()->GetSymbolicShape();
  UT_CHECK_NULL(shape2);
  size_t rank2 = shape2->size();
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));

  if (rank1 >= 2 && rank2 >= 2) {
    size_t m = rank1 - 2;
    size_t k1 = rank1 - 1;
    if (param.trans_a) {
      std::swap(m, k1);
    }
    size_t k2 = rank2 - 2;
    size_t n = rank2 - 1;
    if (param.trans_b) {
      std::swap(k2, n);
    }
    EXPECT_TRUE(shape1->item(m)->EqualsTo(out_shape->item(out_rank - 2)));
    EXPECT_TRUE(shape2->item(n)->EqualsTo(out_shape->item(out_rank - 1)));
    MATMUL_EXPECT_EQ_K(shape1->item(k1), shape2->item(k2));
  } else {
    if (rank1 >= 2) {
      // rank2 == 1
      EXPECT_TRUE(shape1->item(rank1 - 2)->EqualsTo(out_shape->item(out_rank - 1)));
      MATMUL_EXPECT_EQ_K(shape1->item(rank1 - 1), shape2->item(0));
    }
    if (rank2 >= 2) {
      // rank1 == 1
      EXPECT_TRUE(shape2->item(rank2 - 1)->EqualsTo(out_shape->item(out_rank - 1)));
      MATMUL_EXPECT_EQ_K(shape1->item(0), shape2->item(rank2 - 2));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestMatMul,
                        testing::Values(MatMulDynShape{"MatMul", {-1, -1}, {-1, -1}, true, true},
                                        MatMulDynShape{"BatchMatMul", {32, -1, -1}, {16, 32, -1, -1}, false, true},
                                        MatMulDynShape{"MatMulExt", {-1}, {-1}},  // output is scalar
                                        MatMulDynShape{"MatMulExt", {-1}, {-1, -1}},
                                        MatMulDynShape{"MatMulExt", {-1, -1}, {-1}},
                                        MatMulDynShape{"MatMulExt", {-1, -1}, {-1, -1}},
                                        // MatMulDynShape{"MatMulExt", {32, -1, -1}, {-1}},
                                        // MatMulDynShape{"MatMulExt", {-1}, {16, 32, -1, -1}},
                                        MatMulDynShape{"BatchMatMulExt", {32, -1, -1}, {32, -1, -1}}));
}  // namespace mindspore::symshape::test
