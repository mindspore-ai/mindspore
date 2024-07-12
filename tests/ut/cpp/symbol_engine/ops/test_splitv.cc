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
#include "abstract/dshape.h"
#include "common/mockcpp.h"

namespace mindspore::symshape::test {
struct SplitvOp {
  ShapeVector x_shape;
  int64_t axis;
  ShapeVector size_splits;
};

class TestSplitV : public TestSymbolEngine, public testing::WithParamInterface<SplitvOp> {};

using abstract::TensorShape;
TEST_P(TestSplitV, compare_shape_succ) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto split_dim = MakeValue(param.axis);
  auto size_splits = MakeValue(param.size_splits);
  auto num_split = MakeValue<int64_t>(static_cast<int64_t>(param.size_splits.size()));
  auto node =
    cg.NewCNode("SplitV", {x}, {{"split_dim", split_dim}, {"size_splits", size_splits}, {"num_split", num_split}});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestSplitV,
                        testing::Values(SplitvOp{{-2}, -2, {3, 2, -1, 2}},          // dynamic rank
                                        SplitvOp{{10, 10, 10}, 0, {1, 2, 3, 4}},    //
                                        SplitvOp{{10, 10, 10}, 0, {1, 2, -1, 4}},   //
                                        SplitvOp{{10, 10, 10}, 1, {1, -1, 3, 3}},   //
                                        SplitvOp{{10, 10, 10}, 2, {-1, 2, 3, 2}},   //
                                        SplitvOp{{10, 10, 10}, -1, {-1, 2, 3, 2}},  //
                                        SplitvOp{{10, 10, 10}, -2, {3, 2, -1, 2}},  //
                                        SplitvOp{{10, 10, 10}, -3, {2, 2, 3, -1}}   //
                                        ));
}  // namespace mindspore::symshape::test
