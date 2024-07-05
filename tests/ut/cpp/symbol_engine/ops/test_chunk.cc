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
struct ChunkOp {
  ShapeVector x_shape;
  int64_t chunks;
  int64_t dim;
};

class TestChunk : public TestSymbolEngine, public testing::WithParamInterface<ChunkOp> {};

using abstract::TensorShape;
TEST_P(TestChunk, compare_shape_succ) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto chunks = cg.NewValueNode(MakeValue(param.chunks));
  auto dim = cg.NewValueNode(MakeValue(param.dim));
  auto node = cg.NewCNode("Chunk", {x, chunks, dim}, {});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestChunk,
                        testing::Values(ChunkOp{{10}, 3, 0},          //
                                        ChunkOp{{10}, 2, 0},          //
                                        ChunkOp{{10}, 1, 0},          //
                                        ChunkOp{{10, -1}, 3, 0},      //
                                        ChunkOp{{15, -1}, 3, 0},      //
                                        ChunkOp{{-1, 10}, 4, -1},     //
                                        ChunkOp{{-1, 10, 20}, 4, -1}  //
                                        ));
}  // namespace mindspore::symshape::test
