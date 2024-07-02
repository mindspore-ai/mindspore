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
struct StridedSliceOp {
  ShapeVector x_shape;
  ShapeVector begin;
  ShapeVector end;
  ShapeVector strides;
};

class TestStridedSlice : public TestSymbolEngine, public testing::WithParamInterface<StridedSliceOp> {};

using abstract::TensorShape;
TEST_P(TestStridedSlice, static_shape) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto begin = cg.NewValueNode(MakeValue(param.begin));
  auto end = cg.NewValueNode(MakeValue(param.end));
  auto strides = cg.NewValueNode(MakeValue(param.strides));
  auto m = cg.NewValueNode(MakeValue<int64_t>(0));
  auto node = cg.NewCNode("StridedSlice", {x, begin, end, strides, m, m, m, m, m}, {});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

TEST_P(TestStridedSlice, dynamic_shape) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto begin = cg.NewTensorInput("begin", kInt32, {static_cast<int64_t>(param.begin.size())});
  auto end = cg.NewTensorInput("end", kInt32, {static_cast<int64_t>(param.end.size())});
  auto strides = cg.NewTensorInput("strides", kInt32, {static_cast<int64_t>(param.strides.size())});
  auto m = cg.NewValueNode(MakeValue<int64_t>(0));
  auto node = cg.NewCNode("StridedSlice", {x, begin, end, strides, m, m, m, m, m}, {});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());

  abstract::AbstractBasePtrList inputs_args{x->abstract(), MakeValue(param.begin)->ToAbstract(),
                                            MakeValue(param.end)->ToAbstract(), MakeValue(param.strides)->ToAbstract()};
  ASSERT_TRUE(helper_->Infer(inputs_args));
  inputs_args.resize(9, m->abstract());
  auto out_abs = opt::CppInferShapeAndType(GetCNodePrimitive(node), inputs_args);
  node->abstract()->set_shape(out_abs->GetShape());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

INSTANTIATE_TEST_CASE_P(
  TestSymShape, TestStridedSlice,
  testing::Values(StridedSliceOp{{10, 10, 10, 10}, {0, 1, 2, 3}, {10, 9, 8, 7}, {1, 1, 1, 1}},
                  StridedSliceOp{{10, 10, 10, 10}, {0, 1, 2, 3}, {10, 9, 8, 7}, {2, 2, 2, 2}},
                  StridedSliceOp{{10, 10, 10, 10}, {0, 1, 2, 3}, {10, 9, 8, 7}, {3, 3, 3, 3}},
                  StridedSliceOp{{10, 10, 10, 10}, {0, -10, -8, -7}, {10, 9, 8, 7}, {2, 2, 2, 2}},
                  StridedSliceOp{{10, 10, 10, 10}, {0, -10, -8, -5}, {-1, -2, -3, -4}, {2, 2, 2, 2}},
                  StridedSliceOp{{10, 10, 10, 10}, {10, 9, 8, 7}, {0, 1, 2, 3}, {-1, -1, -1, -1}},
                  StridedSliceOp{{10, 10, 10, 10}, {10, 9, 8, 7}, {0, 1, 2, 3}, {-2, -2, -2, -2}},
                  StridedSliceOp{{10, 10, 10, 10}, {10, 9, 8, 7}, {0, 1, 2, 3}, {-3, -3, -3, -3}},
                  StridedSliceOp{{10, 10, 10, 10}, {10, 9, 8, 7}, {0, -10, -8, -7}, {-2, -2, -2, -2}},
                  StridedSliceOp{{10, 10, 10, 10}, {-1, -2, -3, -4}, {0, -10, -8, -5}, {-2, -2, -2, -2}}));
}  // namespace mindspore::symshape::test
