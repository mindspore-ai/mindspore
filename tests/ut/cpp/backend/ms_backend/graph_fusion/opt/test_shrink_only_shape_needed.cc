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

#include <mindspore/core/include/ir/core_ops_primitive.h>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/symbol_engine_builder.h"
#include "backend/ms_backend/graph_fusion/shrink_only_shape_needed.h"
#include "utils/anf_utils.h"
#include "ir/functor.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::graphkernel::test {
namespace {
class SimpleShapeCalcFunctor : public ShapeCalcFunctor {
 public:
  SimpleShapeCalcFunctor() : ShapeCalcFunctor("ShapeCalc") {}
  ShapeArray Calc(const ShapeArray &) const override { return {}; }
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override { return {2}; }
  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &) override {}
};
}  // namespace

/// Feature: test graph kernel pass ShrinkOnlyShapeNeeded
/// Description: sub graph is single output and it is only used by shape op
/// Expectation: replace shape op's input with sub graph's input
TEST_F(GraphKernelCommonTestSuite, single_output_shape_depend) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,Mul");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {-1, -1});
  auto p1 = c.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = c.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = c.NewTensorInput("p3", kFloat32, {-1, -1, -1});
  auto symbol_shape = p0->abstract()->GetShape()->BuildSymbolicShape();
  p0->abstract()->SetSymbolicShape(symbol_shape);
  p1->abstract()->SetSymbolicShape(symbol_shape);
  p2->abstract()->SetSymbolicShape(symbol_shape);
  p3->abstract()->SetSymbolicShape(p3->abstract()->GetShape()->BuildSymbolicShape());
  auto add = c.NewCNodeWithBuildInfo("Add", {p0, p1});
  auto mul = c.NewCNodeWithBuildInfo("Mul", {add, p2});
  auto shape = c.NewCNodeWithBuildInfo("Shape", {mul});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  c.SetOutput(reshape);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::SymbolEngineBuilder>(false),
         std::make_shared<graphkernel::ShrinkOnlyShapeNeeded>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 0);
}

/// Feature: test graph kernel pass ShrinkOnlyShapeNeeded
/// Description: sub graph is single output and it is used by shape op and compute op
/// Expectation: sub graph has no change
TEST_F(GraphKernelCommonTestSuite, single_output_value_depend) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,Mul");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {-1, -1});
  auto p1 = c.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = c.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = c.NewTensorInput("p3", kFloat32, {-1, -1, -1});
  auto symbol_shape = p0->abstract()->GetShape()->BuildSymbolicShape();
  p0->abstract()->SetSymbolicShape(symbol_shape);
  p1->abstract()->SetSymbolicShape(symbol_shape);
  p2->abstract()->SetSymbolicShape(symbol_shape);
  p3->abstract()->SetSymbolicShape(p3->abstract()->GetShape()->BuildSymbolicShape());
  auto add = c.NewCNodeWithBuildInfo("Add", {p0, p1});
  auto mul = c.NewCNodeWithBuildInfo("Mul", {add, p2});
  auto shape = c.NewCNodeWithBuildInfo("Shape", {mul});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  auto sub = c.NewCNodeWithBuildInfo("Sub", {mul, reshape});
  c.SetOutput(sub);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::SymbolEngineBuilder>(false),
         std::make_shared<graphkernel::ShrinkOnlyShapeNeeded>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 1);
}

/// Feature: test graph kernel pass ShrinkOnlyShapeNeeded
/// Description: sub graph is multiple outputs and some outputs are used only by shape op
/// Expectation: the output which is only used by shape op should be remove from the output list of sub graph
TEST_F(GraphKernelCommonTestSuite, multiple_output_shape_depend1) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,Mul");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {-1, -1});
  auto p1 = c.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = c.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = c.NewTensorInput("p3", kFloat32, {-1, -1, -1});
  auto symbol_shape = p0->abstract()->GetShape()->BuildSymbolicShape();
  p0->abstract()->SetSymbolicShape(symbol_shape);
  p1->abstract()->SetSymbolicShape(symbol_shape);
  p2->abstract()->SetSymbolicShape(symbol_shape);
  p3->abstract()->SetSymbolicShape(p3->abstract()->GetShape()->BuildSymbolicShape());
  auto add = c.NewCNodeWithBuildInfo("Add", {p0, p1});
  auto mul = c.NewCNodeWithBuildInfo("Mul", {add, p2});
  auto shape_calc = c.NewCNodeWithBuildInfo("ShapeCalc", {add, c.NewValueNode(MakeValue<int64_t>(0))},
                                            {{"only_depend_shape", MakeValue<std::vector<bool>>({true, false})},
                                             {"functor", std::make_shared<SimpleShapeCalcFunctor>()}});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {p3, shape_calc});
  auto sub = c.NewCNodeWithBuildInfo("Sub", {mul, reshape});
  c.SetOutput(sub);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::SymbolEngineBuilder>(false),
         std::make_shared<graphkernel::ShrinkOnlyShapeNeeded>()});
  auto gk_nodes = GetAllGKNodes(fg);
  ASSERT_EQ(gk_nodes.size(), 1);
  EXPECT_TRUE(IsPrimitiveCNode(GetCNodeFuncGraph(gk_nodes[0])->output(), prim::kPrimMul));
}

/// Feature: test graph kernel pass ShrinkOnlyShapeNeeded
/// Description: sub graph is multiple outputs and some outputs are used only by shape op, but the output shape
///              is not equal to the input shape
/// Expectation: sub graph has no change
TEST_F(GraphKernelCommonTestSuite, multiple_output_shape_depend2) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,Mul");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {-1, -1});
  auto p1 = c.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = c.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = c.NewTensorInput("p3", kFloat32, {-1, -1, -1});
  p0->abstract()->SetSymbolicShape(p0->abstract()->GetShape()->BuildSymbolicShape());
  p1->abstract()->SetSymbolicShape(p1->abstract()->GetShape()->BuildSymbolicShape());
  p2->abstract()->SetSymbolicShape(p2->abstract()->GetShape()->BuildSymbolicShape());
  p3->abstract()->SetSymbolicShape(p3->abstract()->GetShape()->BuildSymbolicShape());
  auto add = c.NewCNodeWithBuildInfo("Add", {p0, p1});
  auto mul = c.NewCNodeWithBuildInfo("Mul", {add, p2});
  auto shape = c.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  auto sub = c.NewCNodeWithBuildInfo("Sub", {mul, reshape});
  c.SetOutput(sub);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::SymbolEngineBuilder>(false),
         std::make_shared<graphkernel::ShrinkOnlyShapeNeeded>()});
  auto gk_nodes = GetAllGKNodes(fg);
  ASSERT_EQ(gk_nodes.size(), 1);
  EXPECT_TRUE(IsPrimitiveCNode(GetCNodeFuncGraph(gk_nodes[0])->output(), prim::kPrimMakeTuple));
}

/// Feature: test graph kernel pass ShrinkOnlyShapeNeeded
/// Description: sub graph is multiple outputs and some outputs are used by shape op and compute op
/// Expectation: sub graph has no change
TEST_F(GraphKernelCommonTestSuite, multiple_output_value_depend) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,Mul");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {-1, -1});
  auto p1 = c.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = c.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = c.NewTensorInput("p3", kFloat32, {-1, -1, -1});
  auto symbol_shape = p0->abstract()->GetShape()->BuildSymbolicShape();
  p0->abstract()->SetSymbolicShape(symbol_shape);
  p1->abstract()->SetSymbolicShape(symbol_shape);
  p2->abstract()->SetSymbolicShape(symbol_shape);
  p3->abstract()->SetSymbolicShape(p3->abstract()->GetShape()->BuildSymbolicShape());
  auto add = c.NewCNodeWithBuildInfo("Add", {p0, p1});
  auto mul = c.NewCNodeWithBuildInfo("Mul", {add, p2});
  auto shape = c.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  auto addn = c.NewCNodeWithBuildInfo("AddN", {add, mul, reshape});
  c.SetOutput(addn);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::SymbolEngineBuilder>(false),
         std::make_shared<graphkernel::ShrinkOnlyShapeNeeded>()});
  auto gk_nodes = GetAllGKNodes(fg);
  ASSERT_EQ(gk_nodes.size(), 1);
  EXPECT_TRUE(IsPrimitiveCNode(GetCNodeFuncGraph(gk_nodes[0])->output(), prim::kPrimMakeTuple));
}
}  // namespace mindspore::graphkernel::test
