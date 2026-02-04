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

#include "backend/ms_backend/graph_fusion/kernel_packet/kernel_packet_common_test_suite.h"
#include "ir/functor.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel::test {
class StubConcatShapeCalcFunctor : public ShapeCalcFunctor {
 public:
  StubConcatShapeCalcFunctor() : ShapeCalcFunctor("ShapeCalc_g_concat") {}
  ShapeArray Calc(const ShapeArray &) const override { return {}; }
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override { return {2, 2}; }
  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &) override {}
};

/// Feature: KernelPacket
/// Description: the ShapeCalc's input is RealMakeTuple but it's only depend shape. (case from SDXL)
/// Expectation: fuse the RealMakeTuple.
TEST_F(TestKernelPacket, realmaketuple_shapecalc) {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto x = gb.NewTensorInput("x", kFloat32, {-1, -1});
  auto mt = gb.NewCNodeWithBuildInfo("RealMakeTuple", {p1, p2});
  auto shape_calc = gb.NewCNodeWithBuildInfo("ShapeCalc", {mt, gb.NewValueNode(MakeValue<int64_t>(0))},
                                             {{"only_depend_shape", MakeValue<std::vector<bool>>({true, false})},
                                              {"functor", std::make_shared<StubConcatShapeCalcFunctor>()}});
  auto shapecalc_0 = gb.NewCNode("TupleGetItem", {shape_calc, gb.NewValueNode(MakeValue<int64_t>(0))});
  auto shape_p1 = gb.NewCNodeWithBuildInfo("Shape", {p1});
  auto out = gb.NewCNodeWithBuildInfo("Slice", {x, shapecalc_0, shape_p1});
  gb.SetOutput(out);
  RunPass(gb.GetGraph(), {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  auto cnodes = TopoSort(gb.GetGraph()->output(), SuccIncoming,
                         [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  // only one packet node
  ASSERT_EQ(cnodes.size(), 1);
}
}  // namespace mindspore::graphkernel::test
