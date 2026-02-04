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

#include <map>
#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/symbol_engine_builder.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "backend/ms_backend/graph_fusion/split_model/split_model_factory.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "common/mockcpp.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::graphkernel {
void SetKernelInfo(const FuncGraphPtr &func_graph);
}

namespace mindspore::graphkernel::test {
class TestGraphSplit : public GraphKernelCommonTestSuite {
 public:
  TestGraphSplit() {}
  void SetUp() override {
    SPLIT_MODEL_REGISTER(kAscendDevice, graphkernel::inner::SplitModelAscend);
  }
};

FuncGraphPtr ConstructGraph_1() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {-1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {-1, 32});
  auto op = c.NewCNodeWithBuildInfo("SoftmaxCrossEntropyWithLogits", {x0, x1}, {});
  c.SetOutput(op);
  return c.GetGraph();
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check main graph multiple output when no inline.
/// Expectation: After split pass, the output should be maketuple, and inputs should be gk node.
TEST_F(TestGraphSplit, no_inline_main_graph_multi_output) {
  SetGraphKernelFlags("--enable_expand_ops=SoftmaxCrossEntropyWithLogits");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(IsPrimitiveCNode(out, prim::kPrimMakeTuple), true);
  auto cnode = dyn_cast_ptr<CNode>(out);
  bool pattern_match = std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                                   [](const AnfNodePtr &anf_node) { return AnfUtils::IsGraphKernel(anf_node); });
  ASSERT_EQ(pattern_match, true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 5);
}

/// Feature: Test graph kernel splitter pass with enable_fusion_pattern_only
/// Description: op will expand, then split, check main graph multiple output when no inline.
/// Expectation: After split pass, the output should be maketuple, and inputs should be gk node.
TEST_F(TestGraphSplit, enable_fusion_pattern_only) {
  SetGraphKernelFlags("--enable_expand_ops=SoftmaxCrossEntropyWithLogits --enable_fusion_pattern_only=elemwise");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(IsPrimitiveCNode(out, prim::kPrimMakeTuple), true);
  auto cnode = dyn_cast_ptr<CNode>(out);
  bool pattern_match = std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                                   [](const AnfNodePtr &anf_node) { return AnfUtils::IsGraphKernel(anf_node); });
  ASSERT_EQ(pattern_match, true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 8);
}

/// Feature: Test graph kernel splitter pass with disable_fusion_pattern
/// Description: op will expand, then split, check main graph multiple output when no inline.
/// Expectation: After split pass, the output should be maketuple, and inputs should be gk node.
TEST_F(TestGraphSplit, disable_fusion_pattern) {
  SetGraphKernelFlags("--enable_expand_ops=SoftmaxCrossEntropyWithLogits --disable_fusion_pattern=reduce");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(IsPrimitiveCNode(out, prim::kPrimMakeTuple), true);
  auto cnode = dyn_cast_ptr<CNode>(out);
  bool pattern_match = std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                                   [](const AnfNodePtr &anf_node) { return AnfUtils::IsGraphKernel(anf_node); });
  ASSERT_EQ(pattern_match, true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 8);
}

/// Feature: Test graph kernel splitter pass with json
/// Description: op will expand, then split, check main graph multiple output when no inline.
/// Expectation: After split pass, the output should be maketuple, and inputs should be gk node.
TEST_F(TestGraphSplit, flag_path) {
  SetGraphKernelFlags("--path=graph_kernel_config/1.json");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(IsPrimitiveCNode(out, prim::kPrimMakeTuple), true);
  auto cnode = dyn_cast_ptr<CNode>(out);
  bool pattern_match = std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                                   [](const AnfNodePtr &anf_node) { return AnfUtils::IsGraphKernel(anf_node); });
  ASSERT_EQ(pattern_match, true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 5);
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check gk graph single output when no inline.
/// Expectation: After split pass, the gk single output should not be maketuple.
TEST_F(TestGraphSplit, no_inline_gk_graph_single_output) {
  SetGraphKernelFlags("--enable_expand_ops=SoftmaxCrossEntropyWithLogits");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  size_t gk_single_return_num = 0;
  auto nodes = TopoSort(fg->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto fg = GetCNodeFuncGraph(node);
      auto out = fg->output();
      if (!IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
        gk_single_return_num += 1;
      }
    }
  }
  ASSERT_EQ(gk_single_return_num, 3);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 5);
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check gk graph multiple output when no inline.
/// Expectation: After split pass, the gk multiple output should be maketuple, and should have tuplegetitem.
TEST_F(TestGraphSplit, no_inline_gk_graph_multi_output) {
  SetGraphKernelFlags("--enable_expand_ops=SoftmaxCrossEntropyWithLogits");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  size_t gk_make_tuple_num = 0;
  size_t main_get_item_num = 0;

  auto nodes = TopoSort(fg->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto fg = GetCNodeFuncGraph(node);
      auto out = fg->output();
      if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
        gk_make_tuple_num += 1;
      }
    } else if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      main_get_item_num += 1;
    }
  }
  ASSERT_EQ(gk_make_tuple_num, 2);
  ASSERT_EQ(main_get_item_num, 4);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 5);
}

bool StubNeedInline(const CommonSplitSchemer*, size_t group_id) {
  std::vector<int> need_inline{1,1,0,1,0};
  return need_inline[group_id] != 0;
}

FuncGraphPtr ConstructGraph_2() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {-1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {-1, 32});
  auto op = c.NewCNodeWithBuildInfo("SiLUGrad", {x0, x1}, {});
  auto op2 = c.NewCNodeWithBuildInfo("AssignAdd", {op, x0}, {});
  c.SetOutput(op2);
  return c.GetGraph();
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check main graph single output when no inline.
/// Expectation: After split pass, the output should be gk node.
TEST_F(TestGraphSplit, no_inline_main_single_out) {
  SetGraphKernelFlags("--enable_expand_ops=SiLUGrad,AssignAdd");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_2();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(AnfUtils::IsGraphKernel(out), true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 5);
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check main graph node and gk node when partial inline.
/// Expectation: After split pass, the main graph should have inline node and gk node.
TEST_F(TestGraphSplit, partial_inline) {
  SetGraphKernelFlags("--enable_expand_ops=SiLUGrad,AssignAdd");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_2();
  MOCKER_CPP(&CommonSplitSchemer::NeedInline, bool (*)(const CommonSplitSchemer*, size_t)).stubs().will(invoke(StubNeedInline));
  MOCKER_CPP(SetKernelInfo, void (*)(const FuncGraphPtr &func_graph)).stubs().will(ignoreReturnValue());

  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(AnfUtils::IsGraphKernel(out), true);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 2);
  ASSERT_EQ(GetAllCNodes(fg).size(), 9);
  GlobalMockObject::verify();
}

/// Feature: Test graph kernel splitter pass
/// Description: op will expand, then split, check main graph node when all inline.
/// Expectation: After split pass, the main graph should have all inline node and no gk node.
TEST_F(TestGraphSplit, all_inline) {
  SetGraphKernelFlags("--enable_expand_ops=SiLUGrad,AssignAdd");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_2();
  MOCKER_CPP(&CommonSplitSchemer::NeedInline, bool (*)(const CommonSplitSchemer*, size_t)).stubs().will(returnValue(true));
  MOCKER_CPP(SetKernelInfo, void (*)(const FuncGraphPtr &func_graph)).stubs().will(ignoreReturnValue());

  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::StaticShapeCluster>(),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  auto out = fg->output();
  ASSERT_EQ(AnfUtils::IsGraphKernel(out), false);
  ASSERT_EQ(GetAllGKNodes(fg).size(), 0);
  ASSERT_EQ(GetAllCNodes(fg).size(), 11);
  GlobalMockObject::verify();
}
}  // namespace mindspore
