/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_cpu.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel::test {
/// Feature: test graph kernel op fusion with special data format
/// Description: op with NHWC format
/// Expectation: the data format of op keep same after graph kernel fusion pass
TEST_F(GraphKernelCommonTestSuite, special_format_nhwc) {
  SetDeviceTarget(kCPUDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=Cast,Reshape");
  SPLIT_MODEL_REGISTER(kCPUDevice, graphkernel::inner::SplitModelCpu);
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat16, {32, 32, 150, 150});
  auto x1 = c.NewTensorInput("x1", kFloat32, {16, 32, 1, 1});
  auto y0 = c.NewCNodeWithBuildInfo("Cast", {x1, c.NewValueNode<int64_t>(kNumberTypeFloat16)});
  auto y1 = c.NewCNodeWithBuildInfo("Reshape", {y0, c.NewValueNode<std::vector<int64_t>>({16, 32, 1, 1})});
  auto y2 = c.NewCNodeWithBuildInfo("Conv2D", {x0, y1},
                                    {{"kernel_size", MakeValue<ShapeVector>({1, 1})},
                                     {"mode", MakeValue<int64_t>(1)},
                                     {"out_channel", MakeValue<int64_t>(16)},
                                     {"pad", MakeValue<ShapeVector>({0, 0, 0, 0})},
                                     {"pad_mode", MakeValue<int64_t>(0)},
                                     {"format", MakeValue<std::string>("NCHW")},
                                     {"group", MakeValue<int64_t>(1)},
                                     {"stride", MakeValue<ShapeVector>({1, 1, 1, 1})},
                                     {"dilation", MakeValue<ShapeVector>({1, 1, 1, 1})}});
  // update data format
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(y1);
  ASSERT_NE(build_info, nullptr);
  build_info->SetOutputsFormat({"NHWC"});
  build_info = AnfAlgo::GetSelectKernelBuildInfo(y2);
  ASSERT_NE(build_info, nullptr);
  build_info->SetInputsFormat({"NHWC", "NHWC"});
  build_info->SetOutputsFormat({"NHWC"});
  c.SetOutput(y2);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(),
               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node->isa<CNode>() && AnfUtils::GetCNodeName(node) == "Conv2D") {
      auto cnode = node->cast<CNodePtr>();
      ASSERT_NE(cnode, nullptr);
      auto info = AnfAlgo::GetSelectKernelBuildInfo(cnode->input(2));
      ASSERT_NE(info, nullptr);
      check = (info->GetAllOutputFormats()[0] == "NHWC");
      break;
    }
  }
  ASSERT_EQ(check, true);
}
}  // namespace mindspore::graphkernel::test
