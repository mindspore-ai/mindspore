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

#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/add_atomic_clean.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  std::string device;
  ShapeArray input_shape;
  ShapeVector axis;
  bool multiple_output{false};
};
}  // namespace

/// Feature: Test graph kernel AtomicClean pass
/// Description: AtomicClean pass on ReduceSum op
/// Expectation: After pass, ReduceSum input[0] should be a clean node(BroadcastTo).
class TestAtomicClean : public GraphKernelCommonTestSuite, public testing::WithParamInterface<Params> {};

TEST_P(TestAtomicClean, atomic_clean) {
  const auto &param = GetParam();
  SetDeviceTarget(param.device);
  SetGraphKernelFlags("--enable_cluster_ops_only=Add,ReduceSum");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, param.input_shape[0]);
  auto x1 = c.NewTensorInput("x1", kFloat32, param.input_shape[1]);
  auto y0 = c.NewCNodeWithBuildInfo("Add", {x0, x1});
  auto axis = c.NewValueNode(MakeValue(param.axis));
  auto keep_dims = c.NewValueNode(MakeValue<bool>(true));
  auto skip_mode = c.NewValueNode(MakeValue<bool>(false));
  auto y1 = c.NewCNodeWithBuildInfo("ReduceSum", {y0, axis, keep_dims, skip_mode});
  auto y2 = param.multiple_output ? c.NewCNodeWithBuildInfo("Mul", {y0, y1}) : c.NewCNodeWithBuildInfo("Abs", {y1});
  c.SetOutput(y2);
  auto fg = c.GetGraph();
  RunPass(fg,
          {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
           std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::AtomicCleanInserter>()});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && IsPrimitiveCNode(node, prim::kPrimDepend)) {
      // %0 = call_BroadcastTo()
      // %1 = call_fuse_ReduceSum(%0)
      // %2 = Depend(%0, %1)
      // %3 = op(%2)
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto broadcast_node = cnode->input(1);
      auto reduce_node = cnode->input(2);
      if (IsPrimitiveCNode(reduce_node, prim::kPrimTupleGetItem)) {
        reduce_node = reduce_node->cast<CNodePtr>()->input(1);
      }
      auto reduce_cnode = reduce_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(reduce_cnode);
      for (size_t i = 1; i < reduce_cnode->inputs().size(); ++i) {
        if (reduce_cnode->input(i) == broadcast_node) {
          check = true;
        }
      }
      break;
    }
  }
  EXPECT_EQ(check, true);
}

INSTANTIATE_TEST_CASE_P(TestPassAtomicClean, TestAtomicClean,
                        testing::Values(Params{"Ascend", {{4096, 1024}, {4096, 1024}}, {0}},
                                        Params{"Ascend", {{4096, 1024}, {4096, 1024}}, {0}, true},
                                        Params{"Ascend", {{1, 512, 1024}, {1, 512, 1024}}, {1}},
                                        Params{"Ascend", {{1, 512, 1024}, {1, 512, 1024}}, {0, 1}},
                                        Params{"Ascend", {{4, 512, 1024}, {4, 512, 1024}}, {1}},
                                        Params{"GPU", {{4096, 1024}, {4096, 1024}}, {-1}},
                                        Params{"GPU", {{4096, 1024}, {4096, 1024}}, {-2}}));
}  // namespace mindspore::graphkernel::test
