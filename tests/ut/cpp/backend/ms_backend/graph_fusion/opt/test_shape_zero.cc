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

#include <mindspore/core/include/ir/core_ops_primitive.h>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel::test {
/// Feature: test graph kernel cluster and expander pass
/// Description: node shape contains zero
/// Expectation: if node shape contains zero, then it should not be fused
TEST_F(GraphKernelCommonTestSuite, shape_zero) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=BroadcastTo --enable_expand_ops=OnesLike,ZerosLike");
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {1, 0});
  auto p1 = c.NewTensorInput("p1", kFloat32, {1, 0});
  auto p2 = c.NewTensorInput("p2", kFloat32, {1});
  auto shape = c.NewValueNode(MakeValue(ShapeVector{1, 0}));
  auto y0 = c.NewCNodeWithBuildInfo("OnesLike", {p0});
  auto y1 = c.NewCNodeWithBuildInfo("ZerosLike", {p1});
  auto y2 = c.NewCNodeWithBuildInfo("BroadcastTo", {p2, shape});
  auto y3 = c.NewCNodeWithBuildInfo("AddN", {y0, y1, y2});
  c.SetOutput(y3);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
               std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 0);
}
}  // namespace mindspore::graphkernel::test
