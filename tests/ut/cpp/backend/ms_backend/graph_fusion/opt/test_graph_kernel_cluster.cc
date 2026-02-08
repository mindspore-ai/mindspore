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
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"

namespace mindspore::graphkernel::test {
class TestGraphCluster : public GraphKernelCommonTestSuite {};

namespace {
FuncGraphPtr ConstructGraph_1() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {-1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {-1, 32});
  auto op = c.NewCNodeWithBuildInfo("Sub", {x0, x1}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {op, op}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Sub", {op1, op}, {});
  auto op3 = c.NewCNodeWithBuildInfo("Add", {op1, op2}, {});
  c.SetOutput(op3);
  return c.GetGraph();
}

FuncGraphPtr ConstructGraph_2() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {-1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {-1, 32});
  auto x2 = c.NewTensorInput("x2", kFloat32, {-1, 32});
  auto op0 = c.NewCNodeWithBuildInfo("Mul", {x0, x1}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {op0, op0}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Mul", {op1, op1}, {});
  auto op3 = c.NewCNodeWithBuildInfo("Add", {x2, x2}, {});
  auto op4 = c.NewCNodeWithBuildInfo("Add", {op2, op3}, {});
  auto op5 = c.NewCNodeWithBuildInfo("Add", {op3, op3}, {});
  auto op6 = c.NewCNodeWithBuildInfo("Mul", {op5, op5}, {});
  auto op7 = c.NewCNodeWithBuildInfo("Add", {op1, op6}, {});
  auto op8 = c.NewCNodeWithBuildInfo("MakeTuple", {op1, op2, op3, op4, op7}, {});
  c.SetOutput(op8);
  return c.GetGraph();
}
}  // namespace

/// Feature: Test graph kernel cluster pass
/// Description: op will cluster, check cluster and not cluster.
/// Expectation: After cluster pass, the gk node and main node should be correct.
TEST_F(TestGraphCluster, cluster_cut_max_id) {
  SetGraphKernelFlags("--disable_cluster_ops=Sub");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 2);
  ASSERT_EQ(GetAllCNodes(fg).size(), 4);
}

/// Feature: Test graph kernel cluster pass check circle
/// Description: op will cluster, use max id cut will have circle and use pre max id.
/// Expectation: After cluster pass, the gk node and main node should be correct.
TEST_F(TestGraphCluster, cluster_cut_max_id_circle) {
  SetGraphKernelFlags("--disable_cluster_ops=Mul");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_2();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 3);
}
}  // namespace mindspore::graphkernel::test
