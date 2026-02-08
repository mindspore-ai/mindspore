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
class TestGraphClusterCheck : public GraphKernelCommonTestSuite {};

namespace {
FuncGraphPtr ConstructGraph_1(TypePtr type) {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", type, {32, 32});
  auto x1 = c.NewTensorInput("x0", kFloat32, {32, 32});
  auto x2 = c.NewTensorInput("x1", kFloat32, {32, 32});
  auto op = c.NewCNodeWithBuildInfo("Select", {x0, x1, x2}, {});
  c.SetOutput(op);
  return c.GetGraph();
}

FuncGraphPtr ConstructGraph_2(TypePtr type) {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", type, {32, 32});
  auto op = c.NewCNodeWithBuildInfo("Log", {x0}, {});
  c.SetOutput(op);
  return c.GetGraph();
}
}  // namespace

/// Feature: Test graph kernel cluster pass
/// Description: select op first input should be bool
/// Expectation: After cluster pass, the gk node and main node should be correct.
TEST_F(TestGraphClusterCheck, select) {
  SetDeviceTarget(kAscendDevice);
  auto fg1 = ConstructGraph_1(kFloat32);
  RunPass(fg1, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg1).size(), 0);
  auto fg2 = ConstructGraph_1(kBool);
  RunPass(fg2, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg2).size(), 1);
}

/// Feature: Test graph kernel cluster pass
/// Description: input type should be same as output type
/// Expectation: After cluster pass, the gk node and main node should be correct.
TEST_F(TestGraphClusterCheck, log) {
  SetDeviceTarget(kAscendDevice);
  auto fg1 = ConstructGraph_2(kInt32);
  RunPass(fg1, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg1).size(), 0);
  auto fg2 = ConstructGraph_2(kFloat32);
  RunPass(fg2, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg2).size(), 1);
}
}  // namespace mindspore::graphkernel::test
