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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"

namespace mindspore::graphkernel::test {
/// Feature: GraphKernelCluster
/// Description: ReduceSum with skip_mode True but axis not empty should be clustered
/// Expectation: Add and ReduceSum are all clustered
TEST_F(GraphKernelCommonTestSuite, reduce_sum_cluster) {
  ConstructGraph gb;
  auto x = gb.NewTensorInput("x", kFloat32, {1024, 1024});
  auto y = gb.NewTensorInput("y", kFloat32, {1024, 1024});
  auto add = gb.NewCNodeWithBuildInfo("Add", {x, y});
  auto axis = gb.NewValueNode(MakeValue(std::vector<int64_t>{0}));
  auto keep_dims = gb.NewValueNode(MakeValue<bool>(false));
  auto skip_mode = gb.NewValueNode(MakeValue<bool>(true));
  auto reduce = gb.NewCNodeWithBuildInfo("ReduceSum", {add, axis, keep_dims, skip_mode});
  gb.SetOutput(reduce);
  auto fg = gb.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
               std::make_shared<graphkernel::StaticShapeCluster>()});
  EXPECT_EQ(GetAllGKNodes(fg).size(), 1);
}
}  // namespace mindspore::graphkernel::test
