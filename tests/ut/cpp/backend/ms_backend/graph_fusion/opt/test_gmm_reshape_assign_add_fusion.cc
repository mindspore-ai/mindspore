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
#include <vector>
#include <string>

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "abstract/abstract_value.h"
#include "include/utils/utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "utils/ms_context.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"

namespace mindspore::graphkernel::test {
struct GmmFusionTestParam {
  ShapeVector shape_a;
  ShapeVector shape_b;
};
class TestGroupedMatmulFusion : public GraphKernelCommonTestSuite,
                                public testing::WithParamInterface<GmmFusionTestParam> {
 public:
  TestGroupedMatmulFusion() {}
};

TEST_P(TestGroupedMatmulFusion, test_gmm_reshape_assign_fusion) {
  // get params
  const auto &param = GetParam();
  SetGraphKernelFlags("--enable_cluster_ops=GroupedMatmul,Reshape");
  SetDeviceTarget(kAscendDevice);
  SPLIT_MODEL_REGISTER(kAscendDevice, graphkernel::inner::SplitModelAscend);

  // construct graph, set abstract and kernel info.
  ConstructGraph c;
  AnfNodePtr x1 = c.NewTensorInput("x1", kFloat16, param.shape_a);
  AnfNodePtr x2 = c.NewTensorInput("x2", kFloat16, param.shape_b);
  AnfNodePtr x3 = c.NewTensorInput("x3", kFloat16, ShapeVector{0});
  AnfNodePtr x4 = c.NewTensorInput("x4", kUInt64, ShapeVector{0});
  AnfNodePtr x5 = c.NewTensorInput("x5", kFloat32, ShapeVector{0});
  AnfNodePtr x6 = c.NewTensorInput("x6", kFloat16, ShapeVector{0});
  AnfNodePtr x7 = c.NewTensorInput("x7", kFloat16, ShapeVector{0});
  AnfNodePtr group_list = c.NewTensorInput("group_list", kInt64, ShapeVector{kDim6});
  auto x9 = c.NewValueNode(MakeValue<int64_t>(3));
  auto x10 = c.NewValueNode(MakeValue<int64_t>(2));
  auto x11 = c.NewValueNode(MakeValue<bool>(false));
  auto x12 = c.NewValueNode(MakeValue<bool>(false));
  auto gmm = c.NewCNodeWithoutInfer("GroupedMatmul", {x1, x2, x3, x4, x5, x6, x7, group_list, x9, x10, x11, x12}, {});
  gmm->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{
    std::make_shared<abstract::AbstractTensor>(kFloat16, ShapeVector{kDim6, param.shape_a[0], param.shape_b.back()})}));
  c.SetGeneralBuildInfo(gmm);
  auto parm_shape = ShapeVector{SizeToLong(kDim6) * param.shape_a[0], param.shape_b.back()};
  auto out = c.NewTensorInput("out", kFloat32, parm_shape);
  auto getitem = c.NewCNodeWithBuildInfo("TupleGetItem", {gmm, c.NewValueNode(MakeValue<int64_t>(0))}, {});
  auto reshape = c.NewCNodeWithBuildInfo("Reshape", {getitem, c.NewValueNode(MakeValue(parm_shape))});
  auto assign_add = c.NewCNodeWithBuildInfo("AssignAdd", {out, reshape});
  c.SetOutput(assign_add);
  RunPass(c.GetGraph(),
          {std::make_shared<ConvertFrontEndToGraphKernel>(), std::make_shared<GraphKernelExpanderCloud>(),
           std::make_shared<StaticShapeCluster>(), std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});

  // // check whether the cluster is successful
  auto fg = c.GetGraph();
  ASSERT_EQ(GetAllGKNodes(fg).size(), 1);
}

INSTANTIATE_TEST_CASE_P(TestGroupedMatmulCases, TestGroupedMatmulFusion,
                        testing::Values(GmmFusionTestParam{{1280, 512}, {512, 2560}},
                                        GmmFusionTestParam{{1280, 512}, {512, 2560}},
                                        GmmFusionTestParam{{2560, 256}, {256, 2560}}));
}  // namespace mindspore::graphkernel::test
