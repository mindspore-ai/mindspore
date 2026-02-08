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
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/ms_backend/graph_fusion/transpose_matmul_fusion.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"

namespace mindspore::graphkernel::test {
struct GmmTestParams {
  ShapeVector shape_a;
  ShapeVector shape_b;
  bool ori_trans_a;
  bool ori_trans_b;
  bool input_a_transpose;
  bool input_b_transpose;
};
class TestTransposeGroupedMatmulFusion : public GraphKernelCommonTestSuite,
                                         public testing::WithParamInterface<GmmTestParams> {
 public:
  TestTransposeGroupedMatmulFusion() {}
};

TEST_P(TestTransposeGroupedMatmulFusion, test_transpose_matmul_fusion) {
  // get params
  const auto &param = GetParam();
  SetGraphKernelFlags("--enable_cluster_ops=GroupedMatmul");
  // construct graph, set abstract and kernel info.
  ConstructGraph c;
  AnfNodePtr x1 = c.NewTensorInput("x1", kFloat16, param.shape_a);
  AnfNodePtr x2 = c.NewTensorInput("x2", kFloat16, param.shape_b);
  AnfNodePtr x3 = c.NewTensorInput("x3", kFloat16, ShapeVector{0});
  AnfNodePtr x4 = c.NewTensorInput("x4", kUInt64, ShapeVector{0});
  AnfNodePtr x5 = c.NewTensorInput("x5", kFloat32, ShapeVector{0});
  AnfNodePtr x6 = c.NewTensorInput("x6", kFloat16, ShapeVector{0});
  AnfNodePtr x7 = c.NewTensorInput("x7", kFloat16, ShapeVector{0});
  AnfNodePtr group_list = c.NewTensorInput("group_list", kInt64, ShapeVector{param.shape_b[0]});
  auto x9 = c.NewValueNode(MakeValue<int64_t>(3));
  auto x10 = c.NewValueNode(MakeValue<int64_t>(0));
  auto x11 = c.NewValueNode(MakeValue<bool>(param.ori_trans_a));
  auto x12 = c.NewValueNode(MakeValue<bool>(param.ori_trans_b));
  if (param.input_a_transpose) {
    auto perm = c.NewValueNode(MakeValue(ShapeVector{1, 0}));
    x1 = c.NewCNodeWithBuildInfo("Transpose", {x1, perm}, {});
  }
  if (param.input_b_transpose) {
    auto perm = c.NewValueNode(ShapeVector{0, 2, 1});
    x2 = c.NewCNodeWithBuildInfo("Transpose", {x2, perm}, {});
  }
  auto gmm = c.NewCNodeWithoutInfer("GroupedMatmul", {x1, x2, x3, x4, x5, x6, x7, group_list, x9, x10, x11, x12}, {});
  gmm->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kFloat16, ShapeVector{param.shape_a[0], param.shape_b.back()}));
  c.SetGeneralBuildInfo(gmm);
  c.SetOutput(gmm);

  // run pass for ir transformation
  RunPass(c.GetGraph(), {std::make_shared<ConvertFrontEndToGraphKernel>(), std::make_shared<TransposeMatmulFusion>()});

  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("x3")
    .AddVar("x4")
    .AddVar("x5")
    .AddVar("x6")
    .AddVar("x7")
    .AddVar("group_list")
    .AddCNode("GroupedMatmul", {
                                 std::make_shared<Primitive>("GroupedMatmul"),
                                 "x1",
                                 "x2",
                                 "x3",
                                 "x4",
                                 "x5",
                                 "x6",
                                 "x7",
                                 "group_list",
                               });

  // check whether the transformation is successful
  auto output = c.GetGraph()->output();
  EXPECT_TRUE(checker.build_pattern_map(output));
  auto cnode = output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetCNodePrimitive(cnode);
  auto trans_a = GetValue<bool>(prim->GetAttr(kTransposeA));
  auto trans_b = GetValue<bool>(prim->GetAttr(kTransposeB));
  EXPECT_EQ(trans_a, param.ori_trans_a ^ param.input_a_transpose);
  EXPECT_EQ(trans_b, param.ori_trans_b ^ param.input_b_transpose);
}

INSTANTIATE_TEST_CASE_P(TestTransposeGroupedMatmulCases, TestTransposeGroupedMatmulFusion,
                        testing::Values(GmmTestParams{{128, 256}, {3, 512, 256}, false, false, false, true},
                                        GmmTestParams{{128, 256}, {4, 512, 256}, true, true, false, true},
                                        GmmTestParams{{256, 256}, {5, 256, 256}, false, false, true, false}));
}  // namespace mindspore::graphkernel::test
