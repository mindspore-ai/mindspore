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
struct TestParams {
  std::string op_name;
  ShapeVector shape_a;
  ShapeVector shape_b;
  ShapeVector perm;
  bool ori_trans_a;
  bool ori_trans_b;
  bool input_a_transpose;
  bool input_b_transpose;
};
class TestTransposeMatMulFusion : public GraphKernelCommonTestSuite, public testing::WithParamInterface<TestParams> {
 public:
  TestTransposeMatMulFusion() {}
};

TEST_P(TestTransposeMatMulFusion, test_transpose_matmul_fusion) {
  // get params
  const auto &param = GetParam();
  // construct graph, set abstract and kernel info.
  ConstructGraph c;
  SetGraphKernelFlags("--enable_cluster_ops=BatchMatMul,MatMul");
  AnfNodePtr x1 = c.NewTensorInput("x1", kFloat16, param.shape_a);
  AnfNodePtr x2 = c.NewTensorInput("x2", kFloat16, param.shape_b);
  auto x3 = c.NewValueNode(MakeValue<bool>(param.ori_trans_a));
  auto x4 = c.NewValueNode(MakeValue<bool>(param.ori_trans_b));
  auto perm = c.NewValueNode(MakeValue(param.perm));
  if (param.input_a_transpose) {
    x1 = c.NewCNodeWithBuildInfo("Transpose", {x1, perm}, {});
  }
  if (param.input_b_transpose) {
    x2 = c.NewCNodeWithBuildInfo("Transpose", {x2, perm}, {});
  }
  auto matmul = c.NewCNodeWithBuildInfo(param.op_name, {x1, x2, x3, x4}, {});
  c.SetOutput(matmul);

  // run pass for ir transformation
  RunPass(c.GetGraph(), {std::make_shared<ConvertFrontEndToGraphKernel>(), std::make_shared<TransposeMatmulFusion>(),
                         std::make_shared<ConvertGraphKernelToFrontEnd>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1").AddVar("x2").AddVar("trans_a").AddVar("trans_b").AddCNode(
    param.op_name, {std::make_shared<Primitive>(param.op_name), "x1", "x2", "trans_a", "trans_b"});

  // check whether the transformation is successful
  auto output = c.GetGraph()->output();
  EXPECT_TRUE(checker.build_pattern_map(output));
  auto cnode = output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool trans_a = GetValue<bool>(cnode->input(kIndex3)->cast<ValueNodePtr>()->value());
  bool trans_b = GetValue<bool>(cnode->input(kIndex4)->cast<ValueNodePtr>()->value());
  EXPECT_EQ(trans_a, param.ori_trans_a ^ param.input_a_transpose);
  EXPECT_EQ(trans_b, param.ori_trans_b ^ param.input_b_transpose);
}

INSTANTIATE_TEST_CASE_P(
  TestTransposeMatMulCases, TestTransposeMatMulFusion,
  testing::Values(TestParams{"MatMul", {1280, 2560}, {5120, 2560}, {1, 0}, false, false, false, true},
                  TestParams{"MatMul", {2560, 2560}, {2560, 2560}, {1, 0}, false, false, true, false},
                  TestParams{"MatMul", {2560, 2560}, {2560, 2560}, {1, 0}, false, true, false, true},
                  TestParams{"MatMul", {2560, 2560}, {2560, 2560}, {1, 0}, true, false, true, false},
                  TestParams{"BatchMatMul", {4, 1280, 2560}, {1, 5120, 2560}, {0, 2, 1}, false, false, false, true},
                  TestParams{
                    "BatchMatMul", {3, 4, 1280, 2560}, {1, 4, 5120, 2560}, {0, 1, 3, 2}, false, false, false, true}));
}  // namespace mindspore::graphkernel::test
