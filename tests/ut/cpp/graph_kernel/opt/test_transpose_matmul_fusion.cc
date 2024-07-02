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

#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/common/graph_kernel/transpose_matmul_fusion.h"

namespace mindspore {
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
class TestTransposeMatMulFusion : public UT::Common, public testing::WithParamInterface<TestParams> {
 public:
  TestTransposeMatMulFusion() {}
};

TEST_P(TestTransposeMatMulFusion, test_transpose_matmul_fusion) {
  // get params
  const auto &param = GetParam();
  // construct graph, set abstract and kernel info.
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat16, param.shape_a);
  auto x2 = c.NewTensorInput("x2", kFloat16, param.shape_b);
  auto x3 = c.NewValueNode(MakeValue<bool>(param.ori_trans_a));
  auto x4 = c.NewValueNode(MakeValue<bool>(param.ori_trans_b));
  auto perm = c.NewValueNode(MakeValue(param.perm));
  AnfNodePtr new_x1 = x1;
  AnfNodePtr new_x2 = x2;
  if (param.input_a_transpose) {
    new_x1 = c.NewCNode("Transpose", {x1, perm}, {});
    c.SetGeneralBuildInfo(new_x1);
  }
  if (param.input_b_transpose) {
    new_x2 = c.NewCNode("Transpose", {x2, perm}, {});
    c.SetGeneralBuildInfo(new_x2);
  }
  auto matmul = c.NewCNode(param.op_name, {new_x1, new_x2, x3, x4}, {});
  c.SetGeneralBuildInfo(matmul);
  c.SetOutput(matmul);

  // run pass for ir transformation
  test::RunPass(c.GetGraph(), {std::make_shared<graphkernel::TransposeMatmulFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1").AddVar("x2").AddVar("trans_a").AddVar("trans_b").AddCNode(
    param.op_name, {std::make_shared<Primitive>(param.op_name), "x1", "x2", "trans_a", "trans_b"});

  // check whether the transformation  is success
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
  testing::Values(TestParams{"MatMul", {128, 256}, {512, 256}, {1, 0}, false, false, false, true},
                  TestParams{"MatMul", {256, 256}, {256, 256}, {1, 0}, false, false, true, false},
                  TestParams{"MatMul", {256, 256}, {256, 256}, {1, 0}, false, true, false, true},
                  TestParams{"MatMul", {256, 256}, {256, 256}, {1, 0}, true, false, true, false},
                  TestParams{"BatchMatMul", {4, 128, 256}, {1, 512, 256}, {0, 2, 1}, false, false, false, true},
                  TestParams{
                    "BatchMatMul", {3, 4, 128, 256}, {1, 4, 512, 256}, {0, 1, 3, 2}, false, false, false, true}));
}  // namespace mindspore
