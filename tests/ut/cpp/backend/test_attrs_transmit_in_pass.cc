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
#include <memory>
#include "common/graph_optimizer_test_framework.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "common/common_test.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/matmul_assignadd_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_matmul_split_fusion.h"
#include "mindspore/ccsrc/backend/ge_backend/pass/fused_cast_add.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"

namespace mindspore {
class TestAttrsTransmitInPass : public UT::Common {
 public:
  TestAttrsTransmitInPass() {}
};

/// Feature: Test stream_id cnode attr transmit in Pass(InferenceMatmulSplitFusion)
/// Description: Fusion success with stream_id=2 for all nodes
/// Expectation: After optimize, match MatmulSplitOut2 with stream_id=2
TEST_F(TestAttrsTransmitInPass, test_stream_id_transmit_in_pass_succ) {
  auto ms_context = MsContext::GetInstance();
  ms_context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {8, 1536, 2048});
  auto input_w = c.NewTensorInput("input_w", kBFloat16, {1792, 2048});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(false));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(true));
  auto shape1 = c.NewValueNode(MakeValue<std::vector<int64_t>>({12288, 2048}));
  auto pre_reshape = c.NewCNode("Reshape", {input_x, shape1}, {{"stream_id", MakeValue((int64_t)2)}});
  auto matmul =
    c.NewCNode("MatMul", {pre_reshape, input_w, transpose_a, transpose_b}, {{"stream_id", MakeValue((int64_t)2)}});
  auto shape2 = c.NewValueNode(MakeValue<std::vector<int64_t>>({8, 1536, 1792}));
  auto reshape = c.NewCNode("Reshape", {matmul, shape2}, {{"stream_id", MakeValue((int64_t)2)}});
  auto size = c.NewValueNode(MakeValue<std::vector<int64_t>>({1600, 192}));
  auto axis = c.NewValueNode(MakeValue<int64_t>(2));
  auto split = c.NewCNode("SplitWithSize", {reshape, size, axis}, {{"stream_id", MakeValue((int64_t)2)}});

  c.SetOutput(split);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x")
    .AddVar("input_w")
    .AddVar("reshape_tuple")
    .AddCNode("MatmulSplitOut2",
              {std::make_shared<Primitive>("MatmulSplitOut2"), "input_x", "input_w", "reshape_tuple"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  auto fusion_node = c.GetGraph()->output()->cast<CNodePtr>();
  EXPECT_TRUE(fusion_node->HasAttr("stream_id"));
  EXPECT_TRUE(GetValue<int64_t>(fusion_node->GetAttr("stream_id")) == 2);
}

/// Feature: Test stream_id cnode attr transmit in Pass(InferenceMatmulSplitFusion)
/// Description: Fusion fail with stream_id not same for all nodes
/// Expectation: After optimize, not match MatmulSplitOut2
TEST_F(TestAttrsTransmitInPass, test_stream_id_transmit_in_pass_fail) {
  auto ms_context = MsContext::GetInstance();
  ms_context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {8, 1536, 2048});
  auto input_w = c.NewTensorInput("input_w", kBFloat16, {1792, 2048});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(false));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(true));
  auto shape1 = c.NewValueNode(MakeValue<std::vector<int64_t>>({12288, 2048}));
  auto pre_reshape = c.NewCNode("Reshape", {input_x, shape1}, {{"stream_id", MakeValue((int64_t)2)}});
  auto matmul =
    c.NewCNode("MatMul", {pre_reshape, input_w, transpose_a, transpose_b}, {{"stream_id", MakeValue((int64_t)4)}});
  auto shape2 = c.NewValueNode(MakeValue<std::vector<int64_t>>({8, 1536, 1792}));
  auto reshape = c.NewCNode("Reshape", {matmul, shape2}, {{"stream_id", MakeValue((int64_t)2)}});
  auto size = c.NewValueNode(MakeValue<std::vector<int64_t>>({1600, 192}));
  auto axis = c.NewValueNode(MakeValue<int64_t>(2));
  auto split = c.NewCNode("SplitWithSize", {reshape, size, axis}, {{"stream_id", MakeValue((int64_t)2)}});

  c.SetOutput(split);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x")
    .AddVar("input_w")
    .AddVar("reshape_tuple")
    .AddCNode("MatmulSplitOut2",
              {std::make_shared<Primitive>("MatmulSplitOut2"), "input_x", "input_w", "reshape_tuple"});
  EXPECT_FALSE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: Test cube_num cnode attr transmit in PatternProcessPass(MatmulAssignaddFusion)
/// Description: Fusion success with cube_num=8 for all nodes
/// Expectation: After optimize, match InplaceMatmulAdd with cube_num=8
TEST_F(TestAttrsTransmitInPass, test_cube_num_transmit_in_patternprocesspass_succ) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {2048, 1536});
  auto weight = c.NewTensorInput("weight", kBFloat16, {2048, 1792});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(true));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul =
    c.NewCNode("MatMul", {input_x, weight, transpose_a, transpose_b}, {{"cube_num", MakeValue((int64_t)8)}});
  auto cast_type = c.NewValueNode(MakeValue<int64_t>(43));
  auto cast = c.NewCNode("Cast", {matmul, cast_type}, {{"cube_num", MakeValue((int64_t)8)}});
  auto out = c.NewTensorInput("out", kFloat32, {1536, 1792});
  auto assign_add = c.NewCNode("AssignAdd", {out, cast}, {{"cube_num", MakeValue((int64_t)8)}});

  c.SetOutput(assign_add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulAssignaddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("weight").AddVar("out").AddCNode(
    "InplaceMatmulAdd", {std::make_shared<Primitive>("InplaceMatmulAdd"), "input_x", "weight", "out"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  auto fusion_node = c.GetGraph()->output()->cast<CNodePtr>();
  EXPECT_TRUE(fusion_node->HasAttr("cube_num"));
  EXPECT_TRUE(GetValue<int64_t>(fusion_node->GetAttr("cube_num")) == 8);
}

/// Feature: Test cube_num cnode attr transmit in PatternProcessPass(MatmulAssignaddFusion)
/// Description: Fusion fail with cube_num not same for all nodes
/// Expectation: After optimize, not match InplaceMatmulAdd
TEST_F(TestAttrsTransmitInPass, test_cube_num_transmit_in_patternprocesspass_fail) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {2048, 1536});
  auto weight = c.NewTensorInput("weight", kBFloat16, {2048, 1792});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(true));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul =
    c.NewCNode("MatMul", {input_x, weight, transpose_a, transpose_b}, {{"cube_num", MakeValue((int64_t)8)}});
  auto cast_type = c.NewValueNode(MakeValue<int64_t>(43));
  auto cast = c.NewCNode("Cast", {matmul, cast_type}, {{"cube_num", MakeValue((int64_t)8)}});
  auto out = c.NewTensorInput("out", kFloat32, {1536, 1792});
  auto assign_add = c.NewCNode("AssignAdd", {out, cast}, {{"cube_num", MakeValue((int64_t)9)}});

  c.SetOutput(assign_add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulAssignaddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("weight").AddVar("out").AddCNode(
    "InplaceMatmulAdd", {std::make_shared<Primitive>("InplaceMatmulAdd"), "input_x", "weight", "out"});
  EXPECT_FALSE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: Test vector_num cnode attr transmit in PatternToPatternPass(FusedCastAdd)
/// Description: Fusion success with vector_num=4 for all nodes
/// Expectation: After optimize, match Add with cube_num=4
TEST_F(TestAttrsTransmitInPass, test_vector_num_transmit_in_patterntopatternpass_succ) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kFloat16, {2048});
  auto input_y = c.NewTensorInput("input_y", kFloat32, {2048});
  auto cast_type = c.NewValueNode(MakeValue<int64_t>(43));
  auto cast = c.NewCNode("Cast", {input_x, cast_type}, {{"vector_num", MakeValue((int64_t)4)}});
  auto add = c.NewCNode("Add", {input_y, cast}, {{"vector_num", MakeValue((int64_t)4)}});

  c.SetOutput(add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::FusedCastAdd>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("input_y").AddCNode(
    "Add", {std::make_shared<Primitive>("Add"), "input_x", "input_y"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  auto fusion_node = c.GetGraph()->output()->cast<CNodePtr>();
  EXPECT_TRUE(fusion_node->HasAttr("vector_num"));
  EXPECT_TRUE(GetValue<int64_t>(fusion_node->GetAttr("vector_num")) == 4);
}

/// Feature: Test vector_num cnode attr transmit in PatternToPatternPass(FusedCastAdd)
/// Description: Fusion fail with vector_num not same for all nodes
/// Expectation: After optimize, not match Add
TEST_F(TestAttrsTransmitInPass, test_vector_num_transmit_in_patterntopatternpass_fail) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kFloat16, {2048});
  auto input_y = c.NewTensorInput("input_y", kFloat32, {2048});
  auto cast_type = c.NewValueNode(MakeValue<int64_t>(43));
  auto cast = c.NewCNode("Cast", {input_x, cast_type}, {{"vector_num", MakeValue((int64_t)4)}});
  auto add = c.NewCNode("Add", {input_y, cast}, {{"vector_num", MakeValue((int64_t)5)}});

  c.SetOutput(add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::FusedCastAdd>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("input_y").AddCNode(
    "Add", {std::make_shared<Primitive>("Add"), "input_x", "input_y"});
  auto output = c.GetGraph()->output()->cast<CNodePtr>();
  EXPECT_TRUE(output->HasAttr("vector_num"));
  EXPECT_TRUE(GetValue<int64_t>(output->GetAttr("vector_num")) == 5);
}
}  // namespace mindspore
