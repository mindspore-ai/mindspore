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
#include "primitive/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/matmul_assignadd_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
class MatmulAssignaddFusion : public UT::Common {
 public:
  MatmulAssignaddFusion() {}
};

/// Feature: A backend ir fusion pass: MatmulAssignaddFusion
/// Description: Convert MatMul+AssignAdd to InplaceMatmulAdd for kbk
/// Expectation: After optimize, match InplaceMatmulAdd.
TEST_F(MatmulAssignaddFusion, test_matmul_assignadd_fusion) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {2048, 1536});
  auto weight = c.NewTensorInput("weight", kBFloat16, {2048, 1792});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(true));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul = c.NewCNode("MatMul", {input_x, weight, transpose_a, transpose_b}, {});
  auto out = c.NewTensorInput("out", kFloat32, {1536, 1792});
  auto assign_add = c.NewCNode("AssignAdd", {out, matmul});

  c.SetOutput(assign_add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulAssignaddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("weight").AddVar("out").AddCNode(
    "InplaceMatmulAdd", {std::make_shared<Primitive>("InplaceMatmulAdd"), "input_x", "weight", "out"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend ir fusion pass: MatmulAssignaddFusion
/// Description: Convert MatMul+Cast+AssignAdd to InplaceMatmulAdd for kbk
/// Expectation: After optimize, match InplaceMatmulAdd.
TEST_F(MatmulAssignaddFusion, test_matmul_assignadd_fusion2) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {2048, 1536});
  auto weight = c.NewTensorInput("weight", kBFloat16, {2048, 1792});
  auto transpose_a = c.NewValueNode(MakeValue<bool>(true));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul = c.NewCNode("MatMul", {input_x, weight, transpose_a, transpose_b}, {});
  auto cast_type = c.NewValueNode(MakeValue<int64_t>(43));
  auto cast = c.NewCNode("Cast", {matmul, cast_type}, {});
  auto out = c.NewTensorInput("out", kFloat32, {1536, 1792});
  auto assign_add = c.NewCNode("AssignAdd", {out, cast});

  c.SetOutput(assign_add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulAssignaddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x").AddVar("weight").AddVar("out").AddCNode(
    "InplaceMatmulAdd", {std::make_shared<Primitive>("InplaceMatmulAdd"), "input_x", "weight", "out"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
