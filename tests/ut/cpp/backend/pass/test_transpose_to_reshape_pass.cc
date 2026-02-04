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
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/common/pass/transpose_to_reshape_pass.h"

namespace mindspore {
class TransposeToReshapePass : public UT::Common {
 public:
  TransposeToReshapePass() {}
};

/// Feature: A backend pass: TransposeToReshapePass
/// Description: Convert Transpose to Reshape under certain condition
/// Expectation: After optimize, match Reshape
TEST_F(TransposeToReshapePass, test_transpose_to_reshape_succ) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kBFloat16, {1, 1, 2, 2, 1, 3, 4});
  auto permute = c.NewValueNode(MakeValue<std::vector<int64_t>>(std::vector<int64_t>{2, 1, 3, 5, 0, 4, 6}));
  auto transpose = c.NewCNode("Transpose", {input, permute}, {});

  c.SetOutput(transpose);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::TransposeToReshapePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("shape").AddCNode(
    "reshape", {std::make_shared<Primitive>("Reshape"), "input", "shape"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: TransposeToReshapePass
/// Description: Cannot convert Transpose to Reshape when data is rearranged
/// Expectation: After optimize, still Transpose
TEST_F(TransposeToReshapePass, test_transpose_to_reshape_fail) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kBFloat16, {1, 1, 2, 2, 1, 3, 4});
  auto permute = c.NewValueNode(MakeValue<std::vector<int64_t>>(std::vector<int64_t>{3, 1, 2, 5, 0, 4, 6}));
  auto transpose = c.NewCNode("Transpose", {input, permute}, {});

  c.SetOutput(transpose);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::TransposeToReshapePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("permute").AddCNode(
    "transpose", {std::make_shared<Primitive>("Transpose"), "input", "permute"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: TransposeToReshapePass
/// Description: Cannot convert Transpose to Reshape when data is rearranged(with negative dim)
/// Expectation: After optimize, still Transpose
TEST_F(TransposeToReshapePass, test_transpose_to_reshape_fail_neg_dim) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kBFloat16, {4, 8});
  auto permute = c.NewValueNode(MakeValue<std::vector<int64_t>>(std::vector<int64_t>{-1, 0}));
  auto transpose = c.NewCNode("Transpose", {input, permute}, {});

  c.SetOutput(transpose);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::TransposeToReshapePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("permute").AddCNode(
    "transpose", {std::make_shared<Primitive>("Transpose"), "input", "permute"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
