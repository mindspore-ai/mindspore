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
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "common/common_test.h"
#include "backend/common/pass/mindir/space_batch_nd_attr_update.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"

namespace mindspore {
class SpaceBatchNDAttrUpdate : public UT::Common {
 public:
  SpaceBatchNDAttrUpdate() {}
};

/// Feature: A backend unify mindir pass: SpaceToBatchNDAttrUpdate
/// Description: Update block_shape and paddings attr to SpaceToBatchND
/// Expectation: After optimize, match SpaceToBatchND with updated block_shape and paddings attr
TEST_F(SpaceBatchNDAttrUpdate, test_space_to_batch_nd_attr_update) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat32, {8, 1, 2, 2});
  auto node = c.NewCNode("SpaceToBatchND", {input},
                         {{"block_shape", MakeValue(std::vector<int64_t>{2, 2})},
                          {"paddings", MakeValue(std::vector<std::vector<int64_t>>{std::vector<int64_t>{0, 0},
                                                                                   std::vector<int64_t>{0, 0}})}});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::SpaceToBatchNDAttrUpdate>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddCNode("updated", {std::make_shared<Primitive>("SpaceToBatchND"), "input"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  auto block_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(c.GetGraph()->output(), "block_shape");
  auto paddings = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(c.GetGraph()->output(), "paddings");
  EXPECT_TRUE(block_shape.size() == 3 && block_shape[0] == 1);
  EXPECT_TRUE(paddings.size() == 3);
}

/// Feature: A backend unify mindir pass: BatchToSpaceNDAttrUpdate
/// Description: Update block_shape and crops attr to BatchToSpaceND
/// Expectation: After optimize, match BatchToSpaceND with updated block_shape and crops attr
TEST_F(SpaceBatchNDAttrUpdate, test_batch_to_space_nd_attr_update) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat32, {8, 1, 2, 2});
  auto node = c.NewCNode(
    "BatchToSpaceND", {input},
    {{"block_shape", MakeValue(std::vector<int64_t>{2, 2})},
     {"crops", MakeValue(std::vector<std::vector<int64_t>>{std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0}})}});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::BatchToSpaceNDAttrUpdate>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddCNode("updated", {std::make_shared<Primitive>("BatchToSpaceND"), "input"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  auto block_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(c.GetGraph()->output(), "block_shape");
  auto crops = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(c.GetGraph()->output(), "crops");
  EXPECT_TRUE(block_shape.size() == 3 && block_shape[0] == 1);
  EXPECT_TRUE(crops.size() == 3);
}
}  // namespace mindspore
