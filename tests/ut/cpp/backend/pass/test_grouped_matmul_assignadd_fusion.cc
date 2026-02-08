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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/grouped_matmul_assignadd_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
class GroupedMatmulAssignaddFusion : public UT::Common {
 public:
  GroupedMatmulAssignaddFusion() {}
};

/// Feature: A backend ir fusion pass: GroupedMatmulAssignaddFusion
/// Description: Convert TransposeExtView+MakeTuple+GroupedMatmul+TupleGetItem+AssignAdd to InplaceGroupedMatmulAdd
/// for kbk
/// Expectation: After optimize, match InplaceGroupedMatmulAdd.
TEST_F(GroupedMatmulAssignaddFusion, test_grouped_matmul_assignadd_fusion) {
  test::ConstructGraph c;
  auto input_x = c.NewTensorInput("input_x", kBFloat16, {2, 2048});
  auto transpose_dim0 = c.NewValueNode(MakeValue<int64_t>(-1));
  auto transpose_dim1 = c.NewValueNode(MakeValue<int64_t>(-2));
  auto transpose_ext = c.NewCNode("TransposeExtView", {input_x, transpose_dim0, transpose_dim1}, {});
  auto maketuple1 = c.NewCNode("MakeTuple", {transpose_ext}, {});

  auto weight = c.NewTensorInput("weight", kBFloat16, {2, 2048});
  auto maketuple2 = c.NewCNode("MakeTuple", {weight}, {});
  auto none1 = c.NewValueNode(std::make_shared<mindspore::None>());
  auto none2 = c.NewValueNode(std::make_shared<mindspore::None>());
  auto none3 = c.NewValueNode(std::make_shared<mindspore::None>());
  auto none4 = c.NewValueNode(std::make_shared<mindspore::None>());
  auto none5 = c.NewValueNode(std::make_shared<mindspore::None>());
  auto group_list = c.NewTensorInput("group_list", kInt64, {2});
  auto split_item = c.NewValueNode(MakeValue<int64_t>(3));
  auto group_type = c.NewValueNode(MakeValue<int64_t>(2));
  auto transpose_a = c.NewValueNode(MakeValue<bool>(false));
  auto transpose_b = c.NewValueNode(MakeValue<bool>(false));
  auto gmm = c.NewCNode("GroupedMatmul",
                        {maketuple1, maketuple2, none1, none2, none3, none4, none5, group_list, split_item, group_type,
                         transpose_a, transpose_b},
                        {});

  auto getitem_idx = c.NewScalarInput("getitem_idx", MakeValue<int64_t>(0), kInt64);
  auto getitem = c.NewCNode("TupleGetItem", {gmm, getitem_idx}, {});
  auto out = c.NewTensorInput("out", kFloat32, {2, 2048, 2048});
  auto assign_add = c.NewCNode("AssignAdd", {out, getitem});

  c.SetOutput(assign_add);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::GroupedMatmulAssignaddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_x")
    .AddVar("weight")
    .AddVar("group_list")
    .AddVar("out")
    .AddCNode("InplaceGroupedMatmulAdd",
              {std::make_shared<Primitive>("InplaceGroupedMatmulAdd"), "input_x", "weight", "group_list", "out"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
