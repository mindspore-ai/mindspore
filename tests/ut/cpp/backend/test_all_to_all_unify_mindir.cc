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
#include <memory>
#include "common/graph_optimizer_test_framework.h"
#include "ops/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/device/ascend/optimizer/mindir/all_to_all_unify_mindir.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
class AllToAllUnifyMindIR : public UT::Common {
 public:
  AllToAllUnifyMindIR() {}
};

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Split+Concat+AllToAll+Split+Concat for kbk
/// Expectation: After optimize, match Split+Concat+AllToAll+Split+Concat.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {2, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)2)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)3)}});
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("split_dim1")
    .AddVar("num_split1")
    .AddVar("concat_dim1")
    .AddVar("split_dim2")
    .AddVar("num_split2")
    .AddVar("concat_dim2")
    .AddCNode("split1", {std::make_shared<Primitive>("Split"), "input", "split_dim1", "num_split1"})
    .AddCNode("concat1", {std::make_shared<Primitive>("Concat"), "split1", "concat_dim1"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "concat1"})
    .AddCNode("split2", {std::make_shared<Primitive>("Split"), "all_to_all", "split_dim2", "num_split2"})
    .AddCNode("concat2", {std::make_shared<Primitive>("Concat"), "split2", "concat_dim2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
