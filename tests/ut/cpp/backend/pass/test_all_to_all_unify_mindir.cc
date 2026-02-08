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
#include "primitive/sequence_ops.h"
#include "common/common_test.h"
#include "backend/common/pass/mindir/all_to_all_unify_mindir.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
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
  auto input = c.NewTensorInput("input", kFloat, {2, -1, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)2)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)3)}});
  std::vector<int64_t> shp{2, -1, 1024, 4096};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
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

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to AllToAll for kbk
/// Expectation: After optimize, match AllToAll.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk0) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {2, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)0)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)0)}});
  std::vector<int64_t> shp{2, 2, 2048, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "input"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Reshape+AllToAll+Reshape+Transpose+Reshape for kbk
/// Expectation: After optimize, match Reshape+AllToAll+Reshape+Transpose+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk1) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)1)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)3)}});
  std::vector<int64_t> shp{1, 1, 2048, 4096};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims")
    .AddVar("reshape_dims1")
    .AddVar("reshape_dims2")
    .AddVar("transpose_dims")
    .AddCNode("reshape", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape"})
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims1"})
    .AddCNode("transpose", {std::make_shared<Primitive>("Transpose"), "reshape1", "transpose_dims"})
    .AddCNode("reshape2", {std::make_shared<Primitive>("Reshape"), "transpose", "reshape_dims2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Reshape+Transpose+Reshape+AllToAll+Reshape for kbk
/// Expectation: After optimize, match Reshape+Transpose+Reshape+AllToAll+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk2) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)2)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)1)}});
  std::vector<int64_t> shp{1, 4, 1024, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims1")
    .AddVar("reshape_dims2")
    .AddVar("transpose_dims")
    .AddVar("reshape_dims")
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims1"})
    .AddCNode("transpose", {std::make_shared<Primitive>("Transpose"), "reshape1", "transpose_dims"})
    .AddCNode("reshape2", {std::make_shared<Primitive>("Reshape"), "transpose", "reshape_dims2"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape2"})
    .AddCNode("reshape", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Reshape+AllToAll+Reshape for kbk
/// Expectation: After optimize, match Reshape+AllToAll+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk3) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)1)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)1)}});
  std::vector<int64_t> shp{1, 2, 2048, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims")
    .AddVar("reshape_dims1")
    .AddCNode("reshape", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape"})
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims1"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to AllToAll+Reshape for kbk
/// Expectation: After optimize, match AllToAll+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk4) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {2, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)0)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)1)}});
  std::vector<int64_t> shp{1, 4, 2048, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims")
    .AddVar("reshape_dims1")
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "input"})
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims1"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Reshape+AllToAll for kbk
/// Expectation: After optimize, match Reshape+AllToAll.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk5) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)1)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)0)}});
  std::vector<int64_t> shp{2, 1, 2048, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims")
    .AddVar("reshape_dims1")
    .AddCNode("reshape", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to eshape+Transpose+Reshape+AllToAll+Reshape+Transpose+Reshape for kbk
/// Expectation: After optimize, match eshape+Transpose+Reshape+AllToAll+Reshape+Transpose+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk6) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)2)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)2)}});
  std::vector<int64_t> shp{1, 2, 2048, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims1")
    .AddVar("reshape_dims2")
    .AddVar("transpose_dims")
    .AddVar("reshape_dims3")
    .AddVar("reshape_dims4")
    .AddVar("transpose_dims1")
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims1"})
    .AddCNode("transpose", {std::make_shared<Primitive>("Transpose"), "reshape1", "transpose_dims"})
    .AddCNode("reshape2", {std::make_shared<Primitive>("Reshape"), "transpose", "reshape_dims2"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape2"})
    .AddCNode("reshape3", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims3"})
    .AddCNode("transpose1", {std::make_shared<Primitive>("Transpose"), "reshape3", "transpose_dims1"})
    .AddCNode("reshape4", {std::make_shared<Primitive>("Reshape"), "transpose1", "reshape_dims4"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to AllToAll+Reshape+Transpose+Reshape for kbk
/// Expectation: After optimize, match AllToAll+Reshape+Transpose+Reshape.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk7) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {2, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)0)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)2)}});
  std::vector<int64_t> shp{1, 2, 4096, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims3")
    .AddVar("reshape_dims4")
    .AddVar("transpose_dims1")
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "input"})
    .AddCNode("reshape3", {std::make_shared<Primitive>("Reshape"), "all_to_all", "reshape_dims3"})
    .AddCNode("transpose1", {std::make_shared<Primitive>("Transpose"), "reshape3", "transpose_dims1"})
    .AddCNode("reshape4", {std::make_shared<Primitive>("Reshape"), "transpose1", "reshape_dims4"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify mindir pass: AllToAllUnifyMindIR
/// Description: Convert AlltoAll to Reshape+Transpose+Reshape+AllToAll for kbk
/// Expectation: After optimize, match Reshape+Transpose+Reshape+AllToAll.
TEST_F(AllToAllUnifyMindIR, test_all_to_all_unify_mindir_kbk8) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat, {1, 2, 2048, 2048});
  std::string group = "hccl_world_group";
  auto node = c.NewCNodeWithoutInfer("AlltoAll", {input},
                                     {{"split_count", MakeValue((int64_t)2)},
                                      {"split_dim", MakeValue((int64_t)2)},
                                      {"group", MakeValue(group)},
                                      {"concat_dim", MakeValue((int64_t)0)}});
  std::vector<int64_t> shp{2, 2, 1024, 2048};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  node->set_abstract(x_abstract);
  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AllToAllUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("reshape_dims1")
    .AddVar("reshape_dims2")
    .AddVar("transpose_dims")
    .AddCNode("reshape1", {std::make_shared<Primitive>("Reshape"), "input", "reshape_dims1"})
    .AddCNode("transpose", {std::make_shared<Primitive>("Transpose"), "reshape1", "transpose_dims"})
    .AddCNode("reshape2", {std::make_shared<Primitive>("Reshape"), "transpose", "reshape_dims2"})
    .AddCNode("all_to_all", {std::make_shared<Primitive>("AllToAll"), "reshape2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
