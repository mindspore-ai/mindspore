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
#include "common/common_test.h"
#include "common/graph_optimizer_test_framework.h"
#include "ops/sequence_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "include/transform/graph_ir/utils.h"
#define private public
#define protected public
#include "plugin/device/ascend/optimizer/mindir/dropout_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/lamb_fission.h"
#undef private
#undef protected

namespace mindspore {
class TestFailureMode : public UT::Common {
 public:
  TestFailureMode() {}
};

/// Feature: Failure mode which test wrong graph input for backend pass
/// Description: Test LambFissionGe with wrong graph(lamb has wrong number of input)
/// Expectation: After pass, throw exception with wrong number of lamb input
TEST_F(TestFailureMode, test_lamb_fission_ge_with_wrong_input_number) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat, {2, 3});
  auto x2 = c.NewTensorInput("x2", kInt32, {2});
  auto node = c.NewCNodeWithoutInfer("Lamb", {x1, x2}, {});
  c.SetOutput(node);
  try {
    test::RunPass(c.GetGraph(), {std::make_shared<opt::LambFissionGe>()});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The input tensor size[2]") != std::string::npos);
    ASSERT_TRUE(std::string(err.what()).find("is not equal to 10") != std::string::npos);
  }
}

/// Feature: Failure mode which test backend pass problem
/// Description: Test DropoutUnifyMindIR0
/// Expectation: After pass, got wrong pattern, expect DropoutGenMask and DropoutDoMask, but got nochanged
TEST_F(TestFailureMode, test_dropout_unify_mindir_0) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat32, {2, 16384});
  auto index = c.NewValueNode(MakeValue((int64_t)0));
  auto dropout = c.NewCNode("Dropout", {input}, {{"keep_prob", MakeValue(true)}});
  auto getitem = c.NewCNodeWithoutInfer("TupleGetItem", {dropout, index}, {});
  c.SetOutput(getitem);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::DropoutUnifyMindIR0>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("shape")
    .AddVar("prob")
    .AddVar("input")
    .AddCNode("genmask", {std::make_shared<Primitive>("DropoutGenMask"), "shape", "prob"})
    .AddCNode("domask", {std::make_shared<Primitive>("DropoutDoMask"), "input", "genmask", "prob"});
  EXPECT_FALSE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: Failure mode which test convert ge adapter
/// Description: Test convert ge adapter with no-exist op
/// Expectation: Got null operator
TEST_F(TestFailureMode, test_convert_no_exist_op) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat, {2, 3});
  auto node = c.NewCNodeWithoutInfer("NoExist", {x1, x2}, {});
  EXPECT_TRUE(transform::FindAdapter(node, false) == nullptr);
}

/// Feature: Failure mode which test nullptr
/// Description: Test KernelInfo with null kernel info
/// Expectation: Got nullptr exception
TEST_F(TestFailureMode, test_kernel_info_nullptr) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat32, {2, 3});
  auto node = c.NewCNode("Add", {x1, x2}, {});
  try {
    auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The pointer [kernel_info] is null.") != std::string::npos);
  }
}
}  // namespace mindspore
