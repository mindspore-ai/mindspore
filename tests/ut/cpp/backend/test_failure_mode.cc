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
#include "primitive/sequence_ops.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/ge_backend/graph_ir/utils.h"
#define private public
#define protected public
#include "backend/common/pass/mindir/dropout_unify_mindir.h"
#include "backend/common/pass/other/lamb_fission.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_map.h"
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

/// Feature: Failure mode which test convert ge adapter
/// Description: Test convert ge adapter with no-exist op
/// Expectation: Got null operator
TEST_F(TestFailureMode, test_convert_no_exist_op) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat, {2, 3});
  auto node = c.NewCNodeWithoutInfer("NoExist", {x1, x2}, {});
  EXPECT_TRUE(device::ascend::FindAdapter(node, false) == nullptr);
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
