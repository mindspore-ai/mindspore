/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::vector<TypePtr> inputs_type;
  ShapeArray inputs_shape;
  bool alpha_is_const;
};
}  // namespace

/// Feature: Test graph kernel AddExt/SubExt expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestBinaryExtExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestBinaryExtExpander, binary_ext_op) {
  const auto &param = GetParam();
  std::vector<std::string> op_names{"AddExt", "SubExt"};
  for (const auto &op_name : op_names) {
    ConstructGraph c;
    auto x0 = c.NewTensorInput("x0", param.inputs_type[0], param.inputs_shape[0]);
    auto x1 = c.NewTensorInput("x1", param.inputs_type[1], param.inputs_shape[1]);
    AnfNodePtr alpha;
    if (param.alpha_is_const) {
      alpha = c.NewValueNode(NewScalar(param.inputs_type[2], 2, true));
    } else {
      alpha = c.NewTensorInput("alpha", param.inputs_type[2], {});
    }
    auto op = c.NewCNodeWithBuildInfo(op_name, {x0, x1, alpha}, {});
    c.SetOutput(op);
    auto fg = c.GetGraph();
    RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
    size_t gk_size = param.can_expand ? 1 : 0;
    ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
  }
}

INSTANTIATE_TEST_CASE_P(TestOpBinaryExt, TestBinaryExtExpander,
                        testing::Values(Params{true, {kFloat16, kFloat16, kFloat32}, {{2, 4}, {1, 4}}, true},
                                        Params{true, {kFloat32, kFloat32, kFloat32}, {{2, 4}, {2, 1}}, true},
                                        Params{true, {kBFloat16, kBFloat16, kFloat32}, {{2, 4}, {2, 4}}, true},
                                        Params{true, {kFloat16, kFloat16, kFloat64}, {{2, 4}, {1, 4}}, true},
                                        Params{true, {kFloat32, kFloat32, kFloat64}, {{2, 4}, {2, 1}}, true},
                                        Params{true, {kBFloat16, kBFloat16, kFloat64}, {{2, 4}, {2, 4}}, true},
                                        Params{true, {kFloat16, kFloat16, kInt64}, {{2, 4}, {1, 4}}, true},
                                        Params{true, {kFloat32, kFloat32, kInt64}, {{2, 4}, {2, 1}}, true},
                                        Params{true, {kBFloat16, kBFloat16, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{true, {kInt32, kInt32, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kBFloat16, kBFloat16, kInt64}, {{2, 4}, {2, 4}}, false},
                                        Params{false, {kBool, kBool, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kInt8, kInt8, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kInt16, kInt16, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kInt64, kInt64, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kUInt8, kUInt8, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kUInt16, kUInt16, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kUInt32, kUInt32, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kUInt64, kUInt64, kInt64}, {{2, 4}, {2, 4}}, true},
                                        Params{false, {kFloat64, kFloat64, kInt64}, {{2, 4}, {2, 4}}, true}));
}  // namespace mindspore::graphkernel::test