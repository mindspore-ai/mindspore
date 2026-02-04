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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::vector<TypePtr> inputs_type;
  ShapeArray inputs_shape;
};
}  // namespace

/// Feature: Test ClipByNormNoDivSum expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestClipByNormNoDivSumExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestClipByNormNoDivSumExpander, clip_by_norm_no_div_sum) {
  const auto &param = GetParam();
  ConstructGraph c;
  ShapeVector output_shape;
  std::vector<AnfNodePtr> inputs(param.inputs_type.size());
  for (size_t i = 0; i < param.inputs_type.size(); ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.inputs_type[i], param.inputs_shape[i]);
    if (i == 0) {
      output_shape = param.inputs_shape[i];
    } else {
      if (param.inputs_shape[i].size() > output_shape.size()) {
        auto n = param.inputs_shape[i].size() - output_shape.size();
        output_shape.insert(output_shape.begin(), n, 1);
      }
      auto n = output_shape.size() - param.inputs_shape[i].size();
      for (size_t j = 0; j < param.inputs_shape[i].size(); ++j) {
        if (param.inputs_shape[i][j] != 1) {
          output_shape[j + n] = param.inputs_shape[i][j];
        }
      }
    }
  }
  auto op = c.NewCNodeWithoutInfer("ClipByNormNoDivSum", inputs);
  op->set_abstract(std::make_shared<abstract::AbstractTensor>(param.inputs_type[0], output_shape));
  c.SetGeneralBuildInfo(op);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpClipByNormNoDivSum, TestClipByNormNoDivSumExpander,
  testing::Values(Params{true, {kFloat16, kFloat16, kFloat16, kFloat16}, {{2, 4}, {2, 4}, {2, 4}, {2, 4}}},
                  Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, {{2, 1}, {4}, {1, 4}, {1, 1}}},
                  Params{true, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, {{1}, {4}, {2, 4}, {1, 4}}},
                  Params{false, {kFloat64, kFloat64, kFloat64, kFloat64}, {{2, 4}, {2, 4}, {2, 4}, {2, 4}}},
                  Params{false, {kInt64, kInt64, kInt64, kInt64}, {{2, 4}, {2, 4}, {2, 4}, {2, 4}}},
                  Params{false, {kInt32, kInt32, kInt32, kInt32}, {{2, 4}, {2, 4}, {2, 4}, {2, 4}}}));
}  // namespace mindspore::graphkernel::test
