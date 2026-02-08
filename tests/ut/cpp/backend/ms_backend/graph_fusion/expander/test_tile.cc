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
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  TypePtr input_type;
  ShapeVector input_shape;
  ShapeVector dims;
};
}  // namespace

/// Feature: Test Tile expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestTileExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestTileExpander, tile) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
  auto dims = c.NewValueNode(MakeValue(param.dims));
  auto op = c.NewCNodeWithBuildInfo("Tile", {x, dims});
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
               std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpTile, TestTileExpander,
                        testing::Values(Params{true, kFloat16, {2, 1}, {1, 4}}, Params{true, kFloat32, {2, 1}, {1, 4}},
                                        Params{true, kBFloat16, {2, 1}, {1, 4}}, Params{true, kInt32, {2, 1}, {1, 4}},
                                        Params{true, kFloat16, {2, 1}, {1, 1}}, Params{true, kFloat16, {1, 5}, {4, 1}},
                                        Params{true, kFloat16, {1, 1}, {1, 1}}, Params{true, kFloat16, {1, 1}, {2, 3}},
                                        Params{false, kInt64, {2, 1}, {1, 4}}, Params{false, kInt8, {2, 1}, {1, 4}},
                                        Params{false, kBool, {2, 1}, {1, 4}}, Params{true, kFloat16, {-1, 1}, {1, 4}},
                                        Params{false, kFloat16, {-1, 1}, {2, 4}}));
}  // namespace mindspore::graphkernel::test
