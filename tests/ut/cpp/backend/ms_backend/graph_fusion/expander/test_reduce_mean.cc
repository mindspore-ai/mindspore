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
  bool keep_dims;
  bool axis_is_const{true};
  bool axis_is_tensor{true};
  ShapeVector axis;
};
}  // namespace

/// Feature: Test ReduceMean expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestReduceMeanExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestReduceMeanExpander, reduce_mean) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
  AnfNodePtr axis;
  if (param.axis_is_const) {
    if (param.axis_is_tensor) {
      axis = c.NewValueNode(tensor::from_vector(param.axis));
    } else {
      axis = c.NewValueNode(MakeValue(param.axis));
    }
  } else {
    axis = c.NewTensorInput("axis", kInt64, {1});
  }
  auto keep_dims = c.NewValueNode(MakeValue<bool>(param.keep_dims));
  auto op = c.NewCNodeWithBuildInfo("ReduceMean", {x, axis, keep_dims});
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
               std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpReduceMean, TestReduceMeanExpander,
                        testing::Values(Params{true, kFloat32, {10, 20}, true, true, true, {0, 1}},
                                        Params{true, kFloat32, {10, 20}, true, true, true, {0}},
                                        Params{true, kFloat32, {10, 20}, true, true, true, {1}},
                                        Params{true, kFloat32, {10, 20}, false, true, true, {0, 1}},
                                        Params{true, kFloat32, {10, 20}, false, true, true, {0}},
                                        Params{true, kFloat32, {10, 20}, false, true, true, {1}},
                                        Params{true, kFloat32, {10, 20}, true, true, false, {0, 1}},
                                        Params{true, kFloat32, {10, 20}, true, true, false, {0}},
                                        Params{true, kFloat32, {10, 20}, true, true, false, {1}},
                                        Params{true, kFloat32, {10, 20}, true, true, false, {}},
                                        Params{true, kFloat32, {10, -1}, true, true, false, {0}},
                                        Params{true, kFloat16, {10, 20}, true, true, false, {0, 1}},
                                        Params{true, kBFloat16, {10, 20}, true, true, false, {0, 1}},
                                        Params{false, kFloat32, {-2}, true, true, false, {0}},
                                        Params{false, kFloat32, {10, -1}, true, true, false, {0, 1}},
                                        Params{false, kFloat32, {10, -1}, true, true, false, {1}},
                                        Params{false, kFloat32, {10, -1}, true, true, false, {}}));
}  // namespace mindspore::graphkernel::test
