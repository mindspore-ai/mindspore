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
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  TypePtr input_type;
  TypePtr value_type;
  TypePtr dst_type;
  bool value_is_const;
};
}  // namespace

/// Feature: Test Muls expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestMulsExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestMulsExpander, muls) {
  const auto &param = GetParam();
  ConstructGraph c;
  ShapeVector shape{10, 20};
  auto x = c.NewTensorInput("x", param.input_type, shape);
  AnfNodePtr value;
  if (param.value_is_const) {
    value = c.NewValueNode(NewScalar(param.value_type, 2, true));
  } else {
    value = c.NewTensorInput("value", param.value_type, {});
  }
  auto op = c.NewCNodeWithBuildInfo("Muls", {x, value});
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, shape, param.dst_type->type_id());
    }
  }
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpMuls, TestMulsExpander,
  testing::Values(Params{true, kFloat16, kInt64, kFloat16, true}, Params{true, kFloat32, kInt64, kFloat32, true},
                  Params{true, kBFloat16, kInt64, kBFloat16, true}, Params{false, kFloat16, kFloat32, kFloat16, true},
                  Params{true, kFloat32, kFloat32, kFloat32, true}, Params{false, kBFloat16, kFloat32, kBFloat16, true},
                  Params{false, kFloat16, kFloat64, kFloat16, true}, Params{true, kFloat32, kFloat64, kFloat32, true},
                  Params{false, kBFloat16, kFloat64, kBFloat16, true}, Params{false, kInt32, kInt64, kInt32, true},
                  Params{false, kInt64, kInt64, kInt64, true}, Params{false, kInt8, kInt64, kInt8, true},
                  Params{false, kFloat16, kInt64, kFloat16, false}, Params{false, kFloat32, kInt64, kFloat32, false},
                  Params{false, kBFloat16, kInt64, kBFloat16, false}));
}  // namespace mindspore::graphkernel::test
