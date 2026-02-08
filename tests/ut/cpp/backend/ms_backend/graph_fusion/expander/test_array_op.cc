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

#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::string op_name;
  TypePtr input_type;
  ShapeVector input_shape;
  TypePtr dst_type{nullptr};
};
}  // namespace

/// Feature: Test ArrayOp expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestArrayOpExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    SetGraphKernelFlags("--enable_expand_ops=Identity");
  }
};

TEST_P(TestArrayOpExpander, array_op) {
  const auto &param = GetParam();
  if (param.op_name != "ZerosLikeExt") {
    ConstructGraph c;
    auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
    auto op = c.NewCNodeWithBuildInfo(param.op_name, {x});
    c.SetOutput(op);
    auto fg = c.GetGraph();
    RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
    size_t gk_size = param.can_expand ? 1 : 0;
    ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
  } else {
    ConstructGraph c;
    auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
    auto dst_type = c.NewValueNode<int64_t>(param.dst_type->type_id());
    auto op = c.NewCNodeWithBuildInfo(param.op_name, {x, dst_type});
    c.SetOutput(op);
    auto fg = c.GetGraph();
    RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
    size_t gk_size = param.can_expand ? 1 : 0;
    ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
  }
}

INSTANTIATE_TEST_CASE_P(
  TestOpArrayOp, TestArrayOpExpander,
  testing::Values(
    // OnesLike
    Params{true, "OnesLike", kFloat16, {16, 16}}, Params{true, "OnesLike", kFloat32, {16, 16}},
    Params{true, "OnesLike", kBFloat16, {16, 16}}, Params{true, "OnesLike", kInt32, {16, 16}},
    Params{false, "OnesLike", kFloat16, {-2}}, Params{false, "OnesLike", kFloat16, {-1, 1, 2}},
    Params{false, "OnesLike", kFloat16, {-1, -1, 2}}, Params{false, "OnesLike", kFloat16, {2, 0, 16}},
    Params{false, "OnesLike", kFloat64, {16, 16}}, Params{false, "OnesLike", kInt8, {16, 16}},
    Params{false, "OnesLike", kInt16, {16, 16}}, Params{false, "OnesLike", kInt64, {16, 16}},
    Params{false, "OnesLike", kBool, {16, 16}},
    // ZerosLike
    Params{true, "ZerosLike", kFloat16, {16, 16}}, Params{true, "ZerosLike", kFloat32, {16, 16}},
    Params{true, "ZerosLike", kBFloat16, {16, 16}}, Params{true, "ZerosLike", kInt32, {16, 16}},
    Params{false, "ZerosLike", kFloat16, {-2}}, Params{false, "ZerosLike", kFloat16, {-1, 1, 2}},
    Params{false, "ZerosLike", kFloat16, {-1, -1, 2}}, Params{false, "ZerosLike", kFloat16, {2, 0, 16}},
    Params{false, "ZerosLike", kFloat64, {16, 16}}, Params{false, "ZerosLike", kInt8, {16, 16}},
    Params{false, "ZerosLike", kInt16, {16, 16}}, Params{false, "ZerosLike", kInt64, {16, 16}},
    Params{false, "ZerosLike", kBool, {16, 16}},
    // Identity
    Params{true, "Identity", kFloat16, {16, 16}}, Params{true, "Identity", kFloat32, {16, 16}},
    Params{true, "Identity", kBFloat16, {16, 16}}, Params{true, "Identity", kInt32, {16, 16}},
    Params{false, "Identity", kFloat16, {-2}}, Params{true, "Identity", kFloat16, {-1, 1, 2}},
    Params{false, "Identity", kFloat16, {-1, -1, 2}}, Params{false, "Identity", kFloat16, {2, 0, 16}},
    // ZerosLikeExt
    Params{true, "ZerosLikeExt", kFloat16, {16, 16}, kFloat16},
    Params{true, "ZerosLikeExt", kFloat32, {16, 16}, kFloat32},
    Params{true, "ZerosLikeExt", kBFloat16, {16, 16}, kBFloat16},
    Params{true, "ZerosLikeExt", kInt32, {16, 16}, kInt32}, Params{true, "ZerosLikeExt", kFloat16, {16, 16}, kFloat32},
    Params{true, "ZerosLikeExt", kFloat16, {16, 16}, kBFloat16},
    Params{true, "ZerosLikeExt", kFloat16, {16, 16}, kInt32}, Params{false, "ZerosLikeExt", kFloat16, {-2}, kFloat16},
    Params{false, "ZerosLikeExt", kFloat16, {-1, 1, 2}, kFloat16},
    Params{false, "ZerosLikeExt", kFloat16, {-1, -1, 2}, kFloat16},
    Params{false, "ZerosLikeExt", kFloat16, {2, 0, 16}, kFloat16},
    Params{false, "ZerosLikeExt", kFloat64, {16, 16}, kFloat64}, Params{false, "ZerosLikeExt", kInt8, {16, 16}, kInt8},
    Params{false, "ZerosLikeExt", kInt16, {16, 16}, kInt16}, Params{false, "ZerosLikeExt", kInt64, {16, 16}, kInt64},
    Params{false, "ZerosLikeExt", kBool, {16, 16}, kBool}));
}  // namespace mindspore::graphkernel::test
