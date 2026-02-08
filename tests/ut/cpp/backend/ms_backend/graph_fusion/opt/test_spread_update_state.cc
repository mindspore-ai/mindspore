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
#include "backend/ms_backend/graph_fusion/core/update_state_formatter.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore::graphkernel::test {
namespace {
void SetBuildInfo(const AnfNodePtr &node, const std::vector<std::string> &inputs_format,
                  const std::vector<std::string> &outputs_format, const std::vector<TypeId> &inputs_device_type,
                  const std::vector<TypeId> &outputs_device_type,
                  const std::vector<kernel::KernelObjectType> &inputs_kernel_object_type,
                  const std::vector<kernel::KernelObjectType> &outputs_kernel_object_type) {
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(inputs_format);
  info_builder.SetOutputsFormat(outputs_format);
  info_builder.SetInputsDeviceType(inputs_device_type);
  info_builder.SetOutputsDeviceType(outputs_device_type);
  info_builder.SetInputsKernelObjectType(inputs_kernel_object_type);
  info_builder.SetOutputsKernelObjectType(outputs_kernel_object_type);
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), node.get());
}
}  // namespace

class TestPassSpreadUpdateState : public GraphKernelCommonTestSuite {};

/// Feature: Test graph kernel SpreadUpdateState pass
/// Description: SpreadUpdateState pass
/// Expectation: After pass, the inputs of UpdateState will be flatten
TEST_F(TestPassSpreadUpdateState, tensor_inplace) {
  SetDeviceTarget(kAscendDevice);
  ConstructGraph c;
  ShapeVector shape{-1, 2, -1, 4};
  auto x0 = c.NewTensorInput("x0", kFloat32, shape);
  auto x1 = c.NewTensorInput("x1", kFloat32, {4});
  auto x2 = c.NewTensorInput("x2", kFloat32, {4});
  auto y0 = c.NewCNodeWithoutInfer("Shape", {x0});
  auto shape_value = MakeValue(shape);
  y0->set_abstract(shape_value->ToAbstract());
  SetBuildInfo(y0, {"DefaultFormat"}, {"DefaultFormat"}, {kNumberTypeFloat32}, {kNumberTypeInt64},
               {kernel::KernelObjectType::TENSOR}, {kernel::KernelObjectType::TUPLE});
  auto y1 = c.NewCNode("TupleGetItem", {y0, c.NewValueNode(MakeValue<int64_t>(2))});
  SetBuildInfo(y1, {"DefaultFormat", "DefaultFormat"}, {"DefaultFormat"}, {kNumberTypeInt64, kNumberTypeInt64},
               {kNumberTypeInt64}, {kernel::KernelObjectType::TUPLE_UNFOLD, kernel::KernelObjectType::SCALAR},
               {kernel::KernelObjectType::SCALAR});
  auto y2 = c.NewCNodeWithBuildInfo("Abs", {x1});
  auto y3 = c.NewCNodeWithBuildInfo("Sqrt", {x2});
  auto y4 = c.NewCNodeWithBuildInfo("MakeTuple", {y2, y3});
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto y5 = c.NewCNodeWithBuildInfo("UpdateState", {u, y4, y0});
  auto y6 = c.NewCNodeWithBuildInfo("MakeTuple", {y1, y5});
  c.SetOutput(y6);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::SpreadUpdateState>()});
  int64_t idx = -1;
  for (const auto &node : GetAllNodes(fg)) {
    if (node != nullptr && IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t i = 1; i < cnode->size(); i++) {
        if (cnode->input(i) == y1) {
          idx = SizeToLong(i);
          break;
        }
      }
    }
  }
  EXPECT_NE(idx, -1);
}
}  // namespace mindspore::graphkernel::test
