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

#include <map>
#include <string>
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_expander_cloud.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore::graphkernel::test {
namespace {
struct RmsNormParams {
  ShapeVector x_shape;
  ShapeVector gamma_shape;
  TypePtr x_type;
  TypePtr gamma_type;
  float eps{1e-6};
};

struct RmsNormGradParams {
  ShapeVector dy_shape;
  ShapeVector x_shape;
  ShapeVector rstd_shape;
  ShapeVector gamma_shape;
  TypePtr dy_type;
  TypePtr x_type;
  TypePtr rstd_type;
  TypePtr gamma_type;
};

void CompareShapeAndType(const AnfNodePtr &node, size_t output_idx, const ShapeVector &expect_shape,
                         const TypeId &expect_type) {
  auto cb = graphkernel::Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto output_shape = cb->GetOutputShape(node, output_idx);
  auto output_type = cb->GetOutputType(node, output_idx);
  if (output_shape != expect_shape || output_type != expect_type) {
    MS_LOG(ERROR) << "output[" << output_idx << "] compare failed";
    MS_LOG(ERROR) << "expect shape: " << expect_shape << " data type: " << TypeIdToString(expect_type);
    MS_LOG(ERROR) << "output shape: " << output_shape << " data type: " << TypeIdToString(output_type);
    ASSERT_TRUE(false);
  }
}
}  // namespace

/// Feature: Test graph kernel RmsNorm expander
/// Description: RmsNorm will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestRmsNormExpander : public GraphKernelCommonTestSuite, public testing::WithParamInterface<RmsNormParams> {
  void SetUp() override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--enable_expand_ops=RmsNorm";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestRmsNormExpander, rms_norm) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x = c.NewTensorInput("x", param.x_type, param.x_shape);
  auto gamma = c.NewTensorInput("gamma", param.gamma_type, param.gamma_shape);
  auto eps = c.NewValueNode(MakeValue(param.eps));
  auto kernel_info = std::make_shared<device::KernelInfo>();
  eps->set_kernel_info(kernel_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
  builder->SetOutputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32});
  builder->SetOutputsKernelObjectType(std::vector<kernel::KernelObjectType>{kernel::KernelObjectType::SCALAR});
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), eps.get());

  auto op = c.NewCNodeWithBuildInfo("RmsNorm", {x, gamma, eps}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto rstd_shape = param.x_shape;
      rstd_shape[rstd_shape.size() - 1] = 1;
      CompareShapeAndType(node, 0, param.x_shape, param.x_type->type_id());
      CompareShapeAndType(node, 1, rstd_shape, kNumberTypeFloat32);
      return;
    }
  }
  ASSERT_TRUE(IsDynamic(param.x_shape));
}

/// Feature: Test graph kernel RmsNormGrad expander
/// Description: RmsNormGrad will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestRmsNormGradExpander : public GraphKernelCommonTestSuite,
                                public testing::WithParamInterface<RmsNormGradParams> {
  void SetUp() override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--enable_expand_ops=RmsNormGrad";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestRmsNormGradExpander, rms_norm_grad) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto dy = c.NewTensorInput("dy", param.dy_type, param.dy_shape);
  auto x = c.NewTensorInput("x", param.x_type, param.x_shape);
  auto rstd = c.NewTensorInput("rstd", param.rstd_type, param.rstd_shape);
  auto gamma = c.NewTensorInput("gamma", param.gamma_type, param.gamma_shape);
  auto op = c.NewCNodeWithBuildInfo("RmsNormGrad", {dy, x, rstd, gamma}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.x_shape, param.x_type->type_id());
      CompareShapeAndType(node, 1, param.gamma_shape, kNumberTypeFloat32);
      return;
    }
  }
  ASSERT_TRUE(IsDynamic(param.x_shape));
}

INSTANTIATE_TEST_CASE_P(TestOpRmsNorm, TestRmsNormExpander,
                        testing::Values(RmsNormParams{{16, 128}, {128}, kFloat16, kFloat16},
                                        RmsNormParams{{16, 128}, {128}, kBFloat16, kBFloat16},
                                        RmsNormParams{{16, 128}, {128}, kFloat32, kFloat32},
                                        RmsNormParams{{16, -1}, {-1}, kFloat16, kFloat16},
                                        RmsNormParams{{-1, 128}, {128}, kFloat16, kFloat16}));

INSTANTIATE_TEST_CASE_P(
  TestOpRmsNormGrad, TestRmsNormGradExpander,
  testing::Values(RmsNormGradParams{{8, 64}, {8, 64}, {8, 1}, {64}, kFloat16, kFloat16, kFloat32, kFloat16},
                  RmsNormGradParams{{8, 64}, {8, 64}, {8, 1}, {64}, kBFloat16, kBFloat16, kFloat32, kBFloat16},
                  RmsNormGradParams{{8, 64}, {8, 64}, {8, 1}, {64}, kFloat32, kFloat32, kFloat32, kFloat32},
                  RmsNormGradParams{{-1, -1}, {-1, -1}, {-1, -1}, {-1}, kFloat16, kFloat16, kFloat32, kFloat16},
                  RmsNormGradParams{{-1, 64}, {-1, 64}, {-1, 1}, {64}, kFloat16, kFloat16, kFloat32, kFloat16}));
}  // namespace mindspore::graphkernel::test
