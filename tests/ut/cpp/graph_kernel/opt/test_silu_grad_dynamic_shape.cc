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
#include "backend/common/graph_kernel/adapter/symbol_engine_builder.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_splitter_with_py.h"
#include "backend/common/graph_kernel/adapter/split_model_ascend.h"
#include "backend/common/graph_kernel/split_model/split_model_factory.h"

namespace mindspore {
namespace {
void Init() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = "--enable_expand_ops=SiLUGrad";
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);

  SPLIT_MODEL_REGISTER(kAscendDevice, graphkernel::inner::SplitModelAscend);
}
}  // namespace

struct SiLUGradParams {
  ShapeVector x0_shape;
  ShapeVector x1_shape;
  TypePtr x0_type;
  TypePtr x1_type;
};

/// Feature: Test graph kernel SiLUGrad dynamic shape
/// Description: SiLUGrad inputs are dynamic shape and it will expanded
/// Expectation: After expand and split pass, the sub graph of SiLUGrad should not be split into multiple sub graphs
class TestSiLUGrad : public GraphKernelCommonTestSuite, public testing::WithParamInterface<SiLUGradParams> {};

TEST_P(TestSiLUGrad, silu_grad) {
  Init();
  const auto &param = GetParam();
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", param.x0_type, param.x0_shape);
  auto x1 = c.NewTensorInput("x1", param.x1_type, param.x1_shape);
  auto op = c.NewCNode("SiLUGrad", {x0, x1}, {});
  c.SetGeneralBuildInfo(op);
  c.SetOutput(op);

  test::RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                               std::make_shared<graphkernel::SymbolEngineBuilder>(false),
                               std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  size_t gk_node_num = 0;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      gk_node_num += 1;
    }
  }
  EXPECT_EQ(gk_node_num, 1);
}

INSTANTIATE_TEST_CASE_P(TestOpSiLUGrad, TestSiLUGrad,
                        testing::Values(SiLUGradParams{{-1, 32}, {-1, 32}, kFloat32, kFloat32}));
}  // namespace mindspore
