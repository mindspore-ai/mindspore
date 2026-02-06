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

#include <map>
#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "abstract/abstract_value.h"
#include "ir/graph_utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/symbol_engine_builder.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "backend/ms_backend/graph_fusion/split_model/split_model_factory.h"

namespace mindspore::graphkernel::test {
namespace {
void Init(const std::string &op_name) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = "--enable_expand_ops=" + op_name;
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);

  SPLIT_MODEL_REGISTER(kAscendDevice, graphkernel::inner::SplitModelAscend);
}
}  // namespace

struct Params {
  std::string op_name;
  ShapeArray input_shape;
  TypePtr input_type;
  size_t target_split_num{1};
};

/// Feature: Test graph kernel op dynamic shape
/// Description: inputs are dynamic shape
/// Expectation: After split pass, the sub graph should not be split into multiple sub graphs
class TestRectifySymbol : public GraphKernelCommonTestSuite, public testing::WithParamInterface<Params> {};

TEST_P(TestRectifySymbol, rectify_symbol) {
  const auto &param = GetParam();
  Init(param.op_name);
  ConstructGraph c;
  std::vector<AnfNodePtr> inputs(param.input_shape.size());
  for (size_t i = 0; i < param.input_shape.size(); ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.input_type, param.input_shape[i]);
  }
  auto op = c.NewCNodeWithBuildInfo(param.op_name, inputs);
  c.SetOutput(op);

  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
                         std::make_shared<graphkernel::SymbolEngineBuilder>(false),
                         std::make_shared<graphkernel::GraphKernelSplitterWithPy>(false)});
  size_t gk_node_num = 0;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      gk_node_num += 1;
    }
  }
  EXPECT_EQ(gk_node_num, param.target_split_num);
}

INSTANTIATE_TEST_CASE_P(TestOpRectifySymbol, TestRectifySymbol,
                        testing::Values(Params{"AddN", {{-1, -1}, {-1, -1}}, kFloat32},
                                        Params{"AddN", {{2, 4}, {-1, 4}}, kInt32},
                                        Params{"AddN", {{2, -1}, {-1, 4}}, kInt32},
                                        Params{"GeLUGrad", {{-1, -1}, {-1, -1}}, kFloat32},
                                        Params{"SiLUGrad", {{-1, -1}, {-1, -1}}, kFloat32},
                                        Params{"SiLUGrad", {{-1, 32}, {-1, 32}}, kFloat32},
                                        Params{"SiLUGrad", {{16, -1}, {16, -1}}, kFloat32},
                                        Params{"SiLUGrad", {{16, 32}, {-1, 32}}, kFloat32},
                                        Params{"SiLUGrad", {{16, 32}, {16, -1}}, kFloat32},
                                        Params{"RmsNormGrad", {{-1, 64}, {-1, 64}, {-1, 1}, {64}}, kFloat32, 3}));
}  // namespace mindspore::graphkernel::test
