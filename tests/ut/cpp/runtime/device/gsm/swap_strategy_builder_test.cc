/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <vector>
#include <map>
#include "common/common_test.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "runtime/device/gsm/swap_strategy_builder.h"

namespace mindspore::device {
class TestSwapStrategyBuilder : public BackendCommon {
 public:
  TestSwapStrategyBuilder() : get_py_func_("gtest_input.runtime.device.gsm.mem_usage_analyzer_test", true) {}

  void SetUp() override {
    auto net = get_py_func_("add_net");
    EXPECT_NE(net, nullptr);
    std::vector<int64_t> shp_x{1, 2, 2, 2};
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
    AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
    auto func_graph = GetFuncGraph(net, args_spec_list);
    kernel_graph_add_net_ = Compile(func_graph);

    net = get_py_func_("add_with_all_reduce_net");
    EXPECT_NE(net, nullptr);
    func_graph = GetFuncGraph(net, args_spec_list);
    kernel_graph_add_with_all_reduce_net_ = Compile(func_graph);
  }

  UT::PyFuncGraphFetcher get_py_func_;
  std::shared_ptr<session::KernelGraph> kernel_graph_add_net_;
  std::shared_ptr<session::KernelGraph> kernel_graph_add_with_all_reduce_net_;
};

/// Feature: SwapStrategyBuilder
/// Description: Test SwapStrategyBuilder with variable mem size
/// Expectation: Pass all test cases
TEST_F(TestSwapStrategyBuilder, test_swap_strategy_with_variable_mem_size) {
  auto builder = std::make_shared<SwapStrategyBuilder>();
  auto context = std::make_shared<SwapContext>();
  auto kernel_graph = kernel_graph_add_net_;
  EXPECT_NE(kernel_graph, nullptr);
  std::vector<std::vector<size_t>> inputs = {{10000, 10000}, {10000, 136}, {100, 136}};
  std::vector<std::vector<size_t>> expects = {{5, 2, 5, 0, 5, 0}, {5, 2, 5, 5, 15, 10}, {5, 2, 5, 5, 15, 10}};
  for (size_t i = 0; i < 3; ++i) {
    context->ddr_mem_size_ = inputs[i][0];
    context->hbm_mem_size_ = inputs[i][1];
    auto strategy = builder->Build(kernel_graph, context);
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->kernel_num_, expects[i][0]);
    EXPECT_EQ(strategy->virtual_node_num_, expects[i][1]);
    EXPECT_EQ(strategy->nodes_.size(), expects[i][2]);
    EXPECT_EQ(strategy->actions_.size(), expects[i][3]);
    EXPECT_EQ(strategy->links_.size(), expects[i][4]);
    std::vector<std::shared_ptr<TensorAction>> all_actions;
    for (auto const &item : strategy->actions_) {
      for (auto const &action : item.second->actions_) {
        (void)all_actions.emplace_back(action);
      }
    }
    EXPECT_EQ(all_actions.size(), expects[i][5]);
  }
}

/// Feature: SwapStrategyBuilder
/// Description: Test SwapStrategyBuilder with offload param
/// Expectation: Pass all test cases
TEST_F(TestSwapStrategyBuilder, test_swap_strategy_with_offload_param) {
  auto builder = std::make_shared<SwapStrategyBuilder>();
  auto context = std::make_shared<SwapContext>();
  auto kernel_graph = kernel_graph_add_net_;
  EXPECT_NE(kernel_graph, nullptr);

  context->ddr_mem_size_ = 10000;
  context->hbm_mem_size_ = 10000;
  for (const auto &kernel : kernel_graph->execution_order()) {
    for (const auto &input : kernel->inputs()) {
      if (!input->isa<Parameter>()) {
        continue;
      }
      const auto &parameter = std::dynamic_pointer_cast<Parameter>(input);
      EXPECT_NE(parameter, nullptr);
      parameter->set_default_param(MakeValue(1));
    }
  }
  std::vector<std::vector<size_t>> inputs = {{true, false}, {false, true}, {true, true}};
  std::vector<std::vector<size_t>> expects = {{5, 2, 5, 5, 15, 10}, {5, 2, 5, 5, 15, 10}, {5, 2, 5, 5, 15, 10}};
  for (size_t i = 0; i < 3; ++i) {
    context->offload_param_to_ddr_ = inputs[i][0];
    context->offload_param_to_disk_ = inputs[i][1];
    auto strategy = builder->Build(kernel_graph, context);
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->kernel_num_, expects[i][0]);
    EXPECT_EQ(strategy->virtual_node_num_, expects[i][1]);
    EXPECT_EQ(strategy->nodes_.size(), expects[i][2]);
    EXPECT_EQ(strategy->actions_.size(), expects[i][3]);
    EXPECT_EQ(strategy->links_.size(), expects[i][4]);
    std::vector<std::shared_ptr<TensorAction>> all_actions;
    for (auto const &item : strategy->actions_) {
      for (auto const &action : item.second->actions_) {
        (void)all_actions.emplace_back(action);
      }
    }
    EXPECT_EQ(all_actions.size(), expects[i][5]);
  }
}

/// Feature: SwapStrategyBuilder
/// Description: Test SwapStrategyBuilder with all reduce nodes
/// Expectation: Pass all test cases
TEST_F(TestSwapStrategyBuilder, test_swap_strategy_with_all_reduce_nodes) {
  auto builder = std::make_shared<SwapStrategyBuilder>();
  auto context = std::make_shared<SwapContext>();
  auto kernel_graph = kernel_graph_add_with_all_reduce_net_;
  EXPECT_NE(kernel_graph, nullptr);

  context->ddr_mem_size_ = 100;
  context->hbm_mem_size_ = 250;
  auto strategy = builder->Build(kernel_graph, context);
  EXPECT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->kernel_num_, 9);
  EXPECT_EQ(strategy->virtual_node_num_, 2);
  EXPECT_EQ(strategy->nodes_.size(), 9);
  EXPECT_EQ(strategy->actions_.size(), 8);
  EXPECT_EQ(strategy->links_.size(), 25);
  std::vector<std::shared_ptr<TensorAction>> all_actions;
  for (auto const &item : strategy->actions_) {
    for (auto const &action : item.second->actions_) {
      (void)all_actions.emplace_back(action);
    }
  }
  EXPECT_EQ(all_actions.size(), 12);
}
}  // namespace mindspore::device