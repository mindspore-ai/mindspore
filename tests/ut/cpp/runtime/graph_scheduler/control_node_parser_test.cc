/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "tests/ut/cpp/common/device_common_test.h"

#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/framework_ops.h"
namespace mindspore {
namespace runtime {
using namespace test;
class ControlNodeParserTest : public UT::Common {
 public:
  ControlNodeParserTest() {}
};

FuncGraphPtr BuildFuncGraph() {
  std::vector<int64_t> shp{2, 2};
  auto func_graph = std::make_shared<FuncGraph>();
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_x = func_graph->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_y = func_graph->add_parameter();
  parameter_y->set_abstract(abstract_y);
  return func_graph;
}

KernelGraphPtr BuildKernelGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &front_node,
                                const ValueNodePtr &prim) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto front_parameter = func_graph->parameters();

  // Build kernel.
  std::vector<AnfNodePtr> inputs{prim};
  for (const auto &parameter : front_parameter) {
    inputs.emplace_back(kernel_graph->NewParameter(parameter->cast<ParameterPtr>()));
  }
  auto backend_node = kernel_graph->NewCNode(inputs);
  std::vector<int64_t> shp{2, 2};
  abstract::AbstractTensorPtr abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  backend_node->set_abstract(abs);
  // build return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), backend_node};
  auto return_node = kernel_graph->NewCNode(return_inputs);

  kernel_graph->set_return(return_node);
  kernel_graph->set_execution_order({backend_node});
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, {front_node});
  return kernel_graph;
}

void BuildGraphs(std::vector<AnfNodePtr> *control_nodes, FuncGraphPtr *func_graph,
                 std::vector<KernelGraphPtr> *kernel_graphs, FuncGraphToKernelGraphGroup *func_graph_to_kernel_graphs) {
  auto root_func_graph = BuildFuncGraph();
  auto true_func_graph = BuildFuncGraph();
  auto false_func_graph = BuildFuncGraph();
  std::vector<int64_t> shp{2, 2};
  abstract::AbstractTensorPtr abs;
  // root graph.
  auto parameters = root_func_graph->parameters();
  // Less.
  std::vector<AnfNodePtr> less_inputs{NewValueNode(prim::kPrimLess), parameters[0], parameters[1]};
  auto less = root_func_graph->NewCNode(less_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  less->set_abstract(abs);
  // True partial.
  std::vector<AnfNodePtr> true_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(true_func_graph),
                                              parameters[0], parameters[1]};
  auto true_partial = root_func_graph->NewCNode(true_partial_inputs);
  control_nodes->emplace_back(true_partial);
  // False partial.
  std::vector<AnfNodePtr> false_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(false_func_graph),
                                               parameters[0], parameters[1]};
  auto false_partial = root_func_graph->NewCNode(false_partial_inputs);
  control_nodes->emplace_back(false_partial);
  // Switch.
  std::vector<AnfNodePtr> switch_inputs{NewValueNode(prim::kPrimSwitch), less, true_partial, false_partial};
  auto switch_node = root_func_graph->NewCNode(switch_inputs);
  auto switch_abs = std::make_shared<FuncGraphAbstractClosure>(false_func_graph, AnalysisContext::DummyContext());
  switch_node->set_abstract(switch_abs);
  control_nodes->emplace_back(switch_node);
  // Call.
  std::vector<AnfNodePtr> call_inputs{switch_node};
  auto root_call_node = root_func_graph->NewCNode(call_inputs);
  auto root_call_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  root_call_node->set_abstract(root_call_abs);
  control_nodes->emplace_back(root_call_node);
  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), root_call_node};
  auto return_node = root_func_graph->NewCNode(return_inputs);
  return_node->set_abstract(root_call_abs);
  control_nodes->emplace_back(return_node);
  root_func_graph->set_return(return_node);

  // true graph.
  auto true_parameters = true_func_graph->parameters();
  // Call.
  std::vector<AnfNodePtr> true_call_inputs{NewValueNode(root_func_graph), true_parameters[0], true_parameters[1]};
  auto true_call_node = true_func_graph->NewCNode(true_call_inputs);
  auto true_call_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  true_call_node->set_abstract(true_call_abs);
  control_nodes->emplace_back(true_call_node);
  // Add.
  std::vector<AnfNodePtr> true_add_inputs{NewValueNode(prim::kPrimAdd), true_parameters[0], true_call_node};
  auto true_add = true_func_graph->NewCNode(true_add_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  true_add->set_abstract(abs);
  // Return.
  std::vector<AnfNodePtr> true_return_inputs{NewValueNode(prim::kPrimReturn), true_add};
  auto true_return_node = true_func_graph->NewCNode(true_return_inputs);
  true_return_node->set_abstract(root_call_abs);
  control_nodes->emplace_back(true_return_node);
  true_func_graph->set_return(true_return_node);

  // false graph.
  // Add.
  auto false_parameters = false_func_graph->parameters();
  std::vector<AnfNodePtr> false_add_inputs{NewValueNode(prim::kPrimAdd), false_parameters[0], false_parameters[1]};
  auto false_add = false_func_graph->NewCNode(false_add_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  false_add->set_abstract(abs);
  // Return.
  std::vector<AnfNodePtr> false_return_inputs{NewValueNode(prim::kPrimReturn), false_add};
  auto false_return_node = false_func_graph->NewCNode(false_return_inputs);
  false_return_node->set_abstract(root_call_abs);
  control_nodes->emplace_back(false_return_node);
  false_func_graph->set_return(false_return_node);

  // Build kernel graph.
  // Root kernel graph.
  auto root_kernel_graph = BuildKernelGraph(root_func_graph, less, NewValueNode(prim::kPrimLess));
  kernel_graphs->emplace_back(root_kernel_graph);
  std::vector<KernelGraphPtr> graphs{root_kernel_graph};
  (*func_graph_to_kernel_graphs)[root_func_graph].emplace_back(graphs);
  // True kernel graph.
  auto true_kernel_graph = BuildKernelGraph(true_func_graph, true_add, NewValueNode(prim::kPrimAdd));
  kernel_graphs->emplace_back(true_kernel_graph);
  graphs[0] = true_kernel_graph;
  (*func_graph_to_kernel_graphs)[true_func_graph].emplace_back(graphs);
  // False kernel graph.
  auto false_kernel_graph = BuildKernelGraph(false_func_graph, false_add, NewValueNode(prim::kPrimAdd));
  kernel_graphs->emplace_back(false_kernel_graph);
  graphs[0] = false_kernel_graph;
  (*func_graph_to_kernel_graphs)[false_func_graph].emplace_back(graphs);

  (*func_graph) = root_func_graph;
}

/// Feature: control flow support dynamic shape.
/// Description: Test the parse interface.
/// Expectation: As expected.
TEST_F(ControlNodeParserTest, Parse) {
  std::vector<AnfNodePtr> control_nodes;
  FuncGraphPtr func_graph;
  std::vector<KernelGraphPtr> kernel_graphs;
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  BuildGraphs(&control_nodes, &func_graph, &kernel_graphs, &func_graph_to_kernel_graphs);

  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);

  auto parser = std::make_shared<ControlNodeParser>();
  DeviceContextKey device_context_key{"CPU", 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  std::vector<DeviceContext *> device_contexts(kernel_graphs.size(), device_context.get());

  parser->Parse(control_nodes, kernel_graphs, device_contexts, func_graph, func_graph_to_kernel_graphs);
  ASSERT_EQ(4, parser->control_node_parameters().size());
}
}  // namespace runtime
}  // namespace mindspore
