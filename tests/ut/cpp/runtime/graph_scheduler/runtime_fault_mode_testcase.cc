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

#include "tests/ut/cpp/common/device_common_test.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"

#include "primitive/comparison_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/math_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace runtime {
using namespace test;
class RuntimeFaultModeTest : public UT::Common {
 public:
  RuntimeFaultModeTest() {}
};

namespace {
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

KernelGraphPtr BuildFaultKernelGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &front_node,
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

  auto make_tuple = kernel_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), backend_node, inputs[1]});

  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), make_tuple};
  auto return_node = kernel_graph->NewCNode(return_inputs);

  kernel_graph->set_return(return_node);
  kernel_graph->set_execution_order({backend_node});
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({make_tuple}, {front_node});
  return kernel_graph;
}

void BuildInvalidIOSizeGraphs(std::vector<AnfNodePtr> *control_nodes, FuncGraphPtr *func_graph,
                              std::vector<KernelGraphPtr> *kernel_graphs,
                              FuncGraphToKernelGraphGroup *func_graph_to_kernel_graphs) {
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
  auto true_kernel_graph = BuildFaultKernelGraph(true_func_graph, true_add, NewValueNode(prim::kPrimAdd));
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
}  // namespace

/// Feature: runtime fault mode testcase.
/// Description: test input and output nodes.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, InvalidIOSize) {
  std::vector<AnfNodePtr> control_nodes;
  FuncGraphPtr func_graph;
  std::vector<KernelGraphPtr> kernel_graphs;
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  BuildInvalidIOSizeGraphs(&control_nodes, &func_graph, &kernel_graphs, &func_graph_to_kernel_graphs);

  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);

  auto parser = std::make_shared<ControlNodeParser>();
  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  std::vector<DeviceContext *> device_contexts(kernel_graphs.size(), device_context.get());
  try {
    parser->Parse(control_nodes, kernel_graphs, device_contexts, func_graph, func_graph_to_kernel_graphs);
  } catch (std::runtime_error const &e) {
    ASSERT_TRUE(std::string(e.what()).find("Failed to find input") != std::string::npos);
  }
}

namespace {
void BuildInvalidFuncGraph1(std::vector<AnfNodePtr> *control_nodes, FuncGraphPtr *func_graph,
                            std::vector<KernelGraphPtr> *kernel_graphs,
                            FuncGraphToKernelGraphGroup *func_graph_to_kernel_graphs) {
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
  // Invalid partial.
  auto invalid_func_graph = BuildFuncGraph();
  std::vector<AnfNodePtr> invalid_return_inputs{NewValueNode(prim::kPrimReturn), invalid_func_graph->parameters()[0]};
  auto invalid_return_node = invalid_func_graph->NewCNode(invalid_return_inputs);
  invalid_func_graph->set_return(invalid_return_node);
  std::vector<AnfNodePtr> invalid_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(invalid_func_graph),
                                                 false_func_graph->parameters()[0], false_func_graph->parameters()[1],
                                                 false_func_graph->parameters()[1]};
  auto invalid_partial = invalid_func_graph->NewCNode(invalid_partial_inputs);
  control_nodes->emplace_back(invalid_partial);
  // Depend.
  auto depend = false_func_graph->NewCNode({NewValueNode(prim::kPrimDepend), false_add, invalid_partial});
  // Return.
  std::vector<AnfNodePtr> false_return_inputs{NewValueNode(prim::kPrimReturn), depend};
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
}  // namespace

/// Feature: runtime fault mode testcase.
/// Description: test valid funcgraph check.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, InvalidFuncGraph1) {
  std::vector<AnfNodePtr> control_nodes;
  FuncGraphPtr func_graph;
  std::vector<KernelGraphPtr> kernel_graphs;
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  BuildInvalidFuncGraph1(&control_nodes, &func_graph, &kernel_graphs, &func_graph_to_kernel_graphs);

  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);

  auto parser = std::make_shared<ControlNodeParser>();
  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  std::vector<DeviceContext *> device_contexts(kernel_graphs.size(), device_context.get());
  try {
    parser->Parse(control_nodes, kernel_graphs, device_contexts, func_graph, func_graph_to_kernel_graphs);
  } catch (std::runtime_error const &e) {
    ASSERT_TRUE(std::string(e.what()).find("Invalid partial input size") != std::string::npos);
  }
}

namespace {
void BuildInvalidFuncGraph2(std::vector<AnfNodePtr> *control_nodes, FuncGraphPtr *func_graph,
                            std::vector<KernelGraphPtr> *kernel_graphs,
                            FuncGraphToKernelGraphGroup *func_graph_to_kernel_graphs) {
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
  auto switch_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
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
}  // namespace

/// Feature: runtime fault mode testcase.
/// Description: test valid funcgraph check.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, InvalidFuncGraph2) {
  std::vector<AnfNodePtr> control_nodes;
  FuncGraphPtr func_graph;
  std::vector<KernelGraphPtr> kernel_graphs;
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  BuildInvalidFuncGraph2(&control_nodes, &func_graph, &kernel_graphs, &func_graph_to_kernel_graphs);

  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);

  auto parser = std::make_shared<ControlNodeParser>();
  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  std::vector<DeviceContext *> device_contexts(kernel_graphs.size(), device_context.get());
  try {
    parser->Parse(control_nodes, kernel_graphs, device_contexts, func_graph, func_graph_to_kernel_graphs);
  } catch (std::runtime_error const &e) {
    ASSERT_TRUE(std::string(e.what()).find("Get func graphs from abstract failed") != std::string::npos);
  }
}

/// Feature: runtime fault mode testcase.
/// Description: test actor valid check.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, InvalidActorSet) {
  std::string actor_name = "Test";
  AID memory_manager_aid;
  auto actor_set = std::make_shared<ActorSet>(actor_name);
  MS_EXCEPTION_IF_NULL(actor_set);

  // Build data source actor.
  auto host_queue = std::make_shared<HostTensorQueue>();
  auto host_queue_ds_actor =
    std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid, nullptr, nullptr, host_queue, "");
  actor_set->data_source_actors_.emplace_back(host_queue_ds_actor);

  // Build data prepare actor.
  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<TensorPtr> *> input_tensors;
  std::vector<AnfNodePtr> control_nodes;
  std::vector<AnfNodePtr> origin_parameters_order;
  ControlNodeParserPtr parser = std::make_shared<ControlNodeParser>();
  KernelMapPosition origin_outputs_order;
  auto graph_compiler_info = std::make_shared<GraphCompilerInfo>(
    graphs, device_contexts, tensors_mask, input_tensors, control_nodes, origin_parameters_order, parser,
    origin_outputs_order, 2, 2, actor_name, false, GraphExecutionStrategy::kPipeline, nullptr, "");
  auto data_prepare_actor = std::make_shared<DataPrepareActor>(
    actor_name, memory_manager_aid, nullptr, nullptr, graph_compiler_info.get(), host_queue_ds_actor, host_queue);
  actor_set->data_prepare_actor_ = data_prepare_actor;

  // Build super kernel actor.
  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  auto graph = std::make_shared<KernelGraph>();
  auto super_kernel_actor = std::make_shared<SuperKernelActor>(actor_name, graph, "", device_context.get(),
                                                               memory_manager_aid, nullptr, nullptr);
  actor_set->super_kernel_actors_.emplace_back(super_kernel_actor);

  // Build output actor.
  std::vector<KernelWithIndex> summary_nodes;
  auto output_actor = std::make_shared<OutputActor>(actor_name, 1, 2, summary_nodes);
  actor_set->output_actor_ = output_actor;

  auto data_arrow0 = std::make_shared<DataArrow>(0, host_queue_ds_actor->GetAID(), 0);
  (void)host_queue_ds_actor->output_data_arrows_.emplace_back(data_arrow0);
  actor_set->data_prepare_actor_->output_data_arrows_.emplace_back(data_arrow0);
  host_queue_ds_actor->input_datas_num_ = 1;

  auto data_arrow1 = std::make_shared<DataArrow>(0, super_kernel_actor->GetAID(), 0);
  auto data_arrow2 = std::make_shared<DataArrow>(1, super_kernel_actor->GetAID(), 1);
  (void)host_queue_ds_actor->output_data_arrows_.emplace_back(data_arrow1);
  (void)host_queue_ds_actor->output_data_arrows_.emplace_back(data_arrow2);
  super_kernel_actor->input_datas_num_ = 2;

  auto data_arrow3 = std::make_shared<DataArrow>(0, output_actor->GetAID(), 0);
  (void)super_kernel_actor->output_data_arrows_.emplace_back(data_arrow3);
  output_actor->input_datas_num_ = 1;

  try {
    SchedulerHelper::CheckActorValid(actor_set.get());
  } catch (std::runtime_error const &e) {
    ASSERT_TRUE(std::string(e.what()).find("The input num of Test is wrong") != std::string::npos);
  }
}

/// Feature: runtime fault mode testcase.
/// Description: test output of memory.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, OutofMemory) {
  AID aid;
  auto &memory_manager_actor = MemoryManagerActor::GetInstance();
  auto device_tensor = std::make_shared<TestDeviceAddress>(nullptr, 2048, device::DeviceType::kCPU);
  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  auto kernel_tensor = std::make_shared<KernelTensor>(device_tensor);
  std::vector<KernelTensorPtr> alloc_list{kernel_tensor};
  OpContext<KernelTensor> op_context;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = RandInt::Instance().Get();
  op_context.results_ = &result;
  memory_manager_actor->AllocateMemory(&alloc_list, device_context.get(), &op_context, aid);
  ASSERT_TRUE(op_context.error_info_.find("Memory not enough") != std::string::npos);
}

/// Feature: runtime fault mode testcase.
/// Description: test output of memory by run graph memory leak.
/// Expectation: As expected.
TEST_F(RuntimeFaultModeTest, OutofMemoryByMemoryLeak) {
  // Build kernel graph.
  std::vector<int64_t> shp{32, 32};
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_x = kernel_graph->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_y = kernel_graph->add_parameter();
  parameter_y->set_abstract(abstract_y);

  // Build kernel.
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), parameter_x, parameter_y};
  auto add_node = kernel_graph->NewCNode(inputs);
  abstract::AbstractTensorPtr add_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  add_node->set_abstract(add_abs);
  add_node->set_kernel_info(std::make_shared<KernelInfo>());
  // build return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), add_node};
  auto return_node = kernel_graph->NewCNode(return_inputs);

  kernel_graph->set_return(return_node);
  kernel_graph->set_execution_order({add_node});
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({add_node}, {add_node});

  DeviceContextKey device_context_key{device::DeviceType::kCPU, 0};
  const char device_name[] = "CPU";
  MS_REGISTER_DEVICE(device_name, TestDeviceContext);
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_tensor = std::make_shared<device::DeviceAddress>(nullptr, 1024, device::DeviceType::kCPU, 0);
  auto kernel_tensor = std::make_shared<KernelTensor>(device_tensor);
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, 0, add_node.get());

  std::vector<tensor::TensorPtr> tensors;
  std::map<string, string> compile_options;
  auto graph_executor = std::make_shared<TestGraphExecutor>();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_executor->RunGraph(kernel_graph, tensors, &tensors, compile_options);

  AID aid;
  auto &memory_manager_actor = MemoryManagerActor::GetInstance();
  std::vector<KernelTensorPtr> alloc_list{kernel_tensor};
  OpContext<KernelTensor> op_context;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = RandInt::Instance().Get();
  op_context.results_ = &result;
  memory_manager_actor->AllocateMemory(&alloc_list, device_context.get(), &op_context, aid);

  ASSERT_TRUE(op_context.error_info_.find("Memory not enough") != std::string::npos);
}
}  // namespace runtime
}  // namespace mindspore
