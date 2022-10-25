/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "graph_scheduler_common_test.h"

namespace mindspore {
namespace runtime {
using namespace test;
class GraphSchedulerTest : public UT::Common {
 public:
  GraphSchedulerTest() {}
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

FuncGraphPtr BuildGraphs() {
  auto root_func_graph = BuildFuncGraph();
  auto true_func_graph = BuildFuncGraph();
  auto false_func_graph = BuildFuncGraph();
  std::vector<int64_t> shp{2, 2};

  // root graph.
  auto parameters = root_func_graph->parameters();
  // Less.
  std::vector<AnfNodePtr> less_inputs{NewValueNode(prim::kPrimLess), parameters[0], parameters[1]};
  auto less = root_func_graph->NewCNode(less_inputs);
  AbstractTensorPtr root_less_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  less->set_abstract(root_less_abs);
  // True partial.
  std::vector<AnfNodePtr> true_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(true_func_graph),
                                              parameters[0], parameters[1]};
  auto true_partial = root_func_graph->NewCNode(true_partial_inputs);
  auto true_partial_abs = std::make_shared<FuncGraphAbstractClosure>(true_func_graph, AnalysisContext::DummyContext());
  true_partial->set_abstract(true_partial_abs);

  // False partial.
  std::vector<AnfNodePtr> false_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(false_func_graph),
                                               parameters[0], parameters[1]};
  auto false_partial = root_func_graph->NewCNode(false_partial_inputs);
  auto false_partial_abs =
    std::make_shared<FuncGraphAbstractClosure>(false_func_graph, AnalysisContext::DummyContext());
  false_partial->set_abstract(false_partial_abs);

  // Switch.
  std::vector<AnfNodePtr> switch_inputs{NewValueNode(prim::kPrimSwitch), less, true_partial, false_partial};
  auto switch_node = root_func_graph->NewCNode(switch_inputs);
  auto switch_abs = std::make_shared<AbstractFuncUnion>(true_partial_abs, false_partial_abs);
  switch_node->set_abstract(switch_abs);

  // Call.
  std::vector<AnfNodePtr> call_inputs{switch_node};
  auto root_call_node = root_func_graph->NewCNode(call_inputs);
  auto call_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  root_call_node->set_abstract(call_abs);
  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), root_call_node};
  auto root_return_node = root_func_graph->NewCNode(return_inputs);
  auto root_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  root_return_node->set_abstract(call_abs);
  root_func_graph->set_return(root_return_node);

  // true graph.
  auto true_parameters = true_func_graph->parameters();
  // Call.
  std::vector<AnfNodePtr> true_call_inputs{NewValueNode(root_func_graph), true_parameters[0], true_parameters[1]};
  auto true_call_node = true_func_graph->NewCNode(true_call_inputs);
  auto true_call_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_call_node->set_abstract(true_call_abs);
  // Add.
  std::vector<AnfNodePtr> true_add_inputs{NewValueNode(prim::kPrimAdd), true_parameters[0], true_call_node};
  auto true_add = true_func_graph->NewCNode(true_add_inputs);
  AbstractTensorPtr true_add_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_add->set_abstract(true_add_abs);
  // Return.
  std::vector<AnfNodePtr> true_return_inputs{NewValueNode(prim::kPrimReturn), true_add};
  auto true_return_node = true_func_graph->NewCNode(true_return_inputs);
  AbstractTensorPtr true_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_return_node->set_abstract(true_return_abs);
  true_func_graph->set_return(true_return_node);

  // false graph.
  // Add.
  auto false_parameters = false_func_graph->parameters();
  std::vector<AnfNodePtr> false_add_inputs{NewValueNode(prim::kPrimAdd), false_parameters[0], false_parameters[1]};
  auto false_add = false_func_graph->NewCNode(false_add_inputs);
  const auto &false_add_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  false_add->set_abstract(false_add_abs);
  // Return.
  std::vector<AnfNodePtr> false_return_inputs{NewValueNode(prim::kPrimReturn), false_add};
  auto false_return_node = false_func_graph->NewCNode(false_return_inputs);
  AbstractTensorPtr false_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  false_return_node->set_abstract(false_return_abs);
  false_func_graph->set_return(false_return_node);
  return root_func_graph;
}
}  // namespace

/// Feature: unify runtime.
/// Description: Test the compile graphs.
/// Expectation: As expected.
TEST_F(GraphSchedulerTest, Transform) {
  const char device_name[] = "CPU";
  uint32_t device_id = 0;

  auto ms_context = MsContext::GetInstance();
  int last_execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  bool last_enable_mindrt = ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  uint32_t last_device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string last_device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device_name);

  FuncGraphPtr func_graph = BuildGraphs(); 
  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);
  MS_REGISTER_DEVICE(device_name, TestDeviceContext);
  DeviceContextKey device_context_key{device_name, device_id};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);

  const auto backend = std::make_shared<compile::MindRTBackend>("vm", device_name, 0);
  const auto actor_info = backend->CompileGraphs(func_graph);
  ASSERT_EQ(actor_info.find("kernel_graph") != std::string::npos, true);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, last_execution_mode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, last_enable_mindrt);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, last_device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, last_device_target);
}
}  // namespace runtime
}  // namespace mindspore
