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
class GraphCompilerTest : public UT::Common {
 public:
  GraphCompilerTest() {}
};

/// Feature: control flow support dynamic shape.
/// Description: Test the parse interface.
/// Expectation: As expected.
TEST_F(GraphCompilerTest, CompileGraph) {
  std::vector<int64_t> shp{2, 2};
  abstract::AbstractTensorPtr abs;

  // Func graph.
  auto func_graph = std::make_shared<FuncGraph>();

  // Parameter.
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_x = func_graph->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_y = func_graph->add_parameter();
  parameter_y->set_abstract(abstract_y);
  auto parameters = func_graph->parameters();

  // Add.
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameters[0], parameters[1]};
  auto add_node = func_graph->NewCNode(add_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  add_node->set_abstract(abs);

  // Reshape.
  std::vector<AnfNodePtr> reshape_inputs{NewValueNode(prim::kPrimReshape), add_node};
  auto reshape_node = func_graph->NewCNode(reshape_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  reshape_node->set_abstract(abs);

  // sub.
  std::vector<AnfNodePtr> sub_inputs{NewValueNode(prim::kPrimSub), reshape_node, parameters[0]};
  auto sub_node = func_graph->NewCNode(sub_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  sub_node->set_abstract(abs);

  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), sub_node};
  auto return_node = func_graph->NewCNode(return_inputs);
  func_graph->set_return(return_node);

  std::vector<AnfNodePtr> nodes{add_node, reshape_node, sub_node};
  std::vector<AnfNodePtr> outputs{sub_node};
  auto segment = std::make_shared<GraphSegment>(nodes, false);

  auto compiler = std::make_shared<GraphCompiler>();
  DeviceContextKey device_context_key{"CPU", 0};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  auto graph_id = compiler->CompileGraph(segment, outputs, device_context.get(), device::RunMode::kKernelMode, false);
  const auto &kernel_graph = compiler->Fetch(graph_id);
  ASSERT_EQ(2, kernel_graph->execution_order().size());
}
}  // namespace runtime
}  // namespace mindspore
