/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "frontend/operator/ops.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace session {

class SessionBasicTest : public UT::Common {
 public:
  SessionBasicTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(SessionBasicTest, ConstructKernelGraph) {
  /*
   * define kernel graph:
   *     x ----- y
   *         add ----- z
   *               mul
   *              return
   */
  auto anf_graph = std::make_shared<FuncGraph>();
  std::vector<int64_t> shape = {2, 32, 224, 224};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape);
  EXPECT_NE(abstract, nullptr);

  auto original_x_parameter = anf_graph->add_parameter();
  EXPECT_NE(original_x_parameter, nullptr);
  original_x_parameter->set_name("original_x_parameter");
  original_x_parameter->set_abstract(abstract);
  auto original_y_parameter = anf_graph->add_parameter();
  EXPECT_NE(original_y_parameter, nullptr);
  original_y_parameter->set_name("original_y_parameter");
  original_y_parameter->set_abstract(abstract);
  std::vector<AnfNodePtr> add_inputs = {NewValueNode(prim::kPrimAdd), original_x_parameter, original_y_parameter};
  auto original_add = anf_graph->NewCNode(add_inputs);
  EXPECT_NE(original_add, nullptr);
  original_add->set_abstract(abstract);

  auto original_z_parameter = anf_graph->add_parameter();
  EXPECT_NE(original_z_parameter, nullptr);
  original_z_parameter->set_name("original_z_parameter");
  original_z_parameter->set_abstract(abstract);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(prim::kPrimMul), original_add, original_z_parameter};
  auto original_mul = anf_graph->NewCNode(mul_inputs);
  EXPECT_NE(original_mul, nullptr);
  original_mul->set_abstract(abstract);

  std::vector<AnfNodePtr> lst = {original_add, original_mul};
  std::vector<AnfNodePtr> outputs = {original_mul};
  session::SessionPtr sess = std::make_shared<session::AscendSession>();
  sess->Init(0);
  auto kernel_graph = sess->ConstructKernelGraph(lst, outputs);
  EXPECT_NE(kernel_graph, nullptr);

  auto inputs = kernel_graph->inputs();
  EXPECT_EQ(inputs.size(), 3);
  auto first_input = inputs[0]->cast<ParameterPtr>();
  EXPECT_NE(first_input, nullptr);
  EXPECT_EQ(first_input->name(), "original_x_parameter");
  auto second_input = inputs[1]->cast<ParameterPtr>();
  EXPECT_NE(second_input, nullptr);
  EXPECT_EQ(second_input->name(), "original_y_parameter");
  auto third_input = inputs[2]->cast<ParameterPtr>();
  EXPECT_NE(third_input, nullptr);
  EXPECT_EQ(third_input->name(), "original_z_parameter");
  kernel_graph->SetExecOrderByDefault();
  auto execution_order = kernel_graph->execution_order();
  EXPECT_EQ(execution_order.size(), 2);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(execution_order[0]), prim::kPrimAdd->name());
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(execution_order[1]), prim::kPrimMul->name());
  auto new_outputs = kernel_graph->outputs();
  EXPECT_EQ(new_outputs.size(), 1);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(new_outputs[0]), prim::kPrimMul->name());
};

}  // namespace session
}  // namespace mindspore