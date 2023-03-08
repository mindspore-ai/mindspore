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
#include "ir/param_info.h"
#include "frontend/operator/ops.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/ccsrc/include/backend/kernel_info.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace session {
using device::KernelInfo;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class KernelGraphTest : public UT::Common {
 public:
  KernelGraphTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(KernelGraphTest, NewValueNode) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto add_value = NewValueNode(MakeValue(static_cast<int64_t>(0)));
  MS_EXCEPTION_IF_NULL(add_value);
  std::vector<int64_t> shape = {1};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape);
  add_value->set_abstract(x_abstract);
  add_value->set_kernel_info(std::make_shared<KernelInfo>());
  auto mutable_kernel_info = dynamic_cast<device::KernelInfo *>(add_value->kernel_info());
  MS_EXCEPTION_IF_NULL(mutable_kernel_info);
  std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
  builder->SetOutputsFormat({kOpFormat_FRAC_Z});
  builder->SetOutputsDeviceType({kFloat32->type_id()});
  mutable_kernel_info->set_select_kernel_build_info(builder->Build());
  auto new_value = kernel_graph->NewValueNode(add_value);
  EXPECT_NE(new_value, nullptr);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(new_value, 0)[0], 1);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(new_value, 0), kFloat32->type_id());
  EXPECT_EQ(AnfAlgo::GetOutputFormat(new_value, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetOutputDeviceDataType(new_value, 0), kTypeUnknown);
}

TEST_F(KernelGraphTest, NewParameter) {
  auto anf_graph = std::make_shared<FuncGraph>();
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test nullptr as input
  auto new_paramter = kernel_graph->NewParameter();
  EXPECT_NE(new_paramter, nullptr);
  EXPECT_TRUE(new_paramter->isa<Parameter>());
  EXPECT_EQ(AnfAlgo::GetOutputFormat(new_paramter, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetOutputDeviceDataType(new_paramter, 0), kMetaTypeNone);
  // test non-weight parameter node as input
  std::vector<int64_t> shape = {2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape);
  auto non_weight_parameter = anf_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(non_weight_parameter);
  non_weight_parameter->set_abstract(x_abstract);
  auto new_non_weight_parameter = kernel_graph->NewParameter(non_weight_parameter);
  EXPECT_NE(new_non_weight_parameter, nullptr);
  new_non_weight_parameter->set_name("non_weight_parameter");
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(new_non_weight_parameter, 0)[1], 32);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(new_non_weight_parameter, 0), kFloat32->type_id());
  EXPECT_EQ(AnfAlgo::GetOutputFormat(new_non_weight_parameter, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetOutputDeviceDataType(new_non_weight_parameter, 0), kFloat32->type_id());
  EXPECT_EQ(new_non_weight_parameter->name(), "non_weight_parameter");
  // test weight parameter node as input
  auto weight_parameter_node = anf_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(weight_parameter_node);
  auto param_value_new = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, shape);
  weight_parameter_node->set_default_param(param_value_new);
  weight_parameter_node->set_abstract(x_abstract);
  auto new_weight_parameter_node = kernel_graph->NewParameter(weight_parameter_node);
  EXPECT_NE(new_weight_parameter_node, nullptr);
  EXPECT_TRUE(new_weight_parameter_node->has_default());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(new_weight_parameter_node, 0)[2], 224);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(new_weight_parameter_node, 0), kFloat32->type_id());
  EXPECT_EQ(AnfAlgo::GetOutputFormat(new_weight_parameter_node, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetOutputDeviceDataType(new_weight_parameter_node, 0), kTypeUnknown);
}

TEST_F(KernelGraphTest, NewCNode) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto add_value = NewValueNode(prim::kPrimAdd);
  std::vector<AnfNodePtr> inputs = {add_value};
  auto new_cnode = kernel_graph->NewCNode(inputs);
  EXPECT_NE(new_cnode, nullptr);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(new_cnode), prim::kPrimAdd->name());
  EXPECT_TRUE(common::AnfAlgo::GetOutputInferShape(new_cnode, 0).empty());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(new_cnode, 0), kMetaTypeNone);
}

TEST_F(KernelGraphTest, MutableInputs) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto x_parameter = kernel_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(x_parameter);
  x_parameter->set_name("x_parameter");
  auto y_parameter = kernel_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(y_parameter);
  y_parameter->set_name("y_parameter");
  std::vector<AnfNodePtr> inputs = {x_parameter, y_parameter};
  auto mutable_inputs = kernel_graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(mutable_inputs);
  *mutable_inputs = inputs;
  auto first_input = kernel_graph->inputs()[0];
  MS_EXCEPTION_IF_NULL(first_input);
  auto first_parameter = first_input->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(first_parameter);
  EXPECT_EQ(first_parameter->name(), "x_parameter");
  auto second_input = kernel_graph->inputs()[1];
  MS_EXCEPTION_IF_NULL(second_input);
  auto second_parameter = second_input->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(second_parameter);
  EXPECT_EQ(second_parameter->name(), "y_parameter");
}

TEST_F(KernelGraphTest, SetExecOrderByDefault) {
  /*
   * define kernel graph:
   *     x ----- y
   *         add ----- z
   *               mul
   *              return
   */
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<int64_t> shape = {2, 32, 224, 224};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape);

  auto x_parameter = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(x_parameter);
  x_parameter->set_name("x_parameter");
  x_parameter->set_abstract(abstract);
  auto y_parameter = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(y_parameter);
  y_parameter->set_name("y_parameter");
  y_parameter->set_abstract(abstract);
  std::vector<AnfNodePtr> add_inputs = {NewValueNode(prim::kPrimAdd), x_parameter, y_parameter};
  auto add = kernel_graph->NewCNode(add_inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_abstract(abstract);

  auto z_parameter = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(z_parameter);
  z_parameter->set_name("z_parameter");
  z_parameter->set_abstract(abstract);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(prim::kPrimMul), add, z_parameter};
  auto mul = kernel_graph->NewCNode(mul_inputs);
  MS_EXCEPTION_IF_NULL(mul);
  mul->set_abstract(abstract);

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), mul};
  auto make_tuple = kernel_graph->NewCNode(make_tuple_inputs);
  kernel_graph->set_output(make_tuple);
  // test outputs() function
  auto outputs = kernel_graph->outputs();
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(outputs[0]), prim::kPrimMul->name());
  // test SetExecOrderByDefault() function
  kernel_graph->SetExecOrderByDefault();
  auto execution_order = kernel_graph->execution_order();
  EXPECT_EQ(execution_order.size(), 2);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(execution_order[0]), prim::kPrimAdd->name());
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(execution_order[1]), prim::kPrimMul->name());
  // test set_execution_order() function
  kernel_graph->set_execution_order({add});
  execution_order = kernel_graph->execution_order();
  EXPECT_EQ(execution_order.size(), 1);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(execution_order[0]), prim::kPrimAdd->name());
}

TEST_F(KernelGraphTest, SetGraphId) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  kernel_graph->set_graph_id(1);
  EXPECT_EQ(kernel_graph->graph_id(), 1);
}

}  // namespace session
}  // namespace mindspore
