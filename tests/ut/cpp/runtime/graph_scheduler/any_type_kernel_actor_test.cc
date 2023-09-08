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

#include "tests/ut/cpp/common/device_common_test.h"

#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace runtime {
using namespace test;
class AnyTypeKernelActorTest : public UT::Common {
 public:
  AnyTypeKernelActorTest() {}
};

namespace {
std::pair<FuncGraphPtr, KernelGraphPtr> BuildAnyTypeGraph() {
  std::vector<int64_t> shp{2, 2};
  auto func_graph = std::make_shared<FuncGraph>();
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat64, shp);
  auto parameter_x = func_graph->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractAny>();
  auto parameter_y = func_graph->add_parameter();
  parameter_y->set_abstract(abstract_y);

  // Add.
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameter_x, parameter_y};
  auto add_node = func_graph->NewCNode(add_inputs);
  auto add_abs = std::make_shared<abstract::AbstractAny>();
  add_node->set_abstract(add_abs);

  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), add_node};
  auto return_node = func_graph->NewCNode(return_inputs);
  auto return_abs = std::make_shared<abstract::AbstractAny>();
  return_node->set_abstract(return_abs);
  func_graph->set_return(return_node);

  auto kernel_graph = std::make_shared<KernelGraph>();
  auto backend_abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat64, shp);
  auto backend_parameter_x = kernel_graph->add_parameter();
  backend_parameter_x->set_abstract(backend_abstract_x);

  auto backend_abstract_y = std::make_shared<abstract::AbstractAny>();
  auto backend_parameter_y = kernel_graph->add_parameter();
  backend_parameter_y->set_abstract(backend_abstract_y);

  // Add.
  std::vector<AnfNodePtr> backend_add_inputs{NewValueNode(prim::kPrimAdd), backend_parameter_x, backend_parameter_y};
  auto backend_add_node = kernel_graph->NewCNode(backend_add_inputs);
  auto backend_add_abs = std::make_shared<abstract::AbstractAny>();
  backend_add_node->set_abstract(backend_add_abs);

  // Return.
  std::vector<AnfNodePtr> backend_return_inputs{NewValueNode(prim::kPrimReturn), backend_add_node};
  auto backend_return_node = kernel_graph->NewCNode(backend_return_inputs);
  auto backend_return_abs = std::make_shared<abstract::AbstractAny>();
  backend_return_node->set_abstract(backend_return_abs);
  kernel_graph->set_return(backend_return_node);
  kernel_graph->set_execution_order({backend_add_node});
  kernel_graph->input_nodes_.emplace_back(backend_parameter_x);
  kernel_graph->input_nodes_.emplace_back(backend_parameter_y);

  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({kernel_graph->output()}, {func_graph->output()});
  kernel_graph->FrontBackendMapAdd(add_node, backend_add_node);
  kernel_graph->FrontBackendMapAdd(parameter_x, backend_parameter_x);
  kernel_graph->FrontBackendMapAdd(parameter_y, backend_parameter_y);

  return std::make_pair(func_graph, kernel_graph);
}
}  // namespace

/// Feature: Pyexecute any type output.
/// Description: Test the compile of any type.
/// Expectation: As expected.
TEST_F(AnyTypeKernelActorTest, RunOpData) {
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
  MS_REGISTER_DEVICE(device_name, TestDeviceContext);
  DeviceContextKey device_context_key{device_name, device_id};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);

  auto graph_pair = BuildAnyTypeGraph();
  auto func_graph = graph_pair.first;
  auto kernel_graph = graph_pair.second;
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  const auto &any_type_kernel_actor =
    std::make_shared<AnyTypeKernelActor>(kernel_graph->ToString() + "_AnyTypeKernelActor", kernel_graph,
                                         device_context.get(), memory_manager_actor->GetAID(), nullptr, nullptr);

  using DataType = float;
  DataType input_0 = 2.0;
  DataType input_1 = 2.0;
  ShapeVector shape = {1};

  OpContext<DeviceAddress> op_context;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = 140429;
  op_context.results_ = &result;

  auto device_address0 = device_context->device_res_manager_->CreateDeviceAddress(
    &input_0, sizeof(DataType), kOpFormat_DEFAULT, TypeId::kNumberTypeFloat32, shape);
  auto op_data0 = std::make_shared<OpData<DeviceTensor>>(any_type_kernel_actor->GetAID(), device_address0.get(), 0);

  auto device_address1 = device_context->device_res_manager_->CreateDeviceAddress(
    &input_1, sizeof(DataType), kOpFormat_DEFAULT, TypeId::kNumberTypeFloat32, shape);
  auto op_data1 = std::make_shared<OpData<DeviceTensor>>(any_type_kernel_actor->GetAID(), device_address1.get(), 1);

  any_type_kernel_actor->input_datas_num_ = 2;
  any_type_kernel_actor->any_type_parameter_indexes_.emplace_back(1);
  any_type_kernel_actor->RunOpData(op_data1.get(), &op_context);

  ASSERT_EQ(any_type_kernel_actor->input_op_datas_[op_context.sequential_num_].size(), 1);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, last_execution_mode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, last_enable_mindrt);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, last_device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, last_device_target);
}
}  // namespace runtime
}  // namespace mindspore
