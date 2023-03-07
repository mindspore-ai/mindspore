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
#include <memory>
#include <vector>
#include <string>
#include "include/backend/kernel_graph.h"
#include "backend/common/session/session_basic.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "backend/common/mem_reuse/kernel_refcount.h"
#include "include/backend/kernel_info.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"
#include "frontend/operator/ops.h"
#include "utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "pipeline/jit/resource.h"
#include "backend/common/mem_reuse/mem_reuse.h"

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace memreuse {
using session::KernelGraph;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
class TestMemReuseWithPy : public UT::Common {
 public:
  TestMemReuseWithPy() : getPyFun_("gtest_input.mem_reuse.mem_reuse_test", true) {}
  void SetUp() {}
  void TearDown() {}

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};
static KernelGraphPtr CreateKernelGraph() {
  /* CreateKernelGraph()
   * @mindspore
   * def f(x):
   *     z=conv2d(x, y)
   *     ret=relu(z)
   *     return ret
   */
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  std::vector<int64_t> shp = {1, 3, 3, 4};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};

  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  y_tensor->set_device_info(device_info);
  tensor::TensorPtr z_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  z_tensor->set_device_info(device_info);
  auto y_const = NewValueNode(y_tensor);
  auto z_const = NewValueNode(z_tensor);
  y_const->set_abstract(y_tensor->ToAbstract());
  z_const->set_abstract(z_tensor->ToAbstract());
  g->MutableInputs()->push_back(y_const);
  g->MutableInputs()->push_back(z_const);

  auto p_conv = std::make_shared<Primitive>("Conv2D");
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_conv->set_attr("input_names", input_names_v);
  p_conv->set_attr("output_names", output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_conv));
  inputs.push_back(y_const);
  inputs.push_back(z_const);

  auto kernelptr_first = g->NewCNode(inputs);
  kernelptr_first->set_abstract(y_tensor->ToAbstract());
  auto tbe_kernel_pack_first = std::make_shared<kernel::KernelPack>();
  auto kernel_mod_first = std::make_shared<kernel::TbeKernelMod>(tbe_kernel_pack_first);
  auto kernel_info_first = std::make_shared<device::KernelInfo>();
  kernel_info_first->set_kernel_mod(kernel_mod_first);
  kernelptr_first->set_kernel_info(kernel_info_first);
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(mindspore::kernel::kPatternConvolution);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_first.get());

  CNodePtr next_cnode_ptr = kernelptr_first;
  auto p_relu = std::make_shared<Primitive>("ReLU6");
  std::vector<std::string> relu_input_names = {"x"};
  std::vector<std::string> relu_output_names = {"output"};
  ValuePtr relu_input_names_v = MakeValue(relu_input_names);
  ValuePtr relu_output_names_v = MakeValue(relu_output_names);
  p_relu->set_attr("input_names", relu_input_names_v);
  p_relu->set_attr("output_names", relu_output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_relu));
  inputs.push_back(next_cnode_ptr);

  auto kernelptr_floor = g->NewCNode(inputs);
  kernelptr_floor->set_abstract(y_tensor->ToAbstract());
  auto tbe_kernel_pack_floor = std::make_shared<kernel::KernelPack>();
  auto kernel_mod_floor = std::make_shared<kernel::TbeKernelMod>(tbe_kernel_pack_floor);
  auto kernel_info_floor = std::make_shared<device::KernelInfo>();
  kernel_info_floor->set_kernel_mod(kernel_mod_floor);
  kernelptr_floor->set_kernel_info(kernel_info_floor);
  KernelBuildInfoBuilder relu_builder;
  relu_builder.SetInputsFormat({kOpFormat_NCHW});
  relu_builder.SetOutputsFormat({kOpFormat_NCHW});
  relu_builder.SetInputsDeviceType({kFloat32->type_id()});
  relu_builder.SetOutputsDeviceType({kFloat32->type_id()});
  relu_builder.SetKernelType(KernelType::TBE_KERNEL);
  relu_builder.SetFusionType(kernel::kPatternElemWise);
  relu_builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
  next_cnode_ptr = kernelptr_floor;

  // return res
  auto p_return = std::make_shared<Primitive>("Return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(next_cnode_ptr);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(y_tensor->ToAbstract());
  g->set_return(ret);
  return g;
}

static KernelGraphPtr CreateGraphWithExecOrder() {
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
  return kernel_graph;
}

TEST_F(TestMemReuseWithPy, KernelRef) {
  KernelRefCountPtr kernel_ref_count_ptr = std::make_shared<KernelRefCount>();
  ASSERT_NE(kernel_ref_count_ptr, nullptr);
  int ref_count = kernel_ref_count_ptr->ref_count_;
  int offset = kernel_ref_count_ptr->offset_;
  size_t size = kernel_ref_count_ptr->size_;
  int index = kernel_ref_count_ptr->index_;
  ASSERT_EQ(ref_count, 0);
  ASSERT_EQ(offset, 0);
  ASSERT_EQ(size, 0);
  ASSERT_EQ(index, -1);
  index = 3;
  size = 512;
  RefCountType ref_count_type_in = mindspore::memreuse::kDynamicRefCount;
  kernel_ref_count_ptr->SetKernelRefCountInfo(index, size, ref_count_type_in);
  ASSERT_EQ(kernel_ref_count_ptr->index_, 3);
  ASSERT_EQ(kernel_ref_count_ptr->size_, 512);
  KernelDefPtr kernel_def_ptr = std::make_shared<KernelDef>();
  ASSERT_NE(kernel_def_ptr, nullptr);
  MembufPtr membuf_ptr = std::make_shared<Membuf>();
  ASSERT_NE(membuf_ptr, nullptr);
}

TEST_F(TestMemReuseWithPy, TestSetInfo) {
  KernelGraphPtr g = CreateKernelGraph();
  ASSERT_NE(g, nullptr);
  g->SetExecOrderByDefault();
  std::vector<FuncGraphPtr> graphs{g};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(g);
  ASSERT_EQ(manager->all_nodes().size(), 8);
  MemReuseUtilPtr mem_reuse_util_ptr = std::make_shared<MemReuseUtil>();
  ASSERT_NE(mem_reuse_util_ptr, nullptr);
  auto ret = mem_reuse_util_ptr->InitDynamicKernelRef(g.get());
  ASSERT_EQ(ret, true);
  mem_reuse_util_ptr->SetWorkSpaceList();
  ASSERT_EQ(mem_reuse_util_ptr->total_wk_ref_list_.size(), 0);
  mem_reuse_util_ptr->SetReuseRefCount();
  ASSERT_EQ(mem_reuse_util_ptr->total_refs_list_.size(), 0);
  auto def_list = mem_reuse_util_ptr->kernel_def_ptr_list();
  ASSERT_EQ(def_list.size(), 0);
  auto exec_graph = CreateGraphWithExecOrder();
  ASSERT_NE(exec_graph, nullptr);
}
}  // namespace memreuse
}  // namespace mindspore
