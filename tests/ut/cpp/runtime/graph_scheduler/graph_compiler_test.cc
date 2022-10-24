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

#include "common/common_test.h"
#include "abstract/abstract_function.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/hardware/device_context.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace runtime {
using KernelGraph = session::KernelGraph;
using FuncGraphAbstractClosure = abstract::FuncGraphAbstractClosure;
using AnalysisContext = abstract::AnalysisContext;
using DeviceContextKey = device::DeviceContextKey;
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;
using DeviceType = device::DeviceType;
using AddressPtr = kernel::AddressPtr;

class TestDeviceAddress : public DeviceAddress {
 public:
  TestDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  ~TestDeviceAddress() {}
  virtual bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const {
    return true;
  }
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                                const std::string &format) const {
    return true;
  }
  virtual void *GetMutablePtr() const { return nullptr; }
  virtual void ClearDeviceMemory() {}
};

class TestKernelMod : public kernel::KernelMod {
 public:
  TestKernelMod() = default;
  ~TestKernelMod() override = default;
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    return true;
  }
  std::vector<kernel::KernelAttr> GetOpSupport() override { return {}; }
};

class TestADeviceResManager : public device::DeviceResManager {
 public:
  TestADeviceResManager() = default;
  ~TestADeviceResManager() override = default;

  virtual bool AllocateMemory(DeviceAddress *const &address, size_t size) const { return true; }
  virtual void FreeMemory(DeviceAddress *const &address) const {}
  virtual void *AllocateMemory(size_t size) const { return nullptr; }
  virtual void FreeMemory(void *const ptr) const {}
  virtual DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format,
                                               TypeId type_id, const ShapeVector &shape) const {
    return std::make_shared<TestDeviceAddress>(nullptr, 0);
  }
};

class TestAKernelExecutor : public device::KernelExecutor {
 public:
  TestAKernelExecutor() = default;
  ~TestAKernelExecutor() override = default;
  virtual void CreateKernel(const std::vector<CNodePtr> &nodes) const {
    for (const auto node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (node->kernel_info() == nullptr) {
        auto kernel_info = std::make_shared<device::KernelInfo>();
        std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
        kernel_info->set_select_kernel_build_info(builder->Build());
        node->set_kernel_info(kernel_info);
      } else {
        const auto &kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
        if (kernel_info->select_kernel_build_info() == nullptr) {
          std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
          kernel_info->set_select_kernel_build_info(builder->Build());
        }
      }
      AnfAlgo::SetOutputAddr(std::make_shared<TestDeviceAddress>(nullptr, 0), 0, node.get());
      auto kernel_mod_ptr = std::make_shared<TestKernelMod>();
      kernel_mod_ptr->SetInputSizeList({4});
      kernel_mod_ptr->SetOutputSizeList({4});
      kernel_mod_ptr->SetWorkspaceSizeList({4});
      AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
    }
  }
};

class TestADeviceContext : public device::DeviceInterface<TestAKernelExecutor, TestADeviceResManager> {
 public:
  explicit TestADeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~TestADeviceContext() override = default;

  virtual void Initialize() {}
  virtual DeviceType GetDeviceType() const { return DeviceType::kCPU; }
  device::RunMode GetRunMode(const FuncGraphPtr &func_graph) const override { return device::RunMode::kKernelMode; }
};

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
  auto device_context = std::make_shared<TestADeviceContext>(device_context_key);
  auto graph_id = compiler->CompileGraph(segment, outputs, device_context.get(), device::RunMode::kKernelMode, false);
  const auto &kernel_graph = compiler->Fetch(graph_id);
  ASSERT_EQ(2, kernel_graph->execution_order().size());
}
}  // namespace runtime
}  // namespace mindspore
