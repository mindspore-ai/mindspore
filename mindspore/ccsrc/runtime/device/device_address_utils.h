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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include "runtime/hardware/device_context.h"
#include "runtime/pynative/op_compiler.h"

namespace mindspore {
using device::DeviceContext;
namespace runtime {
// Extract the methods related to DeviceAddress in GraphCompiler to the DeviceAddressUtils class.
class BACKEND_EXPORT DeviceAddressUtils {
 public:
  static void CopyNonTensorDataToDevice(const device::DeviceContext *device_context,
                                        const device::DeviceAddressPtr &device_address);
  static void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static device::DeviceAddressPtrList CreateDeviceAddressForTensorValue(const DeviceContext *device_context,
                                                                        const ValuePtr &node_value, size_t output_idx,
                                                                        const ValueNodePtr &value_node);
  static void CreateValueNodeDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph,
                                              bool is_gradient_out);
  static vector<device::DeviceAddressPtr> CreateGraphOutputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info,
                                                                         const abstract::AbstractBasePtr &out_abstract);
  static void CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static void CreateDeviceAddressByMapTensorNode(const DeviceContext *device_context, const AnfNodePtr &node,
                                                 size_t index);
  static void UpdateDeviceAddressForInplaceNode(const KernelGraphPtr &graph);
  static void UpdateDeviceAddress(const session::AnfWithOutIndex &cur_pair,
                                  const session::AnfWithOutIndex &origin_pair);
  static void UpdateDeviceAddressForRefNode(const KernelGraphPtr &graph);
  static device::DeviceAddressPtr CloneEmptyDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                          const DeviceContext *device_context);
  static void CreateGraphOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);

  static device::DeviceAddressPtr GetInputAddressForRef(const AnfNodePtr &node,
                                                        const OpCompilerInfoPtr &op_compiler_info);
  static device::DeviceAddressPtr GetOutputAddressForRef(const AnfNodePtr &node,
                                                         const OpCompilerInfoPtr &op_compiler_info,
                                                         size_t output_index);
  static size_t GetTensorDeviceSize(const DeviceContext *device_context, const AnfNodePtr &node,
                                    const ShapeVector &shape, const string &format, TypeId dtype, size_t output_index);

  // Overloading
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t index,
                                       const tensor::TensorPtr &tensor);
  static void MallocForInput(const DeviceContext *device_context, const tensor::TensorPtr &tensor);
  static void MallocForInput(const DeviceContext *device_context, const std::optional<tensor::TensorPtr> &val);
  static void MallocForInput(const DeviceContext *device_context, const std::vector<tensor::TensorPtr> &tensors);
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t index,
                                       const std::optional<tensor::TensorPtr> &val);
  template <typename T>
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t index,
                                       const std::vector<T> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      CreateInputTensorAddress(device_context, index, inputs[i]);
    }
  }

  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const tensor::TensorPtr &tensor);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const std::optional<tensor::TensorPtr> &val);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const ScalarPtr &scalar_value);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const StringImmPtr &string_imm);
  template <typename T>
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const abstract::AbstractBasePtr &abs, size_t index, const T &t) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(abs);

    const auto &shape = abs->GetShape();
    const auto &type = abs->GetType();
    const auto &value = abs->GetValue();
    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(shape, type, value);
    kernel_tensor->set_device_name(device_context->device_context_key().device_name_);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    device_address->set_from_persistent_mem(true);

    if (device_address->GetPtr() == nullptr) {
      CopyNonTensorDataToDevice(device_context, device_address);
    }
    MS_LOG(DEBUG) << "Create input " << abs->ToString() << " device address for " << index
                  << "th input, Shape: " << shape->ToString() << ", Type: " << type->ToString()
                  << ", Value: " << (value ? value->ToString() : "nullptr") << " device address:" << device_address;
    return device_address;
  }

  static void CreateOutputTensorAddress(DeviceContext *device_context, const std::vector<tensor::TensorPtr> &outputs);

  static void MallocForOutputs(DeviceContext *device_context, const std::vector<tensor::TensorPtr> &outputs);

  static device::DeviceAddressPtr CreateOutputAddress(const DeviceContext *device_context,
                                                      const abstract::AbstractBasePtr &abs, size_t index,
                                                      const tensor::TensorPtr &tensor, const string &format = "");

  static device::DeviceAddressPtr CreateOutputAddress(const DeviceContext *device_context,
                                                      const abstract::AbstractBasePtr &abs, size_t index,
                                                      const tensor::TensorPtr &tensor,
                                                      const pynative::DeviceAddressPromisePtr &promise,
                                                      const string &format = "");

  static device::DeviceAddressPtr CreateWorkspaceAddress(const DeviceContext *device_context,
                                                         const size_t &workspace_size);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_
