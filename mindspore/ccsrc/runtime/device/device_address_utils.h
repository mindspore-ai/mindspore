/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include <utility>
#include "runtime/hardware/device_context.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/device/multi_stream_controller.h"
#include "kernel/kernel.h"
#include "mindapi/base/type_traits.h"

template <typename T>
struct is_optional : public std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : public std::true_type {};

namespace mindspore {
using device::DeviceContext;
namespace runtime {
// Extract the methods related to DeviceAddress in GraphCompiler to the DeviceAddressUtils class.
class BACKEND_EXPORT DeviceAddressUtils {
 public:
  static void CreateKernelTensor(const device::DeviceAddressPtr &device_address, const tensor::BaseTensorPtr &tensor);
  static void CreateKernelTensor(const device::DeviceAddressPtr &device_address, const AbstractBasePtr &abs);
  static void CreateKernelTensor(const ValuePtr &input_value);
  static void CreateKernelTensor(const tensor::TensorPtr &input_tensor);
  static void CopyNoneTensorDataToDevice(const device::DeviceContext *device_context,
                                         const device::DeviceAddressPtr &device_address, const ShapeVector &shape = {});
  static void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static device::DeviceAddressPtrList CreateDeviceAddressForTensorValue(const DeviceContext *device_context,
                                                                        const ValuePtr &node_value, size_t output_idx,
                                                                        const ValueNodePtr &value_node);
  static void CreateValueNodeDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph,
                                              bool is_gradient_out);

  static vector<device::DeviceAddressPtr> CreateGraphOutputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info,
                                                                         const abstract::AbstractBasePtr &out_abstract,
                                                                         size_t stream_id);

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
  static size_t GetTensorDeviceSize(const DeviceContext *device_context, const AnfNodePtr &node,
                                    const ShapeVector &shape, const string &format, TypeId dtype, size_t output_index);

  // Overloading
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                       const tensor::BaseTensorPtr &tensor);
  static void MallocForInput(const DeviceContext *device_context, const tensor::BaseTensorPtr &tensor);
  static void MallocForInput(const DeviceContext *device_context, const std::optional<tensor::BaseTensorPtr> &val);
  static void MallocForInput(const DeviceContext *device_context, const std::vector<tensor::BaseTensorPtr> &tensors);
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                       const std::optional<tensor::BaseTensorPtr> &val);
  template <typename T>
  static void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                       const std::vector<T> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      CreateInputTensorAddress(device_context, stream_id, index, inputs[i]);
    }
  }

  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const tensor::BaseTensorPtr &tensor);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const std::optional<tensor::BaseTensorPtr> &val);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const ScalarPtr &scalar_value);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const StringImmPtr &string_imm);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index,
                                                     const TypePtr &type_ptr);
  template <typename T>
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, size_t stream_id,
                                                     const abstract::AbstractBasePtr &abs, size_t index, const T &t) {
    MS_EXCEPTION_IF_NULL(device_context);
    auto tmp_abs = abs;
    if (abs == nullptr) {
      tmp_abs = t->ToAbstract()->Broaden();
    }
    auto shape = tmp_abs->GetShape();
    auto type = tmp_abs->GetType();
    auto value = tmp_abs->GetValue();
    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(shape, type, value);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    device_address->set_from_persistent_mem(true);

    if (device_address->GetPtr() == nullptr) {
      CopyNoneTensorDataToDevice(device_context, device_address);
    }
    MS_LOG(DEBUG) << "Create input " << tmp_abs->ToString() << " device address for " << index
                  << "th input, Shape: " << shape->ToString() << ", Type: " << type->ToString()
                  << ", Value: " << (value ? value->ToString() : "nullptr") << " device address:" << device_address;
    return device_address;
  }

  static void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                        const std::vector<tensor::BaseTensorPtr> &outputs);
  static void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                        const tensor::BaseTensorPtr &output_tensor, size_t size);

  static void MallocForOutputs(const DeviceContext *device_context, const std::vector<tensor::BaseTensorPtr> &outputs);

  static device::DeviceAddressPtr CreateWorkspaceAddressWithoutKernelTensor(const DeviceContext *device_context,
                                                                            size_t stream_id,
                                                                            const size_t &workspace_size);

  static device::DeviceAddressPtr CreateWorkspaceAddress(const DeviceContext *device_context, size_t stream_id,
                                                         const size_t &workspace_size);

  static void UpdateDeviceAddressHostInfoByNode(const device::DeviceAddressPtr &addr, const AnfNodePtr &node,
                                                size_t output_idx);
  static device::DeviceAddressPtr CreateDeviceAddress(const DeviceContext *device_context,
                                                      const tensor::BaseTensorPtr &tensor,
                                                      const ShapeVector &real_shape, const size_t &stream_id);

  // Convert tensor to contiguous tensor.
  static void ConvertContiguousTensorSync(const tensor::BaseTensorPtr &tensor);

  // Convert old_device_address to contiguous device address.
  static device::DeviceAddressPtr ConvertContiguousDeviceAddress(const DeviceContext *device_context,
                                                                 const device::DeviceAddressPtr &old_device_address,
                                                                 bool is_sync);

  template <typename... T>
  static void ProcessCrossStreamAddress(const std::string &op_name, const DeviceContext *device_context,
                                        size_t op_stream_id, const T &... args) {
    // memory_stream_addresses pair : memory_stream_id, address.
    std::vector<std::pair<uint32_t, void *>> cross_stream_addresses;
    (GetCrossStreamAddressInfo(op_stream_id, &cross_stream_addresses, args), ...);
    if (cross_stream_addresses.empty()) {
      return;
    }

    device::MultiStreamController::GetInstance()->Refresh(device_context);
    auto task_id_on_stream =
      device::MultiStreamController::GetInstance()->LaunchTaskIdOnStream(device_context, op_stream_id);
    MS_LOG(DEBUG) << "Launch stream_id:" << op_stream_id << ", task id:" << task_id_on_stream << ", op_name:" << op_name
                  << ", cross_stream_addresses size:" << cross_stream_addresses.size();
    device::MultiStreamController::GetInstance()->RecordEvent(device_context, task_id_on_stream, op_stream_id,
                                                              cross_stream_addresses);
  }

 private:
  // Whether device address of anf node is valid and device address type
  // is consistent with device type, for example, device address type
  // DeviceType::kGPU should be used on GPU device
  static bool NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &node, size_t index);

  static void GetCrossStreamAddressInfoFromInput(size_t op_stream_id,
                                                 std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                                 const tensor::BaseTensorPtr &tensor);

  static void GetCrossStreamAddressInfoFromInput(size_t op_stream_id,
                                                 std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                                 const mindspore::kernel::KernelTensor *tensor);

  static void GetCrossStreamAddressInfoFromInput(size_t op_stream_id,
                                                 std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                                 const device::DeviceAddressPtr &device_address);

  template <typename T>
  static void GetCrossStreamAddressInfo(size_t op_stream_id,
                                        std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                        const std::optional<T> &opt) {
    if (opt.has_value()) {
      return GetCrossStreamAddressInfo(op_stream_id, cross_stream_addresses, opt.value());
    }
  }

  template <typename T>
  static void GetCrossStreamAddressInfo(size_t op_stream_id,
                                        std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                        const std::vector<T> &inputs) {
    if constexpr (!std::is_base_of_v<T, tensor::BaseTensorPtr> &&
                  !std::is_same_v<T, mindspore::kernel::KernelTensor *> &&
                  !std::is_same_v<T, device::DeviceAddressPtr>) {
      return;
    }
    for_each(inputs.begin(), inputs.end(), [op_stream_id, cross_stream_addresses](auto item) {
      GetCrossStreamAddressInfo(op_stream_id, cross_stream_addresses, item);
    });
  }

  template <typename T, typename = typename std::enable_if_t<!is_vector<T>::value && !is_optional<T>::value, T>>
  static void GetCrossStreamAddressInfo(size_t op_stream_id,
                                        std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
                                        const T &input) {
    if constexpr (std::is_base_of_v<T, tensor::BaseTensorPtr> || std::is_same_v<T, mindspore::kernel::KernelTensor *> ||
                  std::is_same_v<T, device::DeviceAddressPtr>) {
      GetCrossStreamAddressInfoFromInput(op_stream_id, cross_stream_addresses, input);
    }
  }
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_
