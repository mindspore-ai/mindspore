/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_UTILS_DEVICE_ADDRESS_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_UTILS_DEVICE_ADDRESS_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "device_address/device_type.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
using KernelTensor = kernel::KernelTensor;
using KernelTensorPtr = kernel::KernelTensorPtr;
// Extract the methods related to DeviceAddress in GraphCompiler to the DeviceAddressUtils class.
class DeviceAddressUtils {
 public:
  static void CreateParameterDeviceAddress(const KernelGraphPtr &graph);
  static std::vector<KernelTensorPtr> CreateKernelTensorForTensorValue(const ValuePtr &node_value, size_t output_idx,
                                                                       const ValueNodePtr &value_node,
                                                                       const KernelGraphPtr &graph);
  static void CreateValueNodeDeviceAddress(const KernelGraphPtr &graph);

  static void CreateDeviceAddressByMapTensorNode(const AnfNodePtr &node, size_t index);

  static KernelTensorPtr CloneEmptyKernelTensor(const KernelTensorPtr &old_kernel_tensor);

  static bool IsContiguousTensor(const tensor::TensorPtr &tensor);

 private:
  // Whether device address of anf node is valid and device address type
  // is consistent with device type, for example, device address type
  // DeviceType::kGPU should be used on GPU device
  static bool NodeDeviceAddressExist(const device::DeviceType &node_device_type, const AnfNodePtr &node, size_t index);
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_UTILS_DEVICE_ADDRESS_UTILS_H_
