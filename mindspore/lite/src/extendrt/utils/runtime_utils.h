/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_RUNTIME_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_RUNTIME_UTILS_H_

#include <vector>
#include <string>

#include "include/backend/device_address.h"
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"
#include "ir/tensor.h"

namespace mindspore {
class RuntimeUtils {
 public:
  static void *GetAddressPtr(device::DeviceAddressPtr address_ptr);
  static void SetAddressPtr(device::DeviceAddressPtr address_ptr, void *ptr);
  static void AllocAddressPtr(device::DeviceAddressPtr address_ptr);

  static kernel::AddressPtr GetAddressFromDevice(device::DeviceAddressPtr address_ptr);

  static std::vector<AnfNodePtr> GetGraphDataInputs(const KernelGraphPtr &kernel_graph);
  static void CopyInputTensorsToKernelGraph(const std::vector<tensor::Tensor> &inputs, KernelGraphPtr kernel_graph);
  static void CopyOutputTensorsFromKernelGraph(std::vector<tensor::Tensor> *outputs, KernelGraphPtr kernel_graph);

  static void AssignKernelGraphAddress(KernelGraphPtr kernel_graph);
  static void AssignValueNodeAddress(KernelGraphPtr kernel_graph);
  static void AssignInputNodeAddress(KernelGraphPtr kernel_graph);
  static void AssignKernelOutputAddress(KernelGraphPtr kernel_graph);
  static device::DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                      TypeId type_id);
  static void UpdateKernelNodeOutputInfo(const AnfNodePtr &kernel_node,
                                         const std::vector<kernel::AddressPtr> &output_addrs);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_RUNTIME_UTILS_H_
