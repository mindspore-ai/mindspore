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
#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "runtime/device/device_address.h"
#include "runtime/device/kernel_info.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::runtime {
class BACKEND_EXPORT OpRuntimeInfo {
 public:
  OpRuntimeInfo(std::vector<std::string> output_format, std::vector<TypeId> output_type,
                std::vector<size_t> output_tensor_size, device::KernelInfo *kernel_info,
                std::vector<std::pair<device::KernelInfo *, size_t>> input_kernel_infos)
      : output_format_(std::move(output_format)),
        output_type_(std::move(output_type)),
        output_tensor_size_(std::move(output_tensor_size)),
        kernel_info_(kernel_info),
        input_kernel_infos_(std::move(input_kernel_infos)) {}
  ~OpRuntimeInfo() = default;

  // Key for user data.
  constexpr static char key[] = "OpRuntimeInfo";

  std::string output_format(size_t index) const;
  TypeId output_type(size_t index) const;
  size_t output_tensor_size(size_t index) const;
  device::DeviceAddressPtr GetOutputDeviceAddress(size_t index) const;
  device::DeviceAddressPtr GetWorkspaceDeviceAddress(size_t index) const;
  device::DeviceAddressPtr GetInputDeviceAddress(size_t index) const;
  size_t GetInputSize() const;
  size_t GetOutputSize() const;
  size_t GetWorkspaceSize() const;

  static void CacheGraphOpRuntimeInfo(const KernelGraphPtr &graph);

 private:
  std::vector<std::string> output_format_;
  std::vector<TypeId> output_type_;
  std::vector<size_t> output_tensor_size_;
  device::KernelInfo *kernel_info_;
  std::vector<std::pair<device::KernelInfo *, size_t>> input_kernel_infos_;
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_
