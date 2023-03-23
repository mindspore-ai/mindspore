/**
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include "include/backend/visible.h"
#include "include/backend/device_address.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace device {
namespace cpu {
class BACKEND_EXPORT CPUDeviceAddress : public DeviceAddress {
 public:
  CPUDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}

  CPUDeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}

  CPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const KernelWithIndex &node_index)
      : DeviceAddress(ptr, size, format, type_id, node_index) {}

  CPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                   uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}

  ~CPUDeviceAddress() override;

  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override;
  bool SyncDeviceToDevice(const DeviceSync *src_device_addr) const override;
  bool SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                          const std::string &format) const override;

  bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                     TypeId host_type, bool trans_flag) const override;
  void ClearDeviceMemory() override;
  void ClearUserData() override;

  DeviceType GetDeviceType() const override { return DeviceType::kCPU; }
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_DEVICE_ADDRESS_H_
