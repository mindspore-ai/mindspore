/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MBUF_DEVICE_ADDRESS_H
#define MINDSPORE_MBUF_DEVICE_ADDRESS_H

#include <string>
#include "include/backend/device_address.h"

namespace mindspore {
namespace device {
class MbufDeviceAddress : public device::DeviceAddress {
 public:
  MbufDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  void SetData(void *data) { set_ptr(data); }

  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override {
    MS_LOG(ERROR) << "Mbuf address does not support sync data from device to host, please use graph mode";
    return false;
  }
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override {
    MS_LOG(ERROR) << "Mbuf address does not support sync data from host to device, please use graph mode";
    return false;
  }
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const override {
    MS_LOG(ERROR) << "Mbuf address does not support sync data from device to host, please use graph mode";
    return false;
  }
  void ClearDeviceMemory() override {}
  device::DeviceType GetDeviceType() const { return DeviceType::kAscend; }
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MBUF_DEVICE_ADDRESS_H
