/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include <memory>
#include "device/device_address.h"
#include "device/ascend/ascend_memory_pool.h"
#include "ir/dtype.h"

namespace mindspore {
#ifdef ENABLE_DEBUGGER
class Debugger;
#endif
namespace device {
namespace ascend {
class AscendDeviceAddress : public DeviceAddress {
 public:
  explicit AscendDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  explicit AscendDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}
  ~AscendDeviceAddress() override;
  bool SyncDeviceToHost(const std::vector<int> &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const std::vector<int> &shape, size_t size, TypeId type, const void *host_ptr) const override;
  DeviceAddressType DeviceType() const override { return DeviceAddressType::kAscend; }
#ifdef ENABLE_DUMP_E2E
  bool DumpMemToFile(bool dump_mode, const std::string &filepath, const std::string &host_fmt,
                     const std::vector<int> &host_shape, TypeId host_type) const;
#endif
#ifdef ENABLE_DEBUGGER
  bool LoadMemToHost(bool dump_mode, const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                     const std::vector<int> &host_shape, TypeId host_type, size_t slot, Debugger *debugger) const;
#endif
 private:
  bool SyncDeviceToHostAndConvertFormat(const std::vector<int> &shape, size_t size, TypeId type, void *host_ptr) const;
  bool ConvertFormatAndSyncHostToDevice(const std::vector<int> &shape, size_t size, TypeId type,
                                        const void *host_ptr) const;
  void SyncStream() const;
};
using AscendDeviceAddressPtr = std::shared_ptr<AscendDeviceAddress>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
