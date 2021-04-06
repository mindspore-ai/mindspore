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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include "runtime/device/device_address.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "ir/dtype.h"
#include "backend/kernel_compiler/kernel.h"
#include "utils/shape_utils.h"

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
  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const override;
  void ClearDeviceMemory() override;
  DeviceAddressType DeviceType() const override { return DeviceAddressType::kAscend; }
  bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                     TypeId host_type, bool trans_flag) const override;
#ifdef ENABLE_DEBUGGER
  bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                     const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev) const override;
#endif

 private:
  bool SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const;
  bool ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const;
  bool SyncDeviceToHostAndConvertFormatBasedOnTransData(const std::vector<size_t> &host_shape,
                                                        const std::vector<size_t> &device_shape, size_t size,
                                                        mindspore::TypeId type, void *host_ptr) const;
  void SyncStream() const;

  void LaunchTransData(kernel::KernelModPtr kernel_mod_ptr, void *output_address_ptr, size_t output_size,
                       const std::vector<size_t> &workspace_size_list) const;
  std::vector<size_t> GetDeviceShape(std::vector<size_t> *host_shape) const;
  std::vector<size_t> GetWorkspaceSizeList(const nlohmann::json &kernel_json) const;
  kernel::KernelModPtr CompileTransDataAndObtainKernelMod(const nlohmann::json &kernel_json) const;
};
using AscendDeviceAddressPtr = std::shared_ptr<AscendDeviceAddress>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
