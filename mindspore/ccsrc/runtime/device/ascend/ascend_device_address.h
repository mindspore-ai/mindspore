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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include <memory>
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
class LaunchKernel;
namespace ascend {
class AscendDeviceAddress : public DeviceAddress {
 public:
  explicit AscendDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  explicit AscendDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}
  explicit AscendDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                               const KernelWithIndex &node_index)
      : DeviceAddress(ptr, size, format, type_id, node_index) {}
  ~AscendDeviceAddress() override;
  bool SyncDeviceToHost(size_t size, void *const host_ptr) const override;
  bool SyncHostToDevice(size_t size, const void *host_ptr) const override;
  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format = "DefaultFormat") const override;
  bool SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                          const std::string &format) const override;
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
  bool SyncDeviceToHostAndConvertFormatBasedOnTransData(const std::vector<size_t> &host_shape, size_t size,
                                                        mindspore::TypeId type, void *host_ptr) const;
  void SyncStream() const;
  std::vector<size_t> GetDeviceShape(std::vector<size_t> *host_shape) const;
  std::shared_ptr<LaunchKernel> CreateLaunchTransData(const std::vector<size_t> &host_shape,
                                                      const std::string &ori_format,
                                                      const std::string &dst_format) const;
  mutable std::shared_ptr<LaunchKernel> launch_transdata_{nullptr};
};
using AscendDeviceAddressPtr = std::shared_ptr<AscendDeviceAddress>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DEVICE_ADDRESS_H_
