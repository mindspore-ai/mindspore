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
#ifndef MINDSPORE_LITE_EXTENDRT_INFER_DEVICE_ADDRESS_H_
#define MINDSPORE_LITE_EXTENDRT_INFER_DEVICE_ADDRESS_H_

#include <string>

#include "include/backend/device_address.h"

using DeviceAddress = mindspore::device::DeviceAddress;

namespace mindspore {
class InferDeviceAddress : public DeviceAddress {
 public:
  InferDeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}
  ~InferDeviceAddress() override { ClearDeviceMemory(); }

  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override;
  void ClearDeviceMemory() override;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_INFER_DEVICE_ADDRESS_H_
