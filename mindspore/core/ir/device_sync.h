/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DEVICE_SYNC_H_
#define MINDSPORE_CORE_IR_DEVICE_SYNC_H_

#include <vector>
#include <memory>
#include <string>

#include "ir/dtype/type.h"
#include "utils/shape_utils.h"

using std::string;

namespace mindspore {
// Interface for data synchornize between device and host.
class DeviceSync {
 public:
  virtual bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const = 0;
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const = 0;
  virtual void *GetMutablePtr() const = 0;
  virtual void ClearDeviceMemory() = 0;
};
using DeviceSyncPtr = std::shared_ptr<DeviceSync>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DEVICE_SYNC_H_
