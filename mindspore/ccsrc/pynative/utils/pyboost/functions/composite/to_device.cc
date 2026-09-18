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

#include "include/pynative/utils/pyboost/functions/composite/to_device.h"

#include <utility>
#include <map>
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/functions/composite/to_base.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr to_device(const mindspore::tensor::TensorPtr &self,
                            const std::optional<mindspore::Int64ImmPtr> &device,
                            const std::optional<mindspore::Int64ImmPtr> &dtype,
                            const mindspore::BoolImmPtr &non_blocking, const mindspore::BoolImmPtr &copy) {
  device::DeviceType device_value = device.has_value()
                                      ? GetDeviceType(static_cast<Device>(GetValue<int64_t>(device.value())))
                                      : GetDeviceTypeChecked(self);
  TypeId dtype_value = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : self->data_type();
  bool non_blocking_value = GetValue<bool>(non_blocking);
  bool copy_value = GetValue<bool>(copy);

  MS_LOG(DEBUG) << "Start ToDevice, input is " << self->ToString()
                << " device:" << device::GetDeviceNameByType(device_value)
                << " dtype:" << TypeIdToType(dtype_value)->ToString() << " non_blocking:" << non_blocking_value
                << " copy:" << copy_value;

  if (device_value != device::DeviceType::kMeta) {
    CheckTensorInitialized(self);
  }

  if (self->device_address() != nullptr && self->device_address()->GetDeviceType() == device::DeviceType::kMeta &&
      device_value != device::DeviceType::kMeta) {
    MS_LOG(EXCEPTION) << "Cannot copy from tensor which is on meta device, there is no data.";
  }

  if (device_value == device::DeviceType::kMeta) {
    ShapeVector shape = self->shape();
    auto data_size = SizeOf(shape) * abstract::TypeIdSize(dtype_value);
    auto device_address = std::make_shared<DeviceAddress>(nullptr, data_size, device::DeviceType::kMeta, 0);
    auto output = std::make_shared<Tensor>(dtype_value, shape, device_address);
    MS_LOG(DEBUG) << "End ToDevice";
    return output;
  }

  auto output = ToProcessor(self, device_value, dtype_value, non_blocking_value, copy_value)
                  .Contiguous()
                  .Cast()
                  .Device()
                  .Copy()
                  .Get();
  MS_LOG(DEBUG) << "End ToDevice";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
