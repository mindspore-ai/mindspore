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
#ifndef MINDSPORE_CCSRC_FRONTEND_IR_MBUF_DEVICE_ADDRESS_H
#define MINDSPORE_CCSRC_FRONTEND_IR_MBUF_DEVICE_ADDRESS_H

#include <string>
#include <memory>
#include "ir/dtype/tensor_type.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace device {
class MbufDeviceAddress : public device::DeviceAddress {
 public:
  MbufDeviceAddress(void *ptr, size_t size, const ShapeVector &shape, TypeId type, const std::string &device_name)
      : DeviceAddress(ptr, size, device::GetDeviceTypeByName(device_name), 0) {
    auto tensor_shape = std::make_shared<abstract::TensorShape>();
    tensor_shape->SetShapeVector(shape);
    auto tensor_type = std::make_shared<TensorType>(TypeIdToType(type));
  }
  void SetData(void *data) { set_ptr(data); }
  device::DeviceType GetDeviceType() const { return DeviceType::kAscend; }
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_IR_MBUF_DEVICE_ADDRESS_H
