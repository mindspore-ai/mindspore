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

#include "device_address/device_type.h"
#include "device_address/device_address.h"
#include "device_address/device_address_maker.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
DeviceAddressPtr MakeGPUDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                      DeviceAddressDeleter &&deleter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_context =
    DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device::DeviceType::kGPU, device_id});
  device_context->Initialize();

  auto device_address = std::make_shared<device::DeviceAddress>(data_ptr, data_size, device::DeviceType::kGPU, 0);
  if (deleter != nullptr) {
    device_address->SetDevicePointerDeleter(std::move(deleter));
  }
  return device_address;
}

REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kGPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeGPUDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
