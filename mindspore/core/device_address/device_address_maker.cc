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

#include "device_address/device_address_maker.h"
#include "device_address/device_address.h"
#include "ir/tensor_data.h"

namespace mindspore {
namespace {
constexpr int kMaxDeviceNum = 6;
DeviceAddressMakerFunc g_device_address_maker[kMaxDeviceNum];

DeviceAddressPtr MakeCPUDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                      DeviceAddressDeleter &&deleter) {
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_address = std::make_shared<DeviceAddress>(data_ptr, data_size, device::DeviceType::kCPU, 0);
  if (deleter != nullptr) {
    device_address->SetDevicePointerDeleter(std::move(deleter));
  }
  return device_address;
}
}  // namespace
DeviceAddressPtr DeviceAddressMaker::make_device_address() {
  auto device_sync = maker_(data_type_, shape_, data_ptr_, std::move(deleter_));
  return device_sync;
}

DeviceAddressMaker &DeviceAddressMaker::set_deleter(std::function<void(void *, bool)> &&deleter) {
  deleter_ = std::move(deleter);
  return *this;
}

DeviceAddressMaker &DeviceAddressMaker::set_maker(DeviceAddressMakerFunc &&maker) {
  maker_ = std::move(maker);
  return *this;
}

void SetDeviceAddressMaker(device::DeviceType device_type, DeviceAddressMakerFunc &&func) {
  g_device_address_maker[static_cast<int>(device_type)] = func;
}

DeviceAddressMakerFunc GetDeviceAddressMaker(device::DeviceType device_target) {
  const auto &maker = g_device_address_maker[static_cast<int>(device_target)];
  MS_EXCEPTION_IF_NULL(maker);
  return maker;
}

DeviceAddressPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, bool init,
                                   device::DeviceType device_type) {
  // todo: set allocator
  if (device_type == device::DeviceType::kCPU) {
    auto tensor_data = tensor::MakeTensorData(data_type, shape);
    // todo: use init after tensor::empty is replaced.
    auto ret = DeviceAddressMaker(init ? tensor_data->data() : nullptr, data_type, shape)
                 .set_deleter([tensor_data](void *, bool) {})
                 .set_maker(GetDeviceAddressMaker(device_type))
                 .make_device_address();
    ret->set_data(std::move(tensor_data));
    return ret;
  }

  auto ret =
    DeviceAddressMaker(nullptr, data_type, shape).set_maker(GetDeviceAddressMaker(device_type)).make_device_address();
  return ret;
}

DeviceAddressPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, tensor::TensorDataPtr &&tensor_data,
                                   device::DeviceType device_type) {
  // Just GET data ptr of tensor_data and don't init the data.
  // todo: use const_data() after tensor::empty is replaced.
  auto ret = DeviceAddressMaker(tensor_data->data(), data_type, shape)
               .set_deleter([tensor_data](void *, bool) {})
               .set_maker(GetDeviceAddressMaker(device_type))
               .make_device_address();
  ret->set_data(std::move(tensor_data));
  return ret;
}

DeviceAddressPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, void *device_data_ptr,
                                   size_t device_offset, device::DeviceType device_type) {
  // Create device address with offset and don't init the data.
  auto ret = DeviceAddressMaker(AddressOffset(device_data_ptr, device_offset), data_type, shape)
               .set_deleter([](void *, bool) {})
               .set_maker(GetDeviceAddressMaker(device_type))
               .make_device_address();
  return ret;
}

template DeviceAddressPtr MakeDeviceAddress<int64_t>(TypeId, int64_t);
template DeviceAddressPtr MakeDeviceAddress<int32_t>(TypeId, int32_t);
template DeviceAddressPtr MakeDeviceAddress<int16_t>(TypeId, int16_t);
template DeviceAddressPtr MakeDeviceAddress<int8_t>(TypeId, int8_t);
template DeviceAddressPtr MakeDeviceAddress<double>(TypeId, double);
template DeviceAddressPtr MakeDeviceAddress<float>(TypeId, float);
template DeviceAddressPtr MakeDeviceAddress<float16>(TypeId, float16);
template DeviceAddressPtr MakeDeviceAddress<float8_e5m2>(TypeId, float8_e5m2);
template DeviceAddressPtr MakeDeviceAddress<float8_e4m3fn>(TypeId, float8_e4m3fn);
template DeviceAddressPtr MakeDeviceAddress<hifloat8>(TypeId, hifloat8);
template DeviceAddressPtr MakeDeviceAddress<bfloat16>(TypeId, bfloat16);
template DeviceAddressPtr MakeDeviceAddress<uint64_t>(TypeId, uint64_t);
template DeviceAddressPtr MakeDeviceAddress<uint32_t>(TypeId, uint32_t);
template DeviceAddressPtr MakeDeviceAddress<uint16_t>(TypeId, uint16_t);
template DeviceAddressPtr MakeDeviceAddress<uint8_t>(TypeId, uint8_t);
template DeviceAddressPtr MakeDeviceAddress<bool>(TypeId, bool);
template DeviceAddressPtr MakeDeviceAddress<int64_t>(TypeId, const ShapeVector &, const std::vector<int64_t> &);
template DeviceAddressPtr MakeDeviceAddress<int32_t>(TypeId, const ShapeVector &, const std::vector<int32_t> &);
template DeviceAddressPtr MakeDeviceAddress<double>(TypeId, const ShapeVector &, const std::vector<double> &);
template DeviceAddressPtr MakeDeviceAddress<float>(TypeId, const ShapeVector &, const std::vector<float> &);

REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kCPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeCPUDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace mindspore
