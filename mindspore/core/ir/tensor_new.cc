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

#include "ir/tensor_new.h"

#include "ir/tensor.h"
#include "device_address/device_address_maker.h"

namespace mindspore {
namespace tensor {
TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

TensorPtr from_spec(TypeId data_type, const ShapeVector &shape, device::DeviceType device_type) {
  if (device_type == device::DeviceType::kNone) {
    auto ret = std::make_shared<Tensor>(data_type, shape, nullptr);
    MS_LOG(DEBUG) << "Make none tensor " << ret->ToString();
    return ret;
  }
  return std::make_shared<Tensor>(data_type, shape, MakeDeviceAddress(data_type, shape, true, device_type));
}

TensorPtr from_spec_fast(TypeId data_type, const ShapeVector &shape, device::DeviceType device_type) {
  if (device_type == device::DeviceType::kCPU) {
    auto nbytes = abstract::TypeIdSize(data_type) * SizeOf(shape);
    // Allocate memory without initializing.
    auto buffer = std::malloc(nbytes);
    auto device_address = DeviceAddressMaker(buffer, data_type, shape)
                            .set_deleter([buffer](void *, bool) { std::free(buffer); })
                            .set_maker(GetDeviceAddressMaker(device_type))
                            .make_device_address();
    return std::make_shared<Tensor>(data_type, shape, device_address);
  }
  return from_spec(data_type, shape, device_type);
}

TensorPtr from_buffer(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len) {
  return std::make_shared<Tensor>(
    data_type, shape, MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, data, data_len)));
}

TensorPtr from_buffer(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) {
  return std::make_shared<Tensor>(
    data_type, shape, MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type)));
}

template TensorPtr from_scalar(int64_t input, const TypePtr &data_type);
template TensorPtr from_scalar(int32_t input, const TypePtr &data_type);
template TensorPtr from_scalar(int16_t input, const TypePtr &data_type);
template TensorPtr from_scalar(int8_t input, const TypePtr &data_type);
template TensorPtr from_scalar(double input, const TypePtr &data_type);
template TensorPtr from_scalar(float input, const TypePtr &data_type);
template TensorPtr from_scalar(float16 input, const TypePtr &data_type);
template TensorPtr from_scalar(float8_e5m2 input, const TypePtr &data_type);
template TensorPtr from_scalar(float8_e4m3fn input, const TypePtr &data_type);
template TensorPtr from_scalar(hifloat8 input, const TypePtr &data_type);
template TensorPtr from_scalar(bfloat16 input, const TypePtr &data_type);
template TensorPtr from_scalar(uint64_t input, const TypePtr &data_type);
template TensorPtr from_scalar(uint32_t input, const TypePtr &data_type);
template TensorPtr from_scalar(uint16_t input, const TypePtr &data_type);
template TensorPtr from_scalar(uint8_t input, const TypePtr &data_type);
template TensorPtr from_scalar(bool input, const TypePtr &data_type);

template TensorPtr from_vector(const std::vector<int64_t> &input, const TypePtr &data_type);
template TensorPtr from_vector(const std::vector<int32_t> &input, const TypePtr &data_type);
template TensorPtr from_vector(const std::vector<double> &input, const TypePtr &data_type);
template TensorPtr from_vector(const std::vector<float> &input, const TypePtr &data_type);
}  // namespace tensor
}  // namespace mindspore
