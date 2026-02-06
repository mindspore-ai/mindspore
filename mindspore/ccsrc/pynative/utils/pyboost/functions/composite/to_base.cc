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

#include "include/pynative/utils/pyboost/functions/composite/to_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
device::DeviceType GetDeviceType(Device device) {
  static std::map<Device, device::DeviceType> device_type_map = {{DEVICE_ASCEND, device::DeviceType::kAscend},
                                                                 {DEVICE_NPU_LOWER, device::DeviceType::kAscend},
                                                                 {DEVICE_CPU, device::DeviceType::kCPU},
                                                                 {DEVICE_CPU_LOWER, device::DeviceType::kCPU}};
  auto iter = device_type_map.find(device);
  if (iter == device_type_map.end()) {
    MS_LOG(EXCEPTION) << "Not support to device for " << device;
  }
  return iter->second;
}

Int64ImmPtr GetDeviceByDeviceType(device::DeviceType device_type) {
  static std::map<device::DeviceType, Int64ImmPtr> device_map = {
    {device::DeviceType::kAscend, MakeValue<int64_t>(static_cast<int64_t>(Device::DEVICE_ASCEND))->cast<Int64ImmPtr>()},
    {device::DeviceType::kCPU, MakeValue<int64_t>(static_cast<int64_t>(Device::DEVICE_CPU))->cast<Int64ImmPtr>()}};
  auto iter = device_map.find(device_type);
  if (iter == device_map.end()) {
    MS_LOG(EXCEPTION) << "Not support to device for " << device_type;
  }
  return iter->second;
}
ToProcessor &ToProcessor::Contiguous() {
  if (!tensor_->is_contiguous()) {
    tensor_ = pyboost::contiguous(tensor_);
  }
  return *this;
}

ToProcessor &ToProcessor::Cast() {
  if (tensor_->data_type() != dtype_) {
    auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
    tensor_ = cast(tensor_, dtype_ptr);
  }
  return *this;
}

ToProcessor &ToProcessor::Device() {
  if (tensor_->device_address()->GetDeviceType() != device_type_) {
    auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
    bool pin_memory = non_blocking_ && device_type_ == device::DeviceType::kCPU;
    auto new_output = pyboost::empty_like(tensor_, dtype_ptr, GetDeviceByDeviceType(device_type_),
                                          std::make_shared<BoolImm>(pin_memory));
    auto non_blocking_ptr = MakeValue<bool>(static_cast<int64_t>(non_blocking_))->cast<BoolImmPtr>();
    pyboost::inplace_copy(new_output, tensor_, non_blocking_ptr);
    tensor_ = new_output;
  }
  return *this;
}

ToProcessor &ToProcessor::Copy() {
  if (origin_ == tensor_.get() && copy_) {
    auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
    bool pin_memory = non_blocking_ && device_type_ == device::DeviceType::kCPU;
    auto new_output = pyboost::empty_like(tensor_, dtype_ptr, GetDeviceByDeviceType(device_type_),
                                          std::make_shared<BoolImm>(pin_memory));
    auto non_blocking_ptr = MakeValue<bool>(static_cast<int64_t>(non_blocking_))->cast<BoolImmPtr>();
    pyboost::inplace_copy(new_output, tensor_, non_blocking_ptr);
    tensor_ = new_output;
  }
  return *this;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
