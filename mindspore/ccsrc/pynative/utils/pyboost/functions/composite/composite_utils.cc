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

#include "include/pynative/utils/pyboost/functions/composite/composite_utils.h"
#include <string>
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

device::DeviceType GetDeviceName(const tensor::TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &device) {
  if (device.has_value()) {
    auto device_name_enum = GetValue<int64_t>(device.value());
    switch (device_name_enum) {
      case DEVICE_ASCEND:
      case DEVICE_NPU_LOWER:
        return device::DeviceType::kAscend;
      case DEVICE_CPU:
      case DEVICE_CPU_LOWER:
        return device::DeviceType::kCPU;
      default:
        MS_LOG(EXCEPTION) << "Only support ['CPU', 'Ascend', 'cpu', 'npu'] for device";
    }
  }

  device::DeviceType device_type =
    (input_tensor != nullptr) ? input_tensor->device_type() : OpRunStatus::Get().device_target();
  MS_LOG(DEBUG) << "Using device: " << device::GetDeviceNameByType(device_type);
  return device_type;
}

void HandlePinMemory(const std::vector<tensor::TensorPtr> &outputs, device::DeviceType device_type) {
  if (device_type != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Only CPU tensor can be pinned. device should be CPU.";
  }
  auto ascend_device_ctx = runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend);
  if (ascend_device_ctx == nullptr || ascend_device_ctx->device_res_manager_ == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find Ascend device context. ascend_device_ctx or device_res_manager is null.";
  }
  auto pin_memory_allocator = ascend_device_ctx->device_res_manager_->pin_mem_allocator();
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &tensor = outputs[i];
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    device_address->set_allocator(pin_memory_allocator);
  }
}

ShapeVector GetShape(const ValueTuplePtr &shape) {
  ShapeVector output_shape;
  output_shape.reserve(shape->size());
  for (size_t i = 0; i < shape->size(); ++i) {
    int64_t shape_i = std::static_pointer_cast<Int64Imm>((*shape)[i])->value();
    output_shape.push_back(shape_i);
  }
  return output_shape;
}

TypeId GetDataType(const tensor::TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &dtype) {
  constexpr TypeId kDefaultDataType = kNumberTypeFloat32;

  if (dtype.has_value()) {
    TypeId data_type = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));
    MS_LOG(DEBUG) << "Using specified dtype: " << TypeIdToString(data_type);
    return data_type;
  }

  if (input_tensor != nullptr) {
    TypeId data_type = static_cast<TypeId>(input_tensor->data_type_c());
    MS_LOG(DEBUG) << "Using input tensor dtype: " << TypeIdToString(data_type);
    return data_type;
  }

  MS_LOG(DEBUG) << "Using default dtype: " << TypeIdToString(kDefaultDataType);
  return kDefaultDataType;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
