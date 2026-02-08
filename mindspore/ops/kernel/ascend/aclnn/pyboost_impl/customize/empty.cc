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

#include "kernel/ascend/aclnn/pyboost_impl/customize/empty.h"
#include <memory>
#include <string>
#include <vector>
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
ShapeVector GetShape(const ValueTuplePtr &shape) {
  ShapeVector output_shape;
  output_shape.reserve(shape->size());
  for (size_t i = 0; i < shape->size(); ++i) {
    int64_t shape_i = std::static_pointer_cast<Int64Imm>((*shape)[i])->value();
    output_shape.push_back(shape_i);
  }
  return output_shape;
}

TypeId GetDataType(const std::optional<Int64ImmPtr> &dtype) {
  // default type: float32
  constexpr TypeId kDefaultDataType = kNumberTypeFloat32;
  if (!dtype.has_value()) {
    return kDefaultDataType;
  }

  auto data_type = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));
  MS_LOG(DEBUG) << "dtype is not None, output tensor's dtype will be set to " << TypeIdToString(data_type);
  return data_type;
}

device::DeviceType GetEmptyDeviceName(const std::optional<Int64ImmPtr> &device) {
  constexpr device::DeviceType kDefaultDevice = device::DeviceType::kAscend;

  if (!device.has_value()) {
    MS_LOG(DEBUG) << "Using default device: " << device::GetDeviceNameByType(kDefaultDevice);
    return kDefaultDevice;
  }

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

  return kDefaultDevice;
}  // namespace
}  // namespace

tensor::TensorPtr EmptyAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &size,
                                       const std::optional<Int64ImmPtr> &dtype,
                                       const std::optional<Int64ImmPtr> &device, const BoolImmPtr &pin_memory) {
  MS_LOG(DEBUG) << "Call EmptyAscendCustomize start";
  ShapeVector output_shape = GetShape(size);
  TypeId data_type = GetDataType(dtype);
  auto device_type = GetEmptyDeviceName(device);

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_type);
  MS_EXCEPTION_IF_NULL(device_ctx);

  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(data_type, output_shape, &outputs);
  PyBoostUtils::PrepareOpOutputs(device_ctx, op->stream_id(), outputs);
  if (pin_memory->value()) {
    if (device_type != device::DeviceType::kCPU) {
      MS_LOG(EXCEPTION) << "Only CPU tensor can be pinned. device should be CPU.";
    }
    auto ascend_device_ctx = runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend);
    if (ascend_device_ctx == nullptr || ascend_device_ctx->device_res_manager_ == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find Ascend device context. ascend_device_ctx or device_res_manager is null.";
    }
    auto pin_memory_allocator = ascend_device_ctx->device_res_manager_->pin_mem_allocator();
    for (auto &tensor : outputs) {
      auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
      device_address->set_allocator(pin_memory_allocator);
    }
  }
  op->set_outputs(outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, device_ctx]() {
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_ctx, outputs);
    MS_LOG(DEBUG) << "Run device task Empty end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
