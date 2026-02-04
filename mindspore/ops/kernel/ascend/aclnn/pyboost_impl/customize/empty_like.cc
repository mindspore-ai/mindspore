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

#include "kernel/ascend/aclnn/pyboost_impl/customize/empty_like.h"
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
TypeId GetDataType(const TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &dtype) {
  TypeId data_type;
  if (dtype.has_value()) {
    data_type = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));
    MS_LOG(DEBUG) << "dtype is not None, output tensor's dtype will be set to " << TypeIdToString(data_type);
  } else {
    data_type = static_cast<TypeId>(input_tensor->data_type_c());
    MS_LOG(DEBUG) << "dtype is None, output tensor's dtype will be set to " << TypeIdToString(data_type);
  }
  return data_type;
}

device::DeviceType GetEmptyLikeDeviceName(const std::optional<Int64ImmPtr> &device) {
  device::DeviceType device_type = device::DeviceType::kAscend;
  if (device.has_value()) {
    auto device_name_enum = GetValue<int64_t>(device.value());
    if (device_name_enum == DEVICE_ASCEND || device_name_enum == DEVICE_NPU_LOWER) {
      device_type = device::DeviceType::kAscend;
    } else if (device_name_enum == DEVICE_CPU || device_name_enum == DEVICE_CPU_LOWER) {
      device_type = device::DeviceType::kCPU;
    } else {
      MS_LOG(EXCEPTION) << "Only support ['CPU', 'Ascend', 'cpu', 'npu'] for device";
    }
  }
  MS_LOG(DEBUG) << "Using '" << device::GetDeviceNameByType(device_type) << "' as the device";
  return device_type;
}
}  // namespace

tensor::TensorPtr EmptyLikeAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                           const std::optional<Int64ImmPtr> &dtype,
                                           const std::optional<Int64ImmPtr> &device, const BoolImmPtr &pin_memory) {
  MS_LOG(DEBUG) << "Call EmptyLikeAscendCustomize start";
  TypeId data_type = GetDataType(input_tensor, dtype);
  auto device_type = GetEmptyLikeDeviceName(device);

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_type);
  MS_EXCEPTION_IF_NULL(device_ctx);

  auto output_shape = input_tensor->shape();
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
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto &tensor = outputs[i];
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
    MS_LOG(DEBUG) << "Run device task EmptyLike end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
