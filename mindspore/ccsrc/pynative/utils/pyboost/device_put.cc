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

#include "include/pynative/utils/pyboost/device_put.h"

#include "include/backend/common/device_address_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/tensor_py.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "ops_utils/memory_overlap.h"

namespace mindspore {
namespace pyboost {
namespace {
bool IsRemoteParameterTensor(const TensorPtr &tensor) {
  if ((common::GetEnv("MS_DEV_HIERARCHICAL_MEMORY") != "1")) {
    return false;
  }
  constexpr auto kParameterDeviceUserDataName = "parameter_device";
  const auto &user_data = tensor->user_data<tensor::TensorPyUserData>(kParameterDeviceUserDataName);
  if (user_data == nullptr || !py::isinstance<py::str>(user_data->obj)) {
    return false;
  }
  const auto &device_str = py::cast<std::string>(user_data->obj);
  return device_str == kToRemote;
}
}  // namespace

void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                              const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(device_context);
  if (tensor == nullptr) {
    return;
  }

  auto addr = tensor->device_address();
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "The " << tensor->ToString() << " is uninitialized. "
                      << "Maybe the Tensor is create by initializer. "
                      << "You need to call Tensor.init_data before using this Tensor. "
                      << "For more detail with 'Tensor', Please refer to "
                      << "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html";
  }

  runtime::CheckAutoH2D(device_context, tensor);

  auto tensor_address = std::static_pointer_cast<device::DeviceAddress>(addr);
  if (tensor_address->GetDeviceType() == device_context->GetDeviceType()) {
    MS_LOG(DEBUG) << "Already have device address of tensor " << tensor->id();
    return;
  }

  // Not type_cast from python scalar or tuple.
  if (tensor->source_type() == ops::OP_DTYPE::DT_BEGIN) {
    runtime::Pipeline::Get().WaitForward();
  }

  MS_LOG(DEBUG) << "Input tensor device type is " << tensor_address->GetDeviceType()
                << " but current device context is " << device_context->GetDeviceType();

  auto tensor_size = LongToSize(tensor->DataNBytes());
  const auto &format = Format::DEFAULT_FORMAT;
  auto device_address =
    std::make_shared<device::DeviceAddress>(nullptr, tensor_size, device_context->GetDeviceType(), stream_id);
  bool is_remote = IsRemoteParameterTensor(tensor);
  device_address->set_remote(is_remote);
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_from_persistent_mem(tensor->is_parameter());

  // keep origin device_address and execute in another thread.
  tensor->set_implicit_copy_address(addr);
  tensor->set_implicit_copy_storage_info(tensor->storage_info());
  tensor->set_implicit_copy_format(tensor->format());
  tensor->set_device_address(device_address);
  tensor->set_storage_info(nullptr);
  tensor->set_format(format);
  MS_LOG(DEBUG) << "Create input tensor device address " << device_address << " for " << index
                << "th input, Shape: " << tensor->shape() << ", Type: " << TypeIdToType(tensor->data_type())->ToString()
                << ", Size:" << tensor_size;

  mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [device_context, tensor]() { kernel::pyboost::PyBoostUtils::MallocForInput(device_context, tensor); }));
}

void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                              const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  const auto &values = value_tuple->value();
  auto size = values.size();
  std::vector<tensor::TensorPtr> tensors;
  for (size_t i = 0; i < size; ++i) {
    const auto &value = values[i];
    if (value != nullptr && value->isa<tensor::Tensor>()) {
      tensors.push_back(GetValue<tensor::TensorPtr>(value));
    }
  }
  CreateInputTensorAddress(device_context, stream_id, index, tensors);
}

void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                              const std::optional<tensor::TensorPtr> &val) {
  if (!val.has_value()) {
    return;
  }
  CreateInputTensorAddress(device_context, stream_id, index, val.value());
}

void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                               const std::vector<tensor::TensorPtr> &outputs) {
  // Delegate address creation to DeviceAddressUtils.
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, stream_id, outputs);

  // Collect tensors that need malloc (newly-created addresses without device ptr).
  std::vector<TensorPtr> to_malloc_tensors;
  to_malloc_tensors.reserve(outputs.size());
  for (const auto &tensor : outputs) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto addr = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (addr != nullptr && addr->GetPtr() == nullptr) {
      to_malloc_tensors.emplace_back(tensor);
    }
  }

  // Dispatch malloc to device thread.
  if (!to_malloc_tensors.empty()) {
    mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([device_context, to_malloc_tensors]() {
        runtime::DeviceAddressUtils::MallocForOutputs(device_context, to_malloc_tensors);
      }));
  }
}

void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                               const tensor::TensorPtr &output_tensor, size_t size) {
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, stream_id, output_tensor, size);
}

void CheckMemoryOverlap(const std::vector<tensor::TensorPtr> &inputs, const std::vector<tensor::TensorPtr> &outputs) {
  mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([inputs, outputs]() {
      for (const auto &output : outputs) {
        for (const auto &input : inputs) {
          ThrowExpectionWhenPartialOverlap(output, input);
        }
      }
    }));
}
}  // namespace pyboost
}  // namespace mindspore
