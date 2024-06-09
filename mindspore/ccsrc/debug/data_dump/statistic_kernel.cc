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
#include "debug/data_dump/statistic_kernel.h"
#include <memory>
#include <string>
#include <vector>
#include "debug/debugger/debugger_utils.h"
#include "include/common/debug/common.h"

namespace mindspore {

namespace {
using TensorPtr = tensor::TensorPtr;
const std::set<TypeId> max_supported_dtype{
  kNumberTypeBFloat16, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat,
  kNumberTypeDouble,   kNumberTypeInt,     kNumberTypeInt8,    kNumberTypeUInt8,   kNumberTypeInt16,
  kNumberTypeInt32,    kNumberTypeInt64,   kNumberTypeBool};
const std::set<TypeId> &min_supported_dtype = max_supported_dtype;
const std::set<TypeId> mean_supported_dtype = {
  kNumberTypeBFloat16, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat, kNumberTypeDouble,
  kNumberTypeInt,      kNumberTypeInt8,    kNumberTypeUInt8,   kNumberTypeInt16,   kNumberTypeInt32, kNumberTypeInt64};

void WarningOnce(const string &device_name, const string &type_name, const string &statistic_name) {
  static std::set<string> warning_once;
  string name = device_name + type_name + statistic_name;
  if (warning_once.find(name) != warning_once.end()) {
    return;
  } else {
    warning_once.insert(name);
    MS_LOG(WARNING) << "In the '" << device_name << "' platform, '" << type_name << "' is not supported for '"
                    << statistic_name << "' statistic dump.";
  }
}

}  // namespace

namespace datadump {

DeviceAddressPtr StatisticKernel::GenerateDeviceAddress(const uint32_t &stream_id, const size_t &mem_size,
                                                        const TypeId &dtype_id, const ShapeVector &shape,
                                                        const ValuePtr &value) {
  auto addr = device_context_->device_res_manager_->AllocateMemory(mem_size, stream_id);
  MS_EXCEPTION_IF_NULL(addr);

  auto tensor = std::make_shared<kernel::KernelTensor>(addr, mem_size, Format::DEFAULT_FORMAT, dtype_id, shape,
                                                       device_context_->device_context_key().device_name_,
                                                       device_context_->device_context_key().device_id_);
  tensor->set_stream_id(stream_id);
  tensor->SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape));
  if (value) {
    tensor->SetValue(value);
  }
  return device_context_->device_res_manager_->CreateDeviceAddress(tensor);
}

TensorPtr StatisticKernel::SyncDeviceToHostTensor(DeviceAddressPtr device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto kernel_tensor = device_addr->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto dtype_id = kernel_tensor->dtype_id();
  const auto &shape_vec = kernel_tensor->GetShapeVector();

  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(dtype_id, shape_vec);
  auto ret_sync = device_addr->SyncDeviceToHost(UnitSizeInBytes(dtype_id), out_tensor->data_c());
  if (!ret_sync) {
    MS_LOG(EXCEPTION) << "Convert format or Copy device mem to host failed";
  }
  return out_tensor;
}

DeviceAddressPtr StatisticKernel::GetWorkSpaceDeviceAddress(const uint32_t stream_id,
                                                            const vector<KernelTensor *> &inputs,
                                                            const vector<KernelTensor *> &outputs) {
  auto ret = kernel_mod_->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call Resize error, error id is " << ret;
  }
  auto work_space = kernel_mod_->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    return runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, stream_id, work_space[0]);
  }
  return nullptr;
}

DeviceAddressPtr StatisticKernel::GetOutputDeviceAddress(const uint32_t stream_id, TypeId dtype_id) {
  ShapeVector shape_vec = {};
  return GenerateDeviceAddress(stream_id, UnitSizeInBytes(dtype_id), dtype_id, shape_vec);
}

TensorPtr StatisticKernel::LaunchKernel(KernelTensor *input) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->GetShapeVector().empty()) {
    return std::make_shared<tensor::Tensor>(input->dtype_id(), input->GetShapeVector(),
                                            const_cast<void *>(input->GetValuePtr()),
                                            UnitSizeInBytes(input->dtype_id()));
  }
  vector<KernelTensor *> inputs{input};
  const auto stream_id = input->stream_id();
  auto output_addr = GetOutputDeviceAddress(stream_id, input->dtype_id());
  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto workspace_addr = GetWorkSpaceDeviceAddress(stream_id, {input}, {output_addr->kernel_tensor().get()});
  bool ret = false;
  if (workspace_addr) {
    ret = kernel_mod_->Launch(inputs, {workspace_addr->kernel_tensor().get()}, {output_addr->kernel_tensor().get()},
                              stream_ptr);
  } else {
    ret = kernel_mod_->Launch(inputs, {}, {output_addr->kernel_tensor().get()}, stream_ptr);
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << "Launch error";
  }
  return SyncDeviceToHostTensor(output_addr);
}

DeviceAddressPtr MeanStatisticKernel::GetAxisDeviceAddress(const uint32_t stream_id, size_t dim) {
  vector<int64_t> axes(dim);
  for (size_t i = 0; i < dim; i++) {
    axes[i] = static_cast<int64_t>(i);
  }
  ShapeVector axes_shape{static_cast<int64_t>(dim)};
  size_t axisbytes = UnitSizeInBytes(kNumberTypeInt64) * dim;
  return GenerateDeviceAddress(stream_id, axisbytes, kNumberTypeInt64, axes_shape, MakeValue(axes));
}

DeviceAddressPtr MeanStatisticKernel::GetKeepDimsDeviceAddress(const uint32_t stream_id) {
  ShapeVector keepdims_shape = {};
  return GenerateDeviceAddress(stream_id, UnitSizeInBytes(kNumberTypeBool), kNumberTypeBool, keepdims_shape,
                               MakeValue(false));
}

DeviceAddressPtr MeanStatisticKernel::GetDtypeDeviceAddress(const uint32_t stream_id, const TypeId &dtype_id) {
  ShapeVector dtype_shape_vec = {1};
  return GenerateDeviceAddress(stream_id, UnitSizeInBytes(dtype_id), dtype_id, dtype_shape_vec);
}

TensorPtr MeanStatisticKernel::LaunchKernel(KernelTensor *input) {
  MS_EXCEPTION_IF_NULL(input);
  const auto stream_id = input->stream_id();
  vector<KernelTensor *> inputs{input};
  auto output_addr = GetOutputDeviceAddress(stream_id, input->dtype_id());
  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(kernel_mod_);

  auto axis = GetAxisDeviceAddress(stream_id, input->GetShapeVector().size());
  MS_EXCEPTION_IF_NULL(axis);
  inputs.emplace_back(axis->kernel_tensor().get());

  auto keepdims = GetKeepDimsDeviceAddress(stream_id);
  inputs.emplace_back(keepdims->kernel_tensor().get());

  auto dtype = GetDtypeDeviceAddress(stream_id, input->dtype_id());
  inputs.emplace_back(dtype->kernel_tensor().get());

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto workspace_addr = GetWorkSpaceDeviceAddress(stream_id, inputs, {output_addr->kernel_tensor().get()});
  bool ret = false;
  if (workspace_addr) {
    ret = kernel_mod_->Launch(inputs, {workspace_addr->kernel_tensor().get()}, {output_addr->kernel_tensor().get()},
                              stream_ptr);
  } else {
    ret = kernel_mod_->Launch(inputs, {}, {output_addr->kernel_tensor().get()}, stream_ptr);
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << kernel_name_ << " kernel launch failed";
  }
  return SyncDeviceToHostTensor(output_addr);
}

TensorPtr StatisticKernelManager::CalMax(const DeviceContext *device_context, KernelTensor *input) {
  if (max_kernel.find(device_context) == max_kernel.end()) {
    max_kernel.emplace(device_context,
                       std::make_unique<StatisticKernel>(device_context, ops::kNameMax, max_supported_dtype));
  }
  auto dtype = input->dtype_id();
  if (max_kernel[device_context]->CheckDataType(dtype)) {
    return max_kernel[device_context]->LaunchKernel(input);
  } else {
    const auto &device_name = device_context->device_context_key_.device_name_;
    const auto &type_name = TypeIdToString(dtype);
    WarningOnce(device_name, type_name, "max");
    return nullptr;
  }
}

TensorPtr StatisticKernelManager::CalMin(const DeviceContext *device_context, KernelTensor *input) {
  if (min_kernel.find(device_context) == min_kernel.end()) {
    min_kernel.emplace(device_context,
                       std::make_unique<StatisticKernel>(device_context, ops::kNameMin, min_supported_dtype));
  }
  auto dtype = input->dtype_id();
  if (min_kernel[device_context]->CheckDataType(dtype)) {
    return min_kernel[device_context]->LaunchKernel(input);
  } else {
    const auto &device_name = device_context->device_context_key_.device_name_;
    const auto &type_name = TypeIdToString(dtype);
    WarningOnce(device_name, type_name, "min");
    return nullptr;
  }
}

TensorPtr StatisticKernelManager::CalMean(const DeviceContext *device_context, KernelTensor *input) {
  if (mean_kernel.find(device_context) == mean_kernel.end()) {
    mean_kernel.emplace(device_context, std::make_unique<MeanStatisticKernel>(device_context, mean_supported_dtype));
  }
  auto dtype = input->dtype_id();
  if (mean_kernel[device_context]->CheckDataType(dtype)) {
    return mean_kernel[device_context]->LaunchKernel(input);
  } else {
    const auto &device_name = device_context->device_context_key_.device_name_;
    const auto &type_name = TypeIdToString(dtype);
    WarningOnce(device_name, type_name, "mean");
    return nullptr;
  }
}

}  // namespace datadump
}  // namespace mindspore
