/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/cxx_api/context.h"
#include <string>
#include <memory>
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "src/runtime/inner_allocator.h"
#include "src/common/log_adapter.h"

namespace mindspore {
constexpr auto kModelOptionCpuEnableFP16 = "mindspore.option.cpu.enable_fp16";
constexpr auto kModelOptionGPUEnableFP16 = "mindspore.option.gpu.enable_fp16";
constexpr auto kModelOptionGPUDeviceID = "mindspore.option.gpu.device_id";
constexpr auto kModelOptionKirinNpuFrequency = "mindspore.option.kirin_npu.frequency";
constexpr auto kModelOptionProvider = "mindspore.option.provider";
constexpr auto kModelOptionProviderDevice = "mindspore.option.provider.device";
constexpr auto kModelOptionDeviceID = "mindspore.option.device_id";
constexpr auto kModelOptionAscend310DeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscend310InsertOpCfgPath = "mindspore.option.ascend310.insert_op_config_file_path";
constexpr auto kModelOptionAscend310InputFormat = "mindspore.option.ascend310.input_format";
constexpr auto kModelOptionAscend310InputShapeMap = "mindspore.option.ascend310.input_shape_map";
constexpr auto kModelOptionAscend310InputShape = "mindspore.option.ascend310.input_shape";
constexpr auto kModelOptionAscend310OutputType = "mindspore.option.ascend310.output_type";
constexpr auto kModelOptionAscend310PrecisionMode = "mindspore.option.ascend310.precision_mode";
constexpr auto kModelOptionAscend310OpSelectImplMode = "mindspore.option.ascend310.op_select_impl_mode";
constexpr auto KModelOptionAscend310FusionSwitchCfgPath = "mindspore.option.ascend310.fusion_switch_config_file_path";
constexpr auto kModelOptionAscend310DynamicBatchSize = "mindspore.option.ascend310.dynamic_batch_size";
constexpr auto kModelOptionAscend310BufferOptimize = "mindspore.option.ascend310.buffer_optimize";

Context::Context() : data_(std::shared_ptr<Data>(new (std::nothrow) Data())) {}

template <class T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
static const U &GetValue(const std::shared_ptr<DeviceInfoContext::Data> &data, const std::string &key) {
  static U empty_result;
  if (data == nullptr) {
    return empty_result;
  }
  auto iter = data->params.find(key);
  if (iter == data->params.end()) {
    return empty_result;
  }
#ifndef SUPPORT_NNIE
  const std::any &value = iter->second;
  return std::any_cast<const U &>(value);
#else
  const std::experimental::any &value = iter->second;
  return std::experimental::any_cast<const U &>(value);
#endif
}

void Context::SetThreadNum(int32_t thread_num) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->thread_num = thread_num;
}

int32_t Context::GetThreadNum() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return data_->thread_num;
}

void Context::SetEnableParallel(bool is_parallel) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->enable_parallel_ = is_parallel;
}

bool Context::GetEnableParallel() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }

  return data_->enable_parallel_;
}

void Context::SetThreadAffinity(int mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->affinity_mode_ = mode;
  return;
}

int Context::GetThreadAffinityMode() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return -1;
  }
  return data_->affinity_mode_;
}

void Context::SetThreadAffinity(const std::vector<int> &core_list) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->affinity_core_list_ = core_list;

  return;
}

std::vector<int32_t> Context::GetThreadAffinityCoreList() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return {};
  }
  return data_->affinity_core_list_;
}

void Context::SetDelegate(const std::shared_ptr<Delegate> &delegate) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->delegate = delegate;
}

std::shared_ptr<Delegate> Context::GetDelegate() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return data_->delegate;
}

std::vector<std::shared_ptr<DeviceInfoContext>> &Context::MutableDeviceInfo() {
  static std::vector<std::shared_ptr<DeviceInfoContext>> empty;
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return empty;
  }
  return data_->device_info_list;
}

DeviceInfoContext::DeviceInfoContext() : data_(std::make_shared<Data>()) {}

std::vector<char> DeviceInfoContext::GetProviderChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionProvider);
  return StringToChar(ref);
}

void DeviceInfoContext::SetProvider(const std::vector<char> &provider) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionProvider] = CharToString(provider);
}

std::vector<char> DeviceInfoContext::GetProviderDeviceChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionProviderDevice);
  return StringToChar(ref);
}

void DeviceInfoContext::SetProviderDevice(const std::vector<char> &device) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionProviderDevice] = CharToString(device);
}

void DeviceInfoContext::SetAllocator(const std::shared_ptr<Allocator> &allocator) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->allocator = allocator;
}

std::shared_ptr<Allocator> DeviceInfoContext::GetAllocator() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return data_->allocator;
}

void CPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionCpuEnableFP16] = is_fp16;
}

bool CPUDeviceInfo::GetEnableFP16() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }
  return GetValue<bool>(data_, kModelOptionCpuEnableFP16);
}

void GPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionGPUEnableFP16] = is_fp16;
}
bool GPUDeviceInfo::GetEnableFP16() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }
  return GetValue<bool>(data_, kModelOptionGPUEnableFP16);
}

void KirinNPUDeviceInfo::SetFrequency(int frequency) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionKirinNpuFrequency] = frequency;
}

int KirinNPUDeviceInfo::GetFrequency() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return GetValue<int>(data_, kModelOptionKirinNpuFrequency);
}

void GPUDeviceInfo::SetDeviceID(uint32_t device_id) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionGPUDeviceID] = device_id;
}

uint32_t GPUDeviceInfo::GetDeviceID() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return GetValue<uint32_t>(data_, kModelOptionGPUDeviceID);
}

void GPUDeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> GPUDeviceInfo::GetPrecisionModeChar() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  std::vector<char> ret;
  return ret;
}

void Ascend910DeviceInfo::SetDeviceID(uint32_t device_id) { MS_LOG(ERROR) << "Unsupported Feature."; }
uint32_t Ascend910DeviceInfo::GetDeviceID() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

void Ascend310DeviceInfo::SetDeviceID(uint32_t device_id) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310DeviceID] = device_id;
}

uint32_t Ascend310DeviceInfo::GetDeviceID() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return GetValue<uint32_t>(data_, kModelOptionAscend310DeviceID);
}

void Ascend310DeviceInfo::SetInsertOpConfigPath(const std::vector<char> &cfg_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310InsertOpCfgPath] = CharToString(cfg_path);
}
std::vector<char> Ascend310DeviceInfo::GetInsertOpConfigPathChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InsertOpCfgPath);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputFormat(const std::vector<char> &format) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310InputFormat] = CharToString(format);
}

std::vector<char> Ascend310DeviceInfo::GetInputFormatChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputFormat);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputShape(const std::vector<char> &shape) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310InputShape] = CharToString(shape);
}
std::vector<char> Ascend310DeviceInfo::GetInputShapeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputShape);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  std::string batchs;
  for (size_t i = 0; i < dynamic_batch_size.size(); ++i) {
    if (i != 0) {
      batchs.push_back(',');
    }
    batchs += std::to_string(dynamic_batch_size[i]);
  }
  data_->params[kModelOptionAscend310DynamicBatchSize] = batchs;
}

std::vector<char> Ascend310DeviceInfo::GetDynamicBatchSizeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310DynamicBatchSize);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310PrecisionMode] = CharToString(precision_mode);
}

std::vector<char> Ascend310DeviceInfo::GetPrecisionModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310PrecisionMode);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310OpSelectImplMode] = CharToString(op_select_impl_mode);
}

std::vector<char> Ascend310DeviceInfo::GetOpSelectImplModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310OpSelectImplMode);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetFusionSwitchConfigPath(const std::vector<char> &cfg_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[KModelOptionAscend310FusionSwitchCfgPath] = CharToString(cfg_path);
}
std::vector<char> Ascend310DeviceInfo::GetFusionSwitchConfigPathChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, KModelOptionAscend310FusionSwitchCfgPath);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputShapeMap(const std::map<int, std::vector<int>> &shape) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310InputShapeMap] = shape;
}

std::map<int, std::vector<int>> Ascend310DeviceInfo::GetInputShapeMap() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::map<int, std::vector<int>>();
  }
  return GetValue<std::map<int, std::vector<int>>>(data_, kModelOptionAscend310InputShapeMap);
}

void Ascend310DeviceInfo::SetOutputType(enum DataType output_type) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310OutputType] = output_type;
}

enum DataType Ascend310DeviceInfo::GetOutputType() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return DataType::kTypeUnknown;
  }
  return GetValue<enum DataType>(data_, kModelOptionAscend310OutputType);
}

void Ascend310DeviceInfo::SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscend310BufferOptimize] = CharToString(buffer_optimize_mode);
}

std::vector<char> Ascend310DeviceInfo::GetBufferOptimizeModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310BufferOptimize);
  return StringToChar(ref);
}
}  // namespace mindspore
