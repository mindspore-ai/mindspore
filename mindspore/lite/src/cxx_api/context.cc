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

#include "include/api/context.h"
#include <string>
#include <memory>
#include <any>
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "src/runtime/allocator.h"
#include "src/common/log_adapter.h"

namespace mindspore {
constexpr auto kModelOptionCpuEnableFP16 = "mindspore.option.cpu.enable_fp16";
constexpr auto kModelOptionCpuThreadAffinity = "mindspore.option.cpu.thread_affinity";
constexpr auto kModelOptionMaliGpuEnableFP16 = "mindspore.option.mali_gpu.enable_fp16";
constexpr auto kModelOptionKirinNpuFrequency = "mindspore.option.kirin_npu.frequency";

struct Context::Data {
  std::vector<std::shared_ptr<DeviceInfoContext>> device_info_list;
  int32_t thread_num = 2;
  std::shared_ptr<Allocator> allocator = nullptr;
};

struct DeviceInfoContext::Data {
  std::map<std::string, std::any> params;
};

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
  const std::any &value = iter->second;

  return std::any_cast<const U &>(value);
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

void Context::SetAllocator(const std::shared_ptr<Allocator> &allocator) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->allocator = allocator;
}
std::shared_ptr<Allocator> Context::GetAllocator() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return data_->allocator;
}

std::vector<std::shared_ptr<DeviceInfoContext>> &Context::MutableDeviceInfo() {
  static std::vector<std::shared_ptr<DeviceInfoContext>> empty;
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return empty;
  }
  return data_->device_info_list;
}

DeviceInfoContext::DeviceInfoContext() : data_(std::shared_ptr<Data>(new (std::nothrow) Data())) {}

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

void CPUDeviceInfo::SetThreadAffinity(int affinity) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionCpuThreadAffinity] = affinity;
}
int CPUDeviceInfo::GetThreadAffinity() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return GetValue<int>(data_, kModelOptionCpuThreadAffinity);
}

void MaliGPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionMaliGpuEnableFP16] = is_fp16;
}
bool MaliGPUDeviceInfo::GetEnableFP16() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }
  return GetValue<bool>(data_, kModelOptionMaliGpuEnableFP16);
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

void NvidiaGPUDeviceInfo::SetDeviceID(uint32_t device_id) { MS_LOG(ERROR) << "Unsupported Feature."; }
uint32_t NvidiaGPUDeviceInfo::GetDeviceID() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

void NvidiaGPUDeviceInfo::SetGpuTrtInferMode(bool gpu_trt_infer_mode) { MS_LOG(ERROR) << "Unsupported Feature."; }
bool NvidiaGPUDeviceInfo::GetGpuTrtInferMode() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

void Ascend910DeviceInfo::SetDeviceID(uint32_t device_id) { MS_LOG(ERROR) << "Unsupported Feature."; }
uint32_t Ascend910DeviceInfo::GetDeviceID() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

void Ascend310DeviceInfo::SetDeviceID(uint32_t device_id) { MS_LOG(ERROR) << "Unsupported Feature."; }
uint32_t Ascend310DeviceInfo::GetDeviceID() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

void Ascend310DeviceInfo::SetDumpConfigPath(const std::vector<char> &cfg_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetDumpConfigPathChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetInsertOpConfigPath(const std::vector<char> &cfg_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetInsertOpConfigPathChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetInputFormat(const std::vector<char> &format) { MS_LOG(ERROR) << "Unsupported Feature."; }
std::vector<char> Ascend310DeviceInfo::GetInputFormatChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetInputShape(const std::vector<char> &shape) { MS_LOG(ERROR) << "Unsupported Feature."; }
std::vector<char> Ascend310DeviceInfo::GetInputShapeChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetDynamicBatchSizeChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetPrecisionModeChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetOpSelectImplModeChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetFusionSwitchConfigPath(const std::vector<char> &cfg_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::vector<char> Ascend310DeviceInfo::GetFusionSwitchConfigPathChar() const {
  std::vector<char> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetInputShapeMap(const std::map<int, std::vector<int>> &shape) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}
std::map<int, std::vector<int>> Ascend310DeviceInfo::GetInputShapeMap() const {
  std::map<int, std::vector<int>> empty;
  MS_LOG(ERROR) << "Unsupported Feature.";
  return empty;
}

void Ascend310DeviceInfo::SetOutputType(enum DataType output_type) { MS_LOG(ERROR) << "Unsupported Feature."; }
enum DataType Ascend310DeviceInfo::GetOutputType() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return DataType::kTypeUnknown;
}
}  // namespace mindspore
