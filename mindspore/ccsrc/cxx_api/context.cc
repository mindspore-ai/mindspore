/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <any>
#include <map>
#include <type_traits>
#include "cxx_api/factory.h"
#include "utils/log_adapter.h"

constexpr auto kModelOptionCpuEnableFP16 = "mindspore.option.cpu.enable_fp16";
constexpr auto kModelOptionCpuThreadAffinity = "mindspore.option.cpu.thread_affinity";
constexpr auto kModelOptionMaliGpuEnableFP16 = "mindspore.option.mali_gpu.enable_fp16";
constexpr auto kModelOptionKirinNpuFrequency = "mindspore.option.kirin_npu.frequency";
constexpr auto kModelOptionDeviceID = "mindspore.option.device_id";
constexpr auto kModelOptionNvidiaGpuDeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionNvidiaGpuTrtInferMode = "mindspore.option.nvidia_gpu.trt_infer_mode";
constexpr auto kModelOptionAscend910DeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscend310DeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscend310DumpCfgPath = "mindspore.option.ascend310.dump_config_file_path";
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

namespace mindspore {
class Allocator {};

struct Context::Data {
  std::vector<std::shared_ptr<DeviceInfoContext>> device_info_list;
  int32_t thread_num;
  std::shared_ptr<Allocator> allocator;
};

struct DeviceInfoContext::Data {
  std::map<std::string, std::any> params;
};

Context::Context() : data_(std::make_shared<Data>()) {}

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
  if (value.type() != typeid(U)) {
    return empty_result;
  }

  return std::any_cast<const U &>(value);
}

void Context::SetThreadNum(int32_t thread_num) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->thread_num = thread_num;
}
int32_t Context::GetThreadNum() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->thread_num;
}

void Context::SetAllocator(const std::shared_ptr<Allocator> &allocator) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->allocator = allocator;
}
std::shared_ptr<Allocator> Context::GetAllocator() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->allocator;
}

std::vector<std::shared_ptr<DeviceInfoContext>> &Context::MutableDeviceInfo() {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->device_info_list;
}

DeviceInfoContext::DeviceInfoContext() : data_(std::make_shared<Data>()) {}

void CPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionCpuEnableFP16] = is_fp16;
}
bool CPUDeviceInfo::GetEnableFP16() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionCpuEnableFP16);
}

void CPUDeviceInfo::SetThreadAffinity(int affinity) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionCpuThreadAffinity] = affinity;
}
int CPUDeviceInfo::GetThreadAffinity() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionCpuThreadAffinity);
}

void MaliGPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionMaliGpuEnableFP16] = is_fp16;
}
bool MaliGPUDeviceInfo::GetEnableFP16() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionMaliGpuEnableFP16);
}

void KirinNPUDeviceInfo::SetFrequency(int frequency) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionKirinNpuFrequency] = frequency;
}
int KirinNPUDeviceInfo::GetFrequency() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<int>(data_, kModelOptionKirinNpuFrequency);
}

void NvidiaGPUDeviceInfo::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionNvidiaGpuDeviceID] = device_id;
}
uint32_t NvidiaGPUDeviceInfo::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<uint32_t>(data_, kModelOptionNvidiaGpuDeviceID);
}

void NvidiaGPUDeviceInfo::SetGpuTrtInferMode(bool gpu_trt_infer_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionNvidiaGpuTrtInferMode] = gpu_trt_infer_mode;
}
bool NvidiaGPUDeviceInfo::GetGpuTrtInferMode() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionNvidiaGpuTrtInferMode);
}

void Ascend910DeviceInfo::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend910DeviceID] = device_id;
}
uint32_t Ascend910DeviceInfo::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<uint32_t>(data_, kModelOptionAscend910DeviceID);
}

void Ascend310DeviceInfo::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310DeviceID] = device_id;
}
uint32_t Ascend310DeviceInfo::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<uint32_t>(data_, kModelOptionAscend310DeviceID);
}

void Ascend310DeviceInfo::SetDumpConfigPath(const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310DumpCfgPath] = CharToString(cfg_path);
}
std::vector<char> Ascend310DeviceInfo::GetDumpConfigPathChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310DeviceID);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInsertOpConfigPath(const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InsertOpCfgPath] = CharToString(cfg_path);
}
std::vector<char> Ascend310DeviceInfo::GetInsertOpConfigPathChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InsertOpCfgPath);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputFormat(const std::vector<char> &format) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputFormat] = CharToString(format);
}
std::vector<char> Ascend310DeviceInfo::GetInputFormatChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputFormat);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputShape(const std::vector<char> &shape) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputShape] = CharToString(shape);
}
std::vector<char> Ascend310DeviceInfo::GetInputShapeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputShape);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size) {
  MS_EXCEPTION_IF_NULL(data_);
  std::string batchs = "";
  for (size_t i = 0; i < dynamic_batch_size.size(); ++i) {
    if (i != 0) {
      batchs.push_back(',');
    }
    batchs += std::to_string(dynamic_batch_size[i]);
  }
  data_->params[kModelOptionAscend310DynamicBatchSize] = batchs;
}
std::vector<char> Ascend310DeviceInfo::GetDynamicBatchSizeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310DynamicBatchSize);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310PrecisionMode] = CharToString(precision_mode);
}
std::vector<char> Ascend310DeviceInfo::GetPrecisionModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310PrecisionMode);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310OpSelectImplMode] = CharToString(op_select_impl_mode);
}
std::vector<char> Ascend310DeviceInfo::GetOpSelectImplModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310OpSelectImplMode);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetFusionSwitchConfigPath(const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[KModelOptionAscend310FusionSwitchCfgPath] = CharToString(cfg_path);
}
std::vector<char> Ascend310DeviceInfo::GetFusionSwitchConfigPathChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, KModelOptionAscend310FusionSwitchCfgPath);
  return StringToChar(ref);
}

void Ascend310DeviceInfo::SetInputShapeMap(const std::map<int, std::vector<int>> &shape) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputShapeMap] = shape;
}
std::map<int, std::vector<int>> Ascend310DeviceInfo::GetInputShapeMap() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<std::map<int, std::vector<int>>>(data_, kModelOptionAscend310InputShapeMap);
}

void Ascend310DeviceInfo::SetOutputType(enum DataType output_type) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310OutputType] = output_type;
}
enum DataType Ascend310DeviceInfo::GetOutputType() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<enum DataType>(data_, kModelOptionAscend310OutputType);
}

void Ascend310DeviceInfo::SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310BufferOptimize] = CharToString(buffer_optimize_mode);
}
std::vector<char> Ascend310DeviceInfo::GetBufferOptimizeModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310BufferOptimize);
  return StringToChar(ref);
}
}  // namespace mindspore
