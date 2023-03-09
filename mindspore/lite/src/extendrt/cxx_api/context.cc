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
#include "src/extendrt/cxx_api/context.h"
#include <string>
#include <memory>
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/lite_types.h"
#include "src/litert/inner_allocator.h"
#include "src/common/log_adapter.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_base.h"
#include "src/extendrt/delegate_graph_executor.h"

namespace mindspore {
constexpr auto kModelOptionCpuEnableFP16 = "mindspore.option.cpu.enable_fp16";
constexpr auto kModelOptionGPUEnableFP16 = "mindspore.option.gpu.enable_fp16";
constexpr auto kModelOptionNPUEnableFP16 = "mindspore.option.npu.enable_fp16";
constexpr auto kModelOptionGPUEnableGLTexture = "mindspore.option.gpu.enable_gl_texture_";
constexpr auto kModelOptionGPUGLContext = "mindspore.option.gpu.gl_context_";
constexpr auto kModelOptionGPUGLDisplay = "mindspore.option.gpu.gl_display_";
constexpr auto kModelOptionGPUDeviceID = "mindspore.option.gpu.device_id";
constexpr auto kModelOptionGPURankID = "mindspore.option.gpu.rank_id";
constexpr auto kModelOptionGPUGroupSize = "mindspore.option.gpu.group_size";
constexpr auto kModelOptionKirinNpuFrequency = "mindspore.option.kirin_npu.frequency";
constexpr auto kModelOptionProvider = "mindspore.option.provider";
constexpr auto kModelOptionProviderDevice = "mindspore.option.provider.device";
constexpr auto kModelOptionDeviceID = "mindspore.option.device_id";
constexpr auto kModelOptionAscendDeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscendInsertOpCfgPath = "mindspore.option.ascend.insert_op_config_file_path";
constexpr auto kModelOptionAscendInputFormat = "mindspore.option.ascend.input_format";
constexpr auto kModelOptionAscendInputShapeMap = "mindspore.option.ascend.input_shape_map";
constexpr auto kModelOptionAscendInputShape = "mindspore.option.ascend.input_shape";
constexpr auto kModelOptionAscendOutputType = "mindspore.option.ascend.output_type";
constexpr auto kModelOptionAscendPrecisionMode = "mindspore.option.ascend.precision_mode";
constexpr auto kModelOptionAscendOpSelectImplMode = "mindspore.option.ascend.op_select_impl_mode";
constexpr auto KModelOptionAscendFusionSwitchCfgPath = "mindspore.option.ascend.fusion_switch_config_file_path";
constexpr auto kModelOptionAscendDynamicBatchSize = "mindspore.option.ascend.dynamic_batch_size";
constexpr auto kModelOptionAscendDynamicImageSize = "mindspore.option.ascend.dynamic_image_size";
constexpr auto kModelOptionAscendBufferOptimize = "mindspore.option.ascend.buffer_optimize";
constexpr auto kModelOptionGPUPrecisionMode = "mindspore.option.gpu.precision_mode";
constexpr auto kModelOptionAscend910DeviceID = kModelOptionDeviceID;
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

void Context::SetInterOpParallelNum(int32_t parallel_num) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->inter_op_parallel_num_ = parallel_num;
}

int32_t Context::GetInterOpParallelNum() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return data_->inter_op_parallel_num_;
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
  if (mode < lite::NO_BIND || mode > lite::MID_CPU) {
    MS_LOG(WARNING) << "Invalid thread affinity mode: " << mode << ", change to NO_BIND mode.";
    data_->affinity_mode_ = lite::NO_BIND;
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

void Context::set_delegate(const std::shared_ptr<AbstractDelegate> &delegate) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->delegate = std::dynamic_pointer_cast<GraphSinkDelegate>(delegate);
}

std::shared_ptr<AbstractDelegate> Context::get_delegate() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return data_->delegate;
}

// deprecated
void Context::SetDelegate(const std::shared_ptr<Delegate> &delegate) { return; }

// deprecated
std::shared_ptr<Delegate> Context::GetDelegate() const { return nullptr; }

void Context::SetMultiModalHW(bool float_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->float_mode = float_mode;
}

bool Context::GetMultiModalHW() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }
  return data_->float_mode;
}

std::vector<std::shared_ptr<DeviceInfoContext>> &Context::MutableDeviceInfo() {
  static std::vector<std::shared_ptr<DeviceInfoContext>> empty{};
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return empty;
  }
  return data_->device_info_list;
}

DeviceInfoContext::DeviceInfoContext() : data_(std::make_shared<Data>()) {}

void DeviceInfoContext::SetProvider(const std::vector<char> &provider) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionProvider] = CharToString(provider);
}

void DeviceInfoContext::SetProviderDevice(const std::vector<char> &device) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionProviderDevice] = CharToString(device);
}

std::vector<char> DeviceInfoContext::GetProviderChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionProvider);
  return StringToChar(ref);
}

std::vector<char> DeviceInfoContext::GetProviderDeviceChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionProviderDevice);
  return StringToChar(ref);
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

void GPUDeviceInfo::SetEnableGLTexture(bool is_enable_gl_texture) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionGPUEnableGLTexture] = is_enable_gl_texture;
}

bool GPUDeviceInfo::GetEnableGLTexture() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return false;
  }
  return GetValue<bool>(data_, kModelOptionGPUEnableGLTexture);
}

void GPUDeviceInfo::SetGLContext(void *gl_context) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionGPUGLContext] = gl_context;
}

void *GPUDeviceInfo::GetGLContext() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return GetValue<void *>(data_, kModelOptionGPUGLContext);
}

void GPUDeviceInfo::SetGLDisplay(void *gl_display) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionGPUGLDisplay] = gl_display;
}

void *GPUDeviceInfo::GetGLDisplay() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return nullptr;
  }
  return GetValue<void *>(data_, kModelOptionGPUGLDisplay);
}

void KirinNPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionNPUEnableFP16] = is_fp16;
}
bool KirinNPUDeviceInfo::GetEnableFP16() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionNPUEnableFP16);
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

int GPUDeviceInfo::GetRankID() const {
  data_->params[kModelOptionGPURankID] = lite::GetRankID();
  return GetValue<int>(data_, kModelOptionGPURankID);
}

int GPUDeviceInfo::GetGroupSize() const {
  data_->params[kModelOptionGPUGroupSize] = lite::GetGPUGroupSize();
  return GetValue<int>(data_, kModelOptionGPUGroupSize);
}

void GPUDeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_LOG(ERROR) << "Unsupported Feature.";
}

std::vector<char> GPUDeviceInfo::GetPrecisionModeChar() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  std::vector<char> ret;
  return ret;
}

void AscendDeviceInfo::SetDeviceID(uint32_t device_id) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendDeviceID] = device_id;
}

uint32_t AscendDeviceInfo::GetDeviceID() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return 0;
  }
  return GetValue<uint32_t>(data_, kModelOptionAscendDeviceID);
}

void AscendDeviceInfo::SetInsertOpConfigPath(const std::vector<char> &cfg_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendInsertOpCfgPath] = CharToString(cfg_path);
}
std::vector<char> AscendDeviceInfo::GetInsertOpConfigPathChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendInsertOpCfgPath);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputFormat(const std::vector<char> &format) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendInputFormat] = CharToString(format);
}

std::vector<char> AscendDeviceInfo::GetInputFormatChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendInputFormat);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputShape(const std::vector<char> &shape) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendInputShape] = CharToString(shape);
}
std::vector<char> AscendDeviceInfo::GetInputShapeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendInputShape);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size) {
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
  data_->params[kModelOptionAscendDynamicBatchSize] = batchs;
}

std::vector<char> AscendDeviceInfo::GetDynamicBatchSizeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendDynamicBatchSize);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetDynamicImageSize(const std::vector<char> &dynamic_image_size) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendDynamicImageSize] = CharToString(dynamic_image_size);
}

std::vector<char> AscendDeviceInfo::GetDynamicImageSizeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendDynamicImageSize);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendPrecisionMode] = CharToString(precision_mode);
}

std::vector<char> AscendDeviceInfo::GetPrecisionModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendPrecisionMode);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendOpSelectImplMode] = CharToString(op_select_impl_mode);
}

std::vector<char> AscendDeviceInfo::GetOpSelectImplModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendOpSelectImplMode);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetFusionSwitchConfigPath(const std::vector<char> &cfg_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[KModelOptionAscendFusionSwitchCfgPath] = CharToString(cfg_path);
}
std::vector<char> AscendDeviceInfo::GetFusionSwitchConfigPathChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, KModelOptionAscendFusionSwitchCfgPath);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputShapeMap(const std::map<int, std::vector<int>> &shape) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendInputShapeMap] = shape;
}

std::map<int, std::vector<int>> AscendDeviceInfo::GetInputShapeMap() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::map<int, std::vector<int>>();
  }
  return GetValue<std::map<int, std::vector<int>>>(data_, kModelOptionAscendInputShapeMap);
}

void AscendDeviceInfo::SetOutputType(enum DataType output_type) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendOutputType] = output_type;
}

enum DataType AscendDeviceInfo::GetOutputType() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return DataType::kTypeUnknown;
  }
  return GetValue<enum DataType>(data_, kModelOptionAscendOutputType);
}

void AscendDeviceInfo::SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return;
  }
  data_->params[kModelOptionAscendBufferOptimize] = CharToString(buffer_optimize_mode);
}

std::vector<char> AscendDeviceInfo::GetBufferOptimizeModeChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return std::vector<char>();
  }
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscendBufferOptimize);
  return StringToChar(ref);
}
}  // namespace mindspore
