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
#include "utils/log_adapter.h"

constexpr auto kGlobalContextDeviceTarget = "mindspore.ascend.globalcontext.device_target";
constexpr auto kGlobalContextDeviceID = "mindspore.ascend.globalcontext.device_id";
constexpr auto kGlobalContextDumpCfgPath = "mindspore.ascend.globalcontext.dump_config_file_path";
constexpr auto kModelOptionInsertOpCfgPath = "mindspore.option.insert_op_config_file_path";  // aipp config file
constexpr auto kModelOptionInputFormat = "mindspore.option.input_format";                    // nchw or nhwc
constexpr auto kModelOptionInputShapeMap = "mindspore.option.input_shape_map";
constexpr auto kModelOptionInputShape = "mindspore.option.input_shape";
// Mandatory while dynamic batch: e.g. "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
constexpr auto kModelOptionOutputType = "mindspore.option.output_type";  // "FP32", "UINT8" or "FP16", default as "FP32"
constexpr auto kModelOptionPrecisionMode = "mindspore.option.precision_mode";
// "force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype" or "allow_mix_precision", default as "force_fp16"
constexpr auto kModelOptionOpSelectImplMode = "mindspore.option.op_select_impl_mode";
constexpr auto KModelOptionFusionSwitchCfgPath = "mindspore.option.fusion_switch_config_file_path";
// "False": Inference with native backend, "True": Inference with Tensor-RT engine, default as "False"
constexpr auto kModelOptionGpuTrtInferMode = "mindspore.option.gpu_trt_infer_mode";
constexpr auto kModelOptionDynamicBatchSize = "mindspore.option.dynamic_batch_size";
constexpr auto kModelOptionDynamicImageSize = "mindspore.option.dynamic_image_size";

namespace mindspore {
struct Context::Data {
  std::map<std::string, std::any> params;
};

Context::Context() : data(std::make_shared<Data>()) {}

template <class T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
static const U &GetValue(const std::shared_ptr<Context> &context, const std::string &key) {
  static U empty_result;
  if (context == nullptr || context->data == nullptr) {
    return empty_result;
  }
  auto iter = context->data->params.find(key);
  if (iter == context->data->params.end()) {
    return empty_result;
  }
  const std::any &value = iter->second;
  if (value.type() != typeid(U)) {
    return empty_result;
  }

  return std::any_cast<const U &>(value);
}

std::shared_ptr<Context> GlobalContext::GetGlobalContext() {
  static std::shared_ptr<Context> g_context = std::make_shared<Context>();
  return g_context;
}

void GlobalContext::SetGlobalDeviceTarget(const std::vector<char> &device_target) {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  if (global_context->data == nullptr) {
    global_context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(global_context->data);
  }
  global_context->data->params[kGlobalContextDeviceTarget] = CharToString(device_target);
}

std::vector<char> GlobalContext::GetGlobalDeviceTargetChar() {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  const std::string &ref = GetValue<std::string>(global_context, kGlobalContextDeviceTarget);
  return StringToChar(ref);
}

void GlobalContext::SetGlobalDeviceID(const uint32_t &device_id) {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  if (global_context->data == nullptr) {
    global_context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(global_context->data);
  }
  global_context->data->params[kGlobalContextDeviceID] = device_id;
}

uint32_t GlobalContext::GetGlobalDeviceID() {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  return GetValue<uint32_t>(global_context, kGlobalContextDeviceID);
}

void GlobalContext::SetGlobalDumpConfigPath(const std::vector<char> &cfg_path) {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  if (global_context->data == nullptr) {
    global_context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(global_context->data);
  }
  global_context->data->params[kGlobalContextDumpCfgPath] = CharToString(cfg_path);
}

std::vector<char> GlobalContext::GetGlobalDumpConfigPathChar() {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  const std::string &ref = GetValue<std::string>(global_context, kGlobalContextDumpCfgPath);
  return StringToChar(ref);
}

void ModelContext::SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionInsertOpCfgPath] = CharToString(cfg_path);
}

std::vector<char> ModelContext::GetInsertOpConfigPathChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionInsertOpCfgPath);
  return StringToChar(ref);
}

void ModelContext::SetInputFormat(const std::shared_ptr<Context> &context, const std::vector<char> &format) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionInputFormat] = CharToString(format);
}

std::vector<char> ModelContext::GetInputFormatChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionInputFormat);
  return StringToChar(ref);
}

void ModelContext::SetInputShape(const std::shared_ptr<Context> &context, const std::vector<char> &shape) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionInputShape] = CharToString(shape);
}

std::vector<char> ModelContext::GetInputShapeChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionInputShape);
  return StringToChar(ref);
}

void ModelContext::SetInputShapeMap(const std::shared_ptr<Context> &context,
                                    const std::map<int, std::vector<int>> &shape) {
  MS_EXCEPTION_IF_NULL(context);
  context->data->params[kModelOptionInputShapeMap] = shape;
}

std::map<int, std::vector<int>> ModelContext::GetInputShapeMap(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::map<int, std::vector<int>>>(context, kModelOptionInputShapeMap);
}

void ModelContext::SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionOutputType] = output_type;
}

enum DataType ModelContext::GetOutputType(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<enum DataType>(context, kModelOptionOutputType);
}

void ModelContext::SetPrecisionMode(const std::shared_ptr<Context> &context, const std::vector<char> &precision_mode) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionPrecisionMode] = CharToString(precision_mode);
}

std::vector<char> ModelContext::GetPrecisionModeChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionPrecisionMode);
  return StringToChar(ref);
}

void ModelContext::SetOpSelectImplMode(const std::shared_ptr<Context> &context,
                                       const std::vector<char> &op_select_impl_mode) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionOpSelectImplMode] = CharToString(op_select_impl_mode);
}

std::vector<char> ModelContext::GetOpSelectImplModeChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionOpSelectImplMode);
  return StringToChar(ref);
}

void ModelContext::SetFusionSwitchConfigPath(const std::shared_ptr<Context> &context,
                                             const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[KModelOptionFusionSwitchCfgPath] = CharToString(cfg_path);
}

std::vector<char> ModelContext::GetFusionSwitchConfigPathChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, KModelOptionFusionSwitchCfgPath);
  return StringToChar(ref);
}

void ModelContext::SetGpuTrtInferMode(const std::shared_ptr<Context> &context,
                                      const std::vector<char> &gpu_trt_infer_mode) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  context->data->params[kModelOptionGpuTrtInferMode] = CharToString(gpu_trt_infer_mode);
}

std::vector<char> ModelContext::GetGpuTrtInferModeChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionGpuTrtInferMode);
  return StringToChar(ref);
}

void ModelContext::SetDynamicBatchSize(const std::shared_ptr<Context> &context, const std::vector<size_t> &batch_size) {
  MS_EXCEPTION_IF_NULL(context);
  if (context->data == nullptr) {
    context->data = std::make_shared<Data>();
    MS_EXCEPTION_IF_NULL(context->data);
  }
  std::string batchs = "";
  for (auto bs : batch_size) {
    batchs += std::to_string(bs) + ",";
  }
  context->data->params[kModelOptionDynamicBatchSize] = batchs;
}

std::vector<char> ModelContext::GetDynamicBatchSizeChar(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  const std::string &ref = GetValue<std::string>(context, kModelOptionDynamicBatchSize);
  return StringToChar(ref);
}
}  // namespace mindspore
