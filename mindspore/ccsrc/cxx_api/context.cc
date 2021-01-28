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
#include "utils/log_adapter.h"

constexpr auto kGlobalContextDeviceTarget = "mindspore.ascend.globalcontext.device_target";
constexpr auto kGlobalContextDeviceID = "mindspore.ascend.globalcontext.device_id";
constexpr auto kModelOptionInsertOpCfgPath = "mindspore.option.insert_op_config_file_path";  // aipp config file
constexpr auto kModelOptionInputFormat = "mindspore.option.input_format";                    // nchw or nhwc
constexpr auto kModelOptionInputShape = "mindspore.option.input_shape";
// Mandatory while dynamic batch: e.g. "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
constexpr auto kModelOptionOutputType = "mindspore.option.output_type";  // "FP32", "UINT8" or "FP16", default as "FP32"
constexpr auto kModelOptionPrecisionMode = "mindspore.option.precision_mode";
// "force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype" or "allow_mix_precision", default as "force_fp16"
constexpr auto kModelOptionOpSelectImplMode = "mindspore.option.op_select_impl_mode";

namespace mindspore {
template <class T>
static T GetValue(const std::shared_ptr<Context> &context, const std::string &key) {
  auto iter = context->params.find(key);
  if (iter == context->params.end()) {
    return T();
  }
  const std::any &value = iter->second;
  if (value.type() != typeid(T)) {
    return T();
  }

  return std::any_cast<T>(value);
}

std::shared_ptr<Context> GlobalContext::GetGlobalContext() {
  static std::shared_ptr<Context> g_context = std::make_shared<Context>();
  return g_context;
}

void GlobalContext::SetGlobalDeviceTarget(const std::string &device_target) {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  global_context->params[kGlobalContextDeviceTarget] = device_target;
}

std::string GlobalContext::GetGlobalDeviceTarget() {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  return GetValue<std::string>(global_context, kGlobalContextDeviceTarget);
}

void GlobalContext::SetGlobalDeviceID(const uint32_t &device_id) {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  global_context->params[kGlobalContextDeviceID] = device_id;
}

uint32_t GlobalContext::GetGlobalDeviceID() {
  auto global_context = GetGlobalContext();
  MS_EXCEPTION_IF_NULL(global_context);
  return GetValue<uint32_t>(global_context, kGlobalContextDeviceID);
}

void ModelContext::SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionInsertOpCfgPath] = cfg_path;
}

std::string ModelContext::GetInsertOpConfigPath(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::string>(context, kModelOptionInsertOpCfgPath);
}

void ModelContext::SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionInputFormat] = format;
}

std::string ModelContext::GetInputFormat(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::string>(context, kModelOptionInputFormat);
}

void ModelContext::SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionInputShape] = shape;
}

std::string ModelContext::GetInputShape(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::string>(context, kModelOptionInputShape);
}

void ModelContext::SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionOutputType] = output_type;
}

enum DataType ModelContext::GetOutputType(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<enum DataType>(context, kModelOptionOutputType);
}

void ModelContext::SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionPrecisionMode] = precision_mode;
}

std::string ModelContext::GetPrecisionMode(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::string>(context, kModelOptionPrecisionMode);
}

void ModelContext::SetOpSelectImplMode(const std::shared_ptr<Context> &context,
                                       const std::string &op_select_impl_mode) {
  MS_EXCEPTION_IF_NULL(context);
  context->params[kModelOptionOpSelectImplMode] = op_select_impl_mode;
}

std::string ModelContext::GetOpSelectImplMode(const std::shared_ptr<Context> &context) {
  MS_EXCEPTION_IF_NULL(context);
  return GetValue<std::string>(context, kModelOptionOpSelectImplMode);
}
}  // namespace mindspore
