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
#ifndef MINDSPORE_INCLUDE_API_CONTEXT_H
#define MINDSPORE_INCLUDE_API_CONTEXT_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "include/api/types.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
constexpr auto kDeviceTypeAscend310 = "Ascend310";
constexpr auto kDeviceTypeAscend910 = "Ascend910";
constexpr auto kDeviceTypeGPU = "GPU";

struct MS_API Context {
 public:
  Context();
  virtual ~Context() = default;
  struct Data;
  std::shared_ptr<Data> data;
};

struct MS_API GlobalContext : public Context {
 public:
  static std::shared_ptr<Context> GetGlobalContext();

  static inline void SetGlobalDeviceTarget(const std::string &device_target);
  static inline std::string GetGlobalDeviceTarget();

  static void SetGlobalDeviceID(const uint32_t &device_id);
  static uint32_t GetGlobalDeviceID();

  static inline void SetGlobalDumpConfigPath(const std::string &cfg_path);
  static inline std::string GetGlobalDumpConfigPath();

 private:
  // api without std::string
  static void SetGlobalDeviceTarget(const std::vector<char> &device_target);
  static std::vector<char> GetGlobalDeviceTargetChar();

  static void SetGlobalDumpConfigPath(const std::vector<char> &cfg_path);
  static std::vector<char> GetGlobalDumpConfigPathChar();
};

struct MS_API ModelContext : public Context {
 public:
  static inline void SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path);
  static inline std::string GetInsertOpConfigPath(const std::shared_ptr<Context> &context);

  static inline void SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format);
  static inline std::string GetInputFormat(const std::shared_ptr<Context> &context);

  static inline void SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape);
  static inline std::string GetInputShape(const std::shared_ptr<Context> &context);

  static void SetInputShapeMap(const std::shared_ptr<Context> &context, const std::map<int, std::vector<int>> &shape);
  static std::map<int, std::vector<int>> GetInputShapeMap(const std::shared_ptr<Context> &context);

  static void SetDynamicBatchSize(const std::shared_ptr<Context> &context,
                                  const std::vector<size_t> &dynamic_batch_size);
  static inline std::string GetDynamicBatchSize(const std::shared_ptr<Context> &context);

  static void SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type);
  static enum DataType GetOutputType(const std::shared_ptr<Context> &context);

  static inline void SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode);
  static inline std::string GetPrecisionMode(const std::shared_ptr<Context> &context);

  static inline void SetOpSelectImplMode(const std::shared_ptr<Context> &context,
                                         const std::string &op_select_impl_mode);
  static inline std::string GetOpSelectImplMode(const std::shared_ptr<Context> &context);

  static inline void SetFusionSwitchConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path);
  static inline std::string GetFusionSwitchConfigPath(const std::shared_ptr<Context> &context);

  static inline void SetGpuTrtInferMode(const std::shared_ptr<Context> &context, const std::string &gpu_trt_infer_mode);
  static inline std::string GetGpuTrtInferMode(const std::shared_ptr<Context> &context);

 private:
  // api without std::string
  static void SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::vector<char> &cfg_path);
  static std::vector<char> GetInsertOpConfigPathChar(const std::shared_ptr<Context> &context);

  static void SetInputFormat(const std::shared_ptr<Context> &context, const std::vector<char> &format);
  static std::vector<char> GetInputFormatChar(const std::shared_ptr<Context> &context);

  static void SetInputShape(const std::shared_ptr<Context> &context, const std::vector<char> &shape);
  static std::vector<char> GetInputShapeChar(const std::shared_ptr<Context> &context);

  static void SetPrecisionMode(const std::shared_ptr<Context> &context, const std::vector<char> &precision_mode);
  static std::vector<char> GetPrecisionModeChar(const std::shared_ptr<Context> &context);

  static void SetOpSelectImplMode(const std::shared_ptr<Context> &context,
                                  const std::vector<char> &op_select_impl_mode);
  static std::vector<char> GetOpSelectImplModeChar(const std::shared_ptr<Context> &context);

  static void SetFusionSwitchConfigPath(const std::shared_ptr<Context> &context, const std::vector<char> &cfg_path);
  static std::vector<char> GetFusionSwitchConfigPathChar(const std::shared_ptr<Context> &context);

  static void SetGpuTrtInferMode(const std::shared_ptr<Context> &context, const std::vector<char> &gpu_trt_infer_mode);
  static std::vector<char> GetGpuTrtInferModeChar(const std::shared_ptr<Context> &context);
  static std::vector<char> GetDynamicBatchSizeChar(const std::shared_ptr<Context> &context);
};

void GlobalContext::SetGlobalDeviceTarget(const std::string &device_target) {
  SetGlobalDeviceTarget(StringToChar(device_target));
}
std::string GlobalContext::GetGlobalDeviceTarget() { return CharToString(GetGlobalDeviceTargetChar()); }

void GlobalContext::SetGlobalDumpConfigPath(const std::string &cfg_path) {
  SetGlobalDumpConfigPath(StringToChar(cfg_path));
}
std::string GlobalContext::GetGlobalDumpConfigPath() { return CharToString(GetGlobalDumpConfigPathChar()); }

void ModelContext::SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path) {
  SetInsertOpConfigPath(context, StringToChar(cfg_path));
}
std::string ModelContext::GetInsertOpConfigPath(const std::shared_ptr<Context> &context) {
  return CharToString(GetInsertOpConfigPathChar(context));
}

void ModelContext::SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format) {
  SetInputFormat(context, StringToChar(format));
}
std::string ModelContext::GetInputFormat(const std::shared_ptr<Context> &context) {
  return CharToString(GetInputFormatChar(context));
}

void ModelContext::SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape) {
  SetInputShape(context, StringToChar(shape));
}
std::string ModelContext::GetInputShape(const std::shared_ptr<Context> &context) {
  return CharToString(GetInputShapeChar(context));
}

void ModelContext::SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode) {
  SetPrecisionMode(context, StringToChar(precision_mode));
}
std::string ModelContext::GetPrecisionMode(const std::shared_ptr<Context> &context) {
  return CharToString(GetPrecisionModeChar(context));
}

void ModelContext::SetOpSelectImplMode(const std::shared_ptr<Context> &context,
                                       const std::string &op_select_impl_mode) {
  SetOpSelectImplMode(context, StringToChar(op_select_impl_mode));
}
std::string ModelContext::GetOpSelectImplMode(const std::shared_ptr<Context> &context) {
  return CharToString(GetOpSelectImplModeChar(context));
}

void ModelContext::SetFusionSwitchConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path) {
  SetFusionSwitchConfigPath(context, StringToChar(cfg_path));
}
std::string ModelContext::GetFusionSwitchConfigPath(const std::shared_ptr<Context> &context) {
  return CharToString(GetFusionSwitchConfigPathChar(context));
}

std::string ModelContext::GetDynamicBatchSize(const std::shared_ptr<Context> &context) {
  return CharToString(GetDynamicBatchSizeChar(context));
}

void ModelContext::SetGpuTrtInferMode(const std::shared_ptr<Context> &context, const std::string &gpu_trt_infer_mode) {
  SetGpuTrtInferMode(context, StringToChar(gpu_trt_infer_mode));
}
std::string ModelContext::GetGpuTrtInferMode(const std::shared_ptr<Context> &context) {
  return CharToString(GetGpuTrtInferModeChar(context));
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CONTEXT_H
