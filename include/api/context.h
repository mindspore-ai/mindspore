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

#include <map>
#include <any>
#include <string>
#include <memory>
#include "include/api/types.h"

namespace mindspore {
constexpr auto kDeviceTypeAscend310 = "Ascend310";
constexpr auto kDeviceTypeAscend910 = "Ascend910";
constexpr auto kDeviceTypeGPU = "GPU";

struct MS_API Context {
  virtual ~Context() = default;
  std::map<std::string, std::any> params;
};

struct MS_API GlobalContext : public Context {
  static std::shared_ptr<Context> GetGlobalContext();

  static void SetGlobalDeviceTarget(const std::string &device_target);
  static std::string GetGlobalDeviceTarget();

  static void SetGlobalDeviceID(const uint32_t &device_id);
  static uint32_t GetGlobalDeviceID();
};

struct MS_API ModelContext : public Context {
  static void SetInsertOpConfigPath(const std::shared_ptr<Context> &context, const std::string &cfg_path);
  static std::string GetInsertOpConfigPath(const std::shared_ptr<Context> &context);

  static void SetInputFormat(const std::shared_ptr<Context> &context, const std::string &format);
  static std::string GetInputFormat(const std::shared_ptr<Context> &context);

  static void SetInputShape(const std::shared_ptr<Context> &context, const std::string &shape);
  static std::string GetInputShape(const std::shared_ptr<Context> &context);

  static void SetOutputType(const std::shared_ptr<Context> &context, enum DataType output_type);
  static enum DataType GetOutputType(const std::shared_ptr<Context> &context);

  static void SetPrecisionMode(const std::shared_ptr<Context> &context, const std::string &precision_mode);
  static std::string GetPrecisionMode(const std::shared_ptr<Context> &context);

  static void SetOpSelectImplMode(const std::shared_ptr<Context> &context, const std::string &op_select_impl_mode);
  static std::string GetOpSelectImplMode(const std::shared_ptr<Context> &context);
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CONTEXT_H
