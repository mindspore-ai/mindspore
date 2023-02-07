/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_UTILS_H_
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include "kernel/oplib/opinfo.h"
namespace mindspore::kernel {
class BACKEND_EXPORT OpInfoUtils {
 public:
  static bool GenerateOpInfos(const std::string &version, const std::string &ascend_path);

 private:
  OpInfoUtils() = default;
  ~OpInfoUtils() = default;
  static bool LoadOpInfoJson(const std::string &version, const std::string &ascend_path, nlohmann::json *js_);
  static bool ParseCommonItem(const nlohmann::json &item, const OpInfoPtr &op_info_ptr);
  static bool ParseAttrs(const nlohmann::json &item, const OpInfoPtr &op_info_ptr);
  static bool ParseOpIOInfo(const nlohmann::json &item, const OpInfoPtr &op_info_ptr);
  static bool ParseOpIOInfoImpl(const nlohmann::json &item, bool is_input, const OpInfoPtr &op_info_ptr);
  static void UpdateInputOrders(const OpInfoPtr &op_info_ptr);
  static void UpdateRefInfo(const OpInfoPtr &op_info_ptr);
  static std::string NormalizeKernelName(const std::string &op_name);
  static void SetKernelName(const OpInfoPtr &op_info_ptr);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_UTILS_H_
