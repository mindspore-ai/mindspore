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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_OPLIB_SUPER_BAR_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_OPLIB_SUPER_BAR_
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <optional>
#include <nlohmann/json.hpp>
#include "include/backend/visible.h"
namespace mindspore::kernel {
class BACKEND_EXPORT SuperBar {
 public:
  SuperBar() = default;
  ~SuperBar() = default;
  static bool LoadSBConfig(const nlohmann::json &js);
  static std::string GetSBMSAttrByKernelAttr(const std::string &op_name, const std::string &attr_name);
  static std::string GetSBKernelAttrByMSAttr(const std::string &op_name, const std::string &attr_name);
  static std::string GetSBNodeAttrDefaultValue(const std::string &op_name, const std::string &attr_name);
  static std::optional<std::map<size_t, size_t>> GetKernelIdxToGraphIdx(const std::string &op_name);
  static std::optional<std::map<size_t, size_t>> GetGraphIdxToKernelIdx(const std::string &op_name);
  static bool IsSkipNode(const std::string &op_name);
  static bool IsSkipDynamicCompileStaticNode(const std::string &op_name);
  static std::vector<size_t> GetSBFallbackOpIndex(const std::string &op_name);

 private:
  static bool LoadSBNodeAttr(const nlohmann::json &js);
  static bool LoadSBNodeAttrDefaultValue(const nlohmann::json &js);
  static bool LoadSBNodeInput(const nlohmann::json &js);
  static bool LoadSBSkipNodes(const nlohmann::json &js);
  static bool LoadSBFallbackOps(const nlohmann::json &js);
  static bool LoadSBSkipDynamicCompileStaticNode(const nlohmann::json &js);
  inline static std::map<std::string, std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>>> node_input_order_ =
    {};
  inline static std::map<std::string, std::map<std::string, std::string>> node_attr_kernel_to_ms_;
  inline static std::map<std::string, std::map<std::string, std::string>> node_attr_ms_to_kernel_;
  inline static std::map<std::string, std::map<std::string, std::string>> node_attr_default_value_map_ = {};
  inline static std::vector<std::string> skip_nodes_;
  inline static std::map<std::string, std::vector<size_t>> fallback_ops_;
  inline static std::vector<std::string> skip_dynamic_compile_static_nodes_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_OPLIB_SUPER_BAR_
