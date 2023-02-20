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
#include "kernel/oplib/super_bar.h"

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "nlohmann/detail/iterators/iter_impl.hpp"
#include "utils/log_adapter.h"
namespace mindspore::kernel {
// super bar config
constexpr auto kNodeAttrMap = "NodeAttrMap";
constexpr auto kAttrDefaultValue = "AttrDefaultValue";
constexpr auto kNodeName = "NodeName";
constexpr auto kInputOrders = "InputOrders";
constexpr auto kSkipNodes = "SkipNodes";
constexpr auto kFallbackOps = "FallbackOps";
constexpr auto kSkipDynamicCompileStatic = "SkipDynamicCompileStatic";
bool SuperBar::LoadSBConfig(const nlohmann::json &js) {
  if (!LoadSBNodeAttr(js)) {
    return false;
  }
  if (!LoadSBNodeInput(js)) {
    return false;
  }
  if (!LoadSBNodeAttrDefaultValue(js)) {
    return false;
  }
  if (!LoadSBSkipDynamicCompileStaticNode(js)) {
    return false;
  }
  if (!LoadSBSkipNodes(js)) {
    return false;
  }
  if (!LoadSBFallbackOps(js)) {
    return false;
  }
  return true;
}

std::string SuperBar::GetSBMSAttrByKernelAttr(const std::string &op_name, const std::string &attr_name) {
  auto iter = node_attr_kernel_to_ms_.find(op_name);
  if (iter == node_attr_kernel_to_ms_.end()) {
    return attr_name;
  }
  auto iter_attr = iter->second.find(attr_name);
  if (iter_attr == iter->second.end()) {
    return attr_name;
  }
  return iter_attr->second;
}

std::string SuperBar::GetSBKernelAttrByMSAttr(const std::string &op_name, const std::string &attr_name) {
  auto iter = node_attr_ms_to_kernel_.find(op_name);
  if (iter == node_attr_ms_to_kernel_.end()) {
    return attr_name;
  }
  auto iter_attr = iter->second.find(attr_name);
  if (iter_attr == iter->second.end()) {
    return attr_name;
  }
  return iter_attr->second;
}

std::string SuperBar::GetSBNodeAttrDefaultValue(const std::string &op_name, const std::string &attr_name) {
  auto iter = node_attr_default_value_map_.find(op_name);
  if (iter == node_attr_default_value_map_.end()) {
    return "";
  }
  auto iter_attr_value = iter->second.find(attr_name);
  if (iter_attr_value == iter->second.end()) {
    return "";
  }
  return iter_attr_value->second;
}

std::optional<std::map<size_t, size_t>> SuperBar::GetKernelIdxToGraphIdx(const std::string &op_name) {
  auto iter = node_input_order_.find(op_name);
  if (iter == node_input_order_.end()) {
    return std::nullopt;
  }
  return iter->second.first;
}

std::optional<std::map<size_t, size_t>> SuperBar::GetGraphIdxToKernelIdx(const std::string &op_name) {
  auto iter = node_input_order_.find(op_name);
  if (iter == node_input_order_.end()) {
    return std::nullopt;
  }
  return iter->second.second;
}

bool SuperBar::IsSkipNode(const std::string &op_name) {
  return (std::find(skip_nodes_.begin(), skip_nodes_.end(), op_name) != skip_nodes_.end());
}

std::vector<size_t> SuperBar::GetSBFallbackOpIndex(const std::string &op_name) {
  auto iter = fallback_ops_.find(op_name);
  if (iter == fallback_ops_.end()) {
    return {};
  }
  return iter->second;
}

bool SuperBar::IsSkipDynamicCompileStaticNode(const std::string &op_name) {
  return (std::find(skip_dynamic_compile_static_nodes_.begin(), skip_dynamic_compile_static_nodes_.end(), op_name) !=
          skip_dynamic_compile_static_nodes_.end());
}

bool SuperBar::LoadSBNodeAttr(const nlohmann::json &js) {
  if (js.find(kNodeAttrMap) == js.end()) {
    MS_LOG(ERROR) << "Find node attr map failed.";
    return false;
  }
  nlohmann::json node_attr_maps = js.at(kNodeAttrMap);
  for (auto iter = node_attr_maps.begin(); iter != node_attr_maps.end(); ++iter) {
    const auto &node_type = iter.key();
    auto attrs = iter->get<nlohmann::json>();
    std::map<std::string, std::string> attr_kernel_to_ms_map;
    std::map<std::string, std::string> attr_ms_to_kernel_map;
    for (auto iter_attr = attrs.begin(); iter_attr != attrs.end(); ++iter_attr) {
      attr_kernel_to_ms_map.insert({iter_attr.key(), iter_attr->get<std::string>()});
      attr_ms_to_kernel_map.insert({iter_attr->get<std::string>(), iter_attr.key()});
    }
    node_attr_kernel_to_ms_[node_type] = attr_kernel_to_ms_map;
    node_attr_ms_to_kernel_[node_type] = attr_ms_to_kernel_map;
  }
  return true;
}

bool SuperBar::LoadSBNodeAttrDefaultValue(const nlohmann::json &js) {
  if (js.find(kAttrDefaultValue) == js.end()) {
    MS_LOG(ERROR) << "Find node attr default value failed.";
    return false;
  }
  nlohmann::json node_attr_maps = js.at(kAttrDefaultValue);
  for (auto iter = node_attr_maps.begin(); iter != node_attr_maps.end(); ++iter) {
    const auto &node_type = iter.key();
    auto attrs = iter->get<nlohmann::json>();
    std::map<std::string, std::string> attr_map;
    for (auto iter_attr = attrs.begin(); iter_attr != attrs.end(); ++iter_attr) {
      attr_map.insert({iter_attr.key(), iter_attr->get<std::string>()});
    }
    node_attr_default_value_map_[node_type] = attr_map;
  }
  return true;
}

bool SuperBar::LoadSBNodeInput(const nlohmann::json &js) {
  if (js.find(kInputOrders) == js.end()) {
    MS_LOG(ERROR) << "Find node input orders key failed, json: " << js.dump();
    return false;
  }
  nlohmann::json node_input_orders = js.at(kInputOrders);
  for (auto iter = node_input_orders.begin(); iter != node_input_orders.end(); ++iter) {
    const auto &node_type = iter.key();
    auto orders = iter->get<std::vector<size_t>>();
    std::map<size_t, size_t> kernel_idx_to_graph_idx;
    std::map<size_t, size_t> graph_idx_to_kernel_idx;
    for (size_t i = 0; i < orders.size(); ++i) {
      (void)kernel_idx_to_graph_idx.insert({i, orders[i]});
      (void)graph_idx_to_kernel_idx.insert({orders[i], i});
    }
    node_input_order_[node_type] = {kernel_idx_to_graph_idx, graph_idx_to_kernel_idx};
  }
  return true;
}

bool SuperBar::LoadSBFallbackOps(const nlohmann::json &js) {
  // some ops like "DeformableOffsets", need delete assist input before AI_CPU kernel select
  auto js_iter = js.find(kFallbackOps);
  if (js_iter == js.end()) {
    MS_LOG(ERROR) << "Find fallback node failed, json: " << js.dump();
    return false;
  }
  const auto &fallback_nodes = js_iter->get<nlohmann::json>();
  for (auto iter = fallback_nodes.begin(); iter != fallback_nodes.end(); ++iter) {
    const auto &node_name = iter.key();
    fallback_ops_[node_name] = iter->get<std::vector<size_t>>();
  }
  return true;
}

bool SuperBar::LoadSBSkipNodes(const nlohmann::json &js) {
  if (js.find(kSkipNodes) == js.end()) {
    MS_LOG(ERROR) << "Find skip node failed, json: " << js.dump();
    return false;
  }
  nlohmann::json skip_nodes = js.at(kSkipNodes);
  if (!skip_nodes.is_array()) {
    MS_LOG(ERROR) << "skip nodes info is error, json: " << skip_nodes.dump();
    return false;
  }
  skip_nodes_ = skip_nodes.get<std::vector<std::string>>();
  return true;
}

bool SuperBar::LoadSBSkipDynamicCompileStaticNode(const nlohmann::json &js) {
  if (js.find(kSkipDynamicCompileStatic) == js.end()) {
    MS_LOG(ERROR) << "Find skip dynamic compile static node failed, json: " << js.dump();
    return false;
  }
  nlohmann::json skip_dynamic_compile_static_nodes = js.at(kSkipDynamicCompileStatic);
  if (!skip_dynamic_compile_static_nodes.is_array()) {
    MS_LOG(ERROR) << "skip nodes info is error, json: " << skip_dynamic_compile_static_nodes.dump();
    return false;
  }
  skip_dynamic_compile_static_nodes_ = skip_dynamic_compile_static_nodes.get<std::vector<std::string>>();
  return true;
}
}  // namespace mindspore::kernel
