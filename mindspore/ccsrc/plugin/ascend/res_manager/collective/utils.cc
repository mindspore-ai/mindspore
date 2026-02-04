/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/collective/utils.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/core/include/utils/ms_utils.h"
#include "mindspore/core/include/utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
class Config {
 public:
  static GroupsVecBuffleSet ParseListConfig(const std::string &list_cfg);
  static GroupsVecBuffleSet ParseStrideConfig(const std::string &stride_cfg);

 private:
  static std::string TrimSpace(const std::string &str);
  static std::vector<std::string> Split(const std::string &str, char delim, bool skip_empty = true);
  static std::optional<std::pair<std::string, std::string>> ParseKeyValue(const std::string &seg, char delim);
  static void InsertStrideGroup(GroupsVecBuffleSet *result, const std::vector<uint32_t> &group,
                                const std::string &value);
  static std::map<std::string, std::map<std::string, std::string>> configs;
  static std::set<std::string> has_parsed_config;

  static std::mutex mutex_;
};

std::map<std::string, std::map<std::string, std::string>> Config::configs;
std::set<std::string> Config::has_parsed_config;
std::mutex Config::mutex_;

std::string Config::TrimSpace(const std::string &str) {
  auto first = str.find_first_not_of(kWhiteSpace);
  if (first == std::string::npos) {
    return "";
  }
  auto last = str.find_last_not_of(kWhiteSpace);
  return str.substr(first, last - first + 1);
}

std::vector<std::string> Config::Split(const std::string &str, char delim, bool skip_empty) {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    item = TrimSpace(item);
    if (!item.empty() || !skip_empty) {
      (void)result.emplace_back(std::move(item));
    }
  }
  return result;
}

std::optional<std::pair<std::string, std::string>> Config::ParseKeyValue(const std::string &seg, char delim) {
  auto delim_pos = seg.find(delim);
  if (delim_pos == std::string::npos || delim_pos == 0 || delim_pos + 1 >= seg.size()) return std::nullopt;
  std::string key = seg.substr(0, delim_pos);
  std::string value = seg.substr(delim_pos + 1);
  key = TrimSpace(key);
  value = TrimSpace(value);
  return std::optional(std::make_pair(key, value));
}

GroupsVecBuffleSet Config::ParseListConfig(const std::string &list_cfg) {
  GroupsVecBuffleSet result;
  // hccl_list_config:0-1-2-3=200|4-5-6-7=100
  for (auto seg : Split(list_cfg, '|')) {
    // 0-2-1-3=200
    auto kv_pair = ParseKeyValue(seg, '=');
    if (!kv_pair.has_value()) {
      continue;
    }
    const auto &[key, value] = kv_pair.value();
    if (key.empty() || value.empty()) continue;

    // "0-2-1-3" -> [0, 1, 2, 3]
    std::vector<unsigned int> key_list;
    for (const auto &elem : Split(key, '-')) {
      try {
        key_list.push_back(static_cast<unsigned int>(std::stoul(elem)));
      } catch (const std::exception &e) {
        key_list.clear();
        break;
      }
    }
    if (key_list.empty()) continue;
    std::sort(key_list.begin(), key_list.end());
    key_list.erase(std::unique(key_list.begin(), key_list.end()), key_list.end());
    std::string val = value;
    if (val.size() > 2 && (val.substr(val.size() - 2) == "MB" || val.substr(val.size() - 2) == "mb")) {
      val = val.substr(0, val.size() - 2);
      val = TrimSpace(val);
    }
    result.insert(std::make_pair(std::move(key_list), std::move(val)));
  }
  return result;
}

void Config::InsertStrideGroup(GroupsVecBuffleSet *result, const std::vector<uint32_t> &group,
                               const std::string &value) {
  if (group.empty()) {
    return;
  }
  std::string val = value;
  if (val.size() > 2 && (val.substr(val.size() - 2) == "MB" || val.substr(val.size() - 2) == "mb")) {
    val = val.substr(0, val.size() - 2);
    val = TrimSpace(val);
  }
  result->insert(std::make_pair(group, std::move(val)));
}

GroupsVecBuffleSet Config::ParseStrideConfig(const std::string &stride_cfg) {
  GroupsVecBuffleSet result;
  // hccl_stride_config:0-4:2=100MB|4-7:1=200MB
  for (const auto &seg : Split(stride_cfg, '|')) {
    // 0-4:2=100
    auto first_kv_pair = ParseKeyValue(seg, '=');
    if (!first_kv_pair.has_value()) continue;
    const auto &[range_step_str, value] = first_kv_pair.value();
    if (range_step_str.empty() || value.empty()) continue;

    // 0-4:2 -> range: 0-4, step: 2
    auto second_kv_pair = ParseKeyValue(range_step_str, ':');
    if (!second_kv_pair.has_value()) continue;
    const auto &[range_str, step_str] = second_kv_pair.value();

    // range: 0-4, step: 2 -> [0-2-4]:100
    auto dash_pos = range_str.find('-');
    if (dash_pos == std::string::npos || dash_pos == 0 || dash_pos + 1 >= range_str.size()) continue;
    try {
      uint32_t start = static_cast<uint32_t>(std::stoul(range_str.substr(0, dash_pos)));
      uint32_t end = static_cast<uint32_t>(std::stoul(range_str.substr(dash_pos + 1)));
      uint32_t step = static_cast<uint32_t>(std::stoul(step_str));
      if (start > end || step == 0) continue;
      std::vector<uint32_t> group;
      for (uint32_t i = start; i <= end; i += step) {
        group.push_back(i);
        if (i + step > end) break;
      }
      if ((end - start) % step == 0 && (group.empty() || group.back() != end)) {
        group.push_back(end);
      }
      InsertStrideGroup(&result, group, value);
    } catch (const std::exception &) {
      continue;
    }
  }
  return result;
}
}  // namespace

using mindspore::common::GetConfigValue;

std::map<std::vector<unsigned int>, uint32_t> GetHcclBuffleConfig() {
  std::map<std::vector<unsigned int>, uint32_t> result;
  const auto &hccl_list_config = GetConfigValue(kHcclConf, kHcclListConfig);
  const auto &hccl_stride_config = GetConfigValue(kHcclConf, kHcclStrideConfig);
  GroupsVecBuffleSet hccl_list_set = Config::ParseListConfig(hccl_list_config);
  GroupsVecBuffleSet hccl_stride_set = Config::ParseStrideConfig(hccl_stride_config);

  for (const auto &item : hccl_list_set) {
    uint32_t value = 0;
    try {
      if (!item.second.empty()) {
        value = static_cast<uint32_t>(std::stoul(item.second));
      }
    } catch (const std::exception &) {
      continue;
    }
    if (value > 0) {
      result[item.first] = value;
    }
  }

  for (const auto &item : hccl_stride_set) {
    uint32_t value = 0;
    try {
      if (!item.second.empty()) {
        value = static_cast<uint32_t>(std::stoul(item.second));
      }
    } catch (const std::exception &) {
      continue;
    }
    if (value > 0) {
      result[item.first] = value;
    }
  }

  return result;
}

std::string GetHcclConfigValue(const std::string &hccl_config) {
  const auto &value = GetConfigValue(kHcclConf, hccl_config);
  return value;
}

bool IsEnableHcclConfig(const std::string &enable_config) {
  const auto &value = GetConfigValue(kHcclConf, enable_config);
  return ((value == "True") || (value == "true"));
}

bool IsDisableHcclConfig(const std::string &enable_config) {
  const auto &value = GetConfigValue(kHcclConf, enable_config);
  return ((value == "False") || (value == "false"));
}

uint32_t HcclBufferSize(const std::vector<unsigned int> &rank_id_list, const uint32_t &buffer_size) {
  uint32_t update_buffsize = buffer_size;
  if (IsEnableHcclConfig(kHcclEnableConfig)) {
    std::vector<unsigned int> sorted_ranks(rank_id_list.begin(), rank_id_list.end());
    std::sort(sorted_ranks.begin(), sorted_ranks.end());
    sorted_ranks.erase(std::unique(sorted_ranks.begin(), sorted_ranks.end()), sorted_ranks.end());

    // modify default buffer size ​​uniformly
    // "enable_hccl_config:True,hccl_customized_default:100MB"
    const auto &align_size = GetHcclConfigValue(kHcclCustomizedDefault);
    if (!align_size.empty()) {
      uint32_t value = static_cast<uint32_t>(std::stoul(align_size));
      update_buffsize = value;
    }

    // modify the buffer size of specific groups
    // "enable_hccl_config:True,hccl_list_config:0-1-2-3=200MB|4-5-6-7=100MB,hccl_stride_config:0-4:2=100MB"
    const std::map<std::vector<unsigned int>, uint32_t> &rank_list_buffle_size = GetHcclBuffleConfig();
    auto iter = rank_list_buffle_size.find(sorted_ranks);
    if (iter != rank_list_buffle_size.end() && iter->second > 0) {
      update_buffsize = iter->second;
    }
  }
  return update_buffsize;
}

uint32_t GetHcclBufferSize(const std::string &group_name, const std::vector<unsigned int> &rank_id_list) {
  uint32_t buffer_size = 0;
  std::string rank_list = VectorUtils::PrintVector(rank_id_list);
  std::string buffer_size_str = "default hcclBufferSize: " + std::to_string(HCCL_COMM_DEFAULT_BUFFSIZE) + " MB.";
  buffer_size = HcclBufferSize(rank_id_list, buffer_size);

  // MS_DEV_HCCL_CONF
  if (buffer_size > 0) {
    buffer_size_str = "customized hcclBufferSize: " + std::to_string(buffer_size) + " MB.";
  }

  // HCCL_BUFFSIZE
  if (buffer_size == 0) {
    uint32_t default_size = 0;
    std::string hccl_buffer_env = common::GetEnv("HCCL_BUFFSIZE");
    if (!hccl_buffer_env.empty()) {
      default_size = static_cast<uint32_t>(std::stoul(hccl_buffer_env));
    }
    if (default_size > 0) {
      buffer_size = default_size;
      buffer_size_str = "HCCL_BUFFSIZE: " + std::to_string(buffer_size) + " MB.";
    }
  }

  MS_LOG(INFO) << "HcclGroup " << group_name << ", ranks are " << rank_list << ", " << buffer_size_str;
  return buffer_size;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
