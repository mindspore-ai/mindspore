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

#include "tools/silent_detect/silent_detect_config_parser.h"

#include <iostream>
#include <sstream>
#include <string>
#include "include/utils/parallel_context.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace {
constexpr auto kMsNpuAsdConfig = "MS_NPU_ASD_CONFIG";

const char kSpaceChar = ' ';
const char kColonSeparator = ':';
const char kEqualSeparator = '=';
const char kCommaSeparator = ',';
const int kNumber3 = 3;

constexpr auto kEnable = "enable";
constexpr auto kWithChecksum = "with_checksum";
constexpr auto kCoolDownKey = "cooldown";
constexpr auto kStrikesNumKey = "strikes_num";
constexpr auto kStrikesWindowKey = "strikes_window";
constexpr auto kChecksumCooldownKey = "checksum_cooldown";
constexpr auto kUpperThresh1Key = "upper_thresh1";
constexpr auto kUpperThresh2Key = "upper_thresh2";
constexpr auto kGradSampleIntervalKey = "grad_sample_interval";

const int kDefaultGradSampleInterval = 10;
const int kDefaultUpperThresh1 = 1e6;
const int kDefaultUpperThresh2 = 1e2;
const int kDefaultCooldown = 5;
const int kDefaultStrikesNum = 3;
const int kDefaultStrikesWindow = 480;
const int kDefaultChecksumCooldown = 180;
}  // namespace

namespace mindspore {
namespace silentdetect {
bool SilentDetectConfigParser::ParseConfigs(const std::string &config_str) {
  ConfigMap config_map;
  std::stringstream ss(config_str);
  std::string item;

  while (std::getline(ss, item, kCommaSeparator)) {
    item = Trim(item);
    if (item.empty()) continue;

    size_t pos = item.find(kColonSeparator);
    if (pos == std::string::npos) {
      pos = item.find(kEqualSeparator);
    }

    if (pos != std::string::npos) {
      std::string key = Trim(item.substr(0, pos));
      std::string value = Trim(item.substr(pos + 1));
      if (!key.empty()) {
        config_map[key] = value;
      } else {
        MS_LOG(WARNING) << "Empty key in config: " << item;
      }
    } else {
      MS_LOG(WARNING) << "Invalid config format: " << item;
    }
  }

  ParseEnable(config_map);
  ParseWithChecksum(config_map);
  ParseCooldown(config_map);
  ParseStrikeNum(config_map);
  ParseStrikeWindow(config_map);
  ParseChecksumCooldown(config_map);
  ParseUpperThresh1(config_map);
  ParseUpperThresh2(config_map);
  ParseGradSampleInterval(config_map);

  return true;
}

SilentDetectConfigParser::SilentDetectConfigParser()
    : enable_(false),
      with_checksum_(false),
      grad_sample_interval_(kDefaultGradSampleInterval),
      upper_thresh1_(kDefaultUpperThresh1),
      upper_thresh2_(kDefaultUpperThresh2),
      cooldown_(kDefaultCooldown),
      strikes_num_(kDefaultStrikesNum),
      strikes_window_(kDefaultStrikesWindow),
      checksum_cooldown_(kDefaultChecksumCooldown) {
  Init();
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "SilentDetect initialized with enable: " << enable_
                                  << ", with_checksum: " << with_checksum_
                                  << ", grad_sample_interval: " << grad_sample_interval_
                                  << ", upper_thresh1: " << upper_thresh1_ << ", upper_thresh2: " << upper_thresh2_
                                  << ", cooldown: " << cooldown_ << "min, strikes_num: " << strikes_num_
                                  << ", strikes_window: " << strikes_window_
                                  << "min, checksum_cooldown: " << checksum_cooldown_ << "min.";
}

std::string SilentDetectConfigParser::Trim(const std::string &str) {
  size_t first = str.find_first_not_of(kSpaceChar);
  if (first == std::string::npos) {
    return "";
  }
  size_t last = str.find_last_not_of(kSpaceChar);
  return str.substr(first, (last - first + 1));
}

bool SilentDetectConfigParser::IsPositiveInteger(const std::string &str) {
  if (str.empty()) return false;
  for (char c : str) {
    if (!std::isdigit(c)) return false;
  }
  return str != "0";
}

bool SilentDetectConfigParser::IsIntegerGreaterEqual3(const std::string &str) {
  if (!IsPositiveInteger(str)) return false;
  try {
    int value = std::stoi(str);
    return value >= kNumber3;
  } catch (const std::exception &e) {
    return false;
  }
}

void SilentDetectConfigParser::ParseEnable(const ConfigMap &config_map) {
  auto it = config_map.find(kEnable);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (value == "true") {
#if defined(__linux__)
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto parallel_mode = parallel_context->parallel_mode();
    if (parallel_mode != parallel::kAutoParallel && parallel_mode != parallel::kSemiAutoParallel) {
      MS_LOG(WARNING) << "Silent detect supports '" << parallel::kAutoParallel << "' and '"
                      << parallel::kSemiAutoParallel << "' parallel_mode, but got '" << parallel_mode
                      << "'. It may not take effect.";
    }
#endif
    enable_ = true;
  } else if (value == "false") {
    enable_ = false;
  } else {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-enable value '" << value << "' is invalid, use the default value of "
                    << enable_ << ".";
  }
}

void SilentDetectConfigParser::ParseWithChecksum(const ConfigMap &config_map) {
#if defined(__linux__) && defined(WITH_BACKEND)
  auto it = config_map.find(kWithChecksum);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (value == "true") {
    with_checksum_ = true;
  } else if (value == "false") {
    with_checksum_ = false;
  } else {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-with_checksum value '" << value << "' is invalid, use the default value of "
                    << with_checksum_ << ".";
  }
#else
  MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-with_checksum only supported on linux platform, use the default value of "
                  << with_checksum_ << ".";
#endif
}

void SilentDetectConfigParser::ParseCooldown(const ConfigMap &config_map) {
  auto it = config_map.find(kCoolDownKey);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsPositiveInteger(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-cooldown value '" << value
                    << "' is not a positive integer, use the default value of " << cooldown_ << ".";
    return;
  }

  cooldown_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseStrikeNum(const ConfigMap &config_map) {
  auto it = config_map.find(kStrikesNumKey);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsPositiveInteger(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-strikes_num value '" << value
                    << "' is not a positive integer, use the default value of " << strikes_num_ << ".";
    return;
  }

  strikes_num_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseStrikeWindow(const ConfigMap &config_map) {
  auto it = config_map.find(kStrikesWindowKey);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsPositiveInteger(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-strikes_window value '" << value
                    << "' is not a positive integer, use the default value of " << strikes_window_ << ".";
    return;
  }

  strikes_window_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseChecksumCooldown(const ConfigMap &config_map) {
  auto it = config_map.find(kChecksumCooldownKey);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsPositiveInteger(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-checksum_cooldown value '" << value
                    << "' is not a positive integer, use the default value of " << checksum_cooldown_ << ".";
    return;
  }

  checksum_cooldown_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseUpperThresh1(const ConfigMap &config_map) {
  auto it = config_map.find(kUpperThresh1Key);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsIntegerGreaterEqual3(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-upper_thresh1 value '" << value << "' is invalid, use the default value of "
                    << upper_thresh1_ << ".";
    return;
  }

  upper_thresh1_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseUpperThresh2(const ConfigMap &config_map) {
  auto it = config_map.find(kUpperThresh2Key);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsIntegerGreaterEqual3(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-upper_thresh2 value '" << value << "' is invalid, use the default value of "
                    << upper_thresh2_ << ".";
    return;
  }

  upper_thresh2_ = std::stoi(value);
}

void SilentDetectConfigParser::ParseGradSampleInterval(const ConfigMap &config_map) {
  auto it = config_map.find(kGradSampleIntervalKey);
  if (it == config_map.end()) return;

  const std::string &value = it->second;
  if (!IsPositiveInteger(value)) {
    MS_LOG(WARNING) << "MS_NPU_ASD_CONFIG-grad_sample_interval value '" << value
                    << "' is not a positive integer, use the default value of " << grad_sample_interval_ << ".";
    return;
  }

  grad_sample_interval_ = std::stoi(value);
}

void SilentDetectConfigParser::Init() {
  std::string config_str = common::GetEnv(kMsNpuAsdConfig);
  if (!config_str.empty()) {
    config_str = Trim(config_str);
    if (!config_str.empty()) {
      ParseConfigs(config_str);
    }
  }
  config_func_ = {{kGradSampleIntervalKey, [this]() { return GetGradSampleInterval(); }},
                  {kUpperThresh1Key, [this]() { return GetUpperThresh1(); }},
                  {kUpperThresh2Key, [this]() { return GetUpperThresh2(); }},
                  {kCoolDownKey, [this]() { return GetCooldown(); }},
                  {kStrikesNumKey, [this]() { return GetStrikesNum(); }},
                  {kStrikesWindowKey, [this]() { return GetStrikesWindow(); }},
                  {kChecksumCooldownKey, [this]() { return GetChecksumCooldown(); }}};
}

int SilentDetectConfigParser::GetConfig(const std::string &name) {
  auto it = config_func_.find(name);
  if (it == config_func_.end()) {
    MS_EXCEPTION(ValueError) << "Config name '" << name << "' is not in silent detect config.";
  }
  return it->second();
}

std::string SilentDetectConfigParser::GetSilentDetectFeatureName(const std::string &name) {
  return std::string(kSilentDetectFeatureFlag) + name;
}

bool IsSilentDetectEnable() {
  auto &silentdetect = SilentDetectConfigParser::GetInstance();
  return silentdetect.IsEnable();
}

}  // namespace silentdetect
}  // namespace mindspore
