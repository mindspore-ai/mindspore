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

#ifndef MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECT_CONFIG_PARSER_H_
#define MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECT_CONFIG_PARSER_H_

#include <functional>
#include <map>
#include <string>
#include "tools/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace silentdetect {
constexpr auto kSilentDetectFeatureFlag = "<sdc-feature-value>";

using ConfigMap = std::map<std::string, std::string>;

class TOOLS_EXPORT SilentDetectConfigParser {
 public:
  static SilentDetectConfigParser &GetInstance() {
    static SilentDetectConfigParser instance;
    return instance;
  }

  SilentDetectConfigParser(const SilentDetectConfigParser &) = delete;
  SilentDetectConfigParser &operator=(const SilentDetectConfigParser &) = delete;

  bool IsEnable() const { return enable_; }
  bool IsWithChecksum() const { return with_checksum_; }
  int GetGradSampleInterval() const { return grad_sample_interval_; }
  int GetUpperThresh1() const { return upper_thresh1_; }
  int GetUpperThresh2() const { return upper_thresh2_; }
  int GetCooldown() const { return cooldown_; }
  int GetStrikesNum() const { return strikes_num_; }
  int GetStrikesWindow() const { return strikes_window_; }
  int GetChecksumCooldown() const { return checksum_cooldown_; }
  int GetConfig(const std::string &name);
  std::string GetSilentDetectFeatureName(const std::string &name);

 private:
  SilentDetectConfigParser();
  void Init();
  bool ParseConfigs(const std::string &);

  void ParseEnable(const ConfigMap &);
  void ParseWithChecksum(const ConfigMap &);
  void ParseCooldown(const ConfigMap &);
  void ParseStrikeNum(const ConfigMap &);
  void ParseStrikeWindow(const ConfigMap &);
  void ParseChecksumCooldown(const ConfigMap &);
  void ParseUpperThresh1(const ConfigMap &);
  void ParseUpperThresh2(const ConfigMap &);
  void ParseGradSampleInterval(const ConfigMap &);

  std::string Trim(const std::string &);
  bool IsPositiveInteger(const std::string &);
  bool IsIntegerGreaterEqual3(const std::string &);

  bool enable_;
  bool with_checksum_;
  int grad_sample_interval_;
  int upper_thresh1_;
  int upper_thresh2_;
  int cooldown_;
  int strikes_num_;
  int strikes_window_;
  int checksum_cooldown_;
  std::map<std::string, std::function<int()>> config_func_;
};

bool IsSilentDetectEnable();

}  // namespace silentdetect
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECT_CONFIG_PARSER_H_
