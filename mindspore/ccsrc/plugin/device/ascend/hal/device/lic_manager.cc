/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/lic_manager.h"
#include <regex>
#include "utils/ms_context.h"
#include "runtime/dev.h"
#include "opt_info/opt_info.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace {
constexpr auto kFeKey = "opt_module.fe";
constexpr auto kOpTuneKey = "opt_module.op_tune";
constexpr auto kPassKey = "opt_module.pass";
constexpr auto kRlTuneKey = "opt_module.rl_tune";
constexpr auto kAllOpen = "ALL";

static const std::map<std::string, OptPassEnum> kPassCodeMap = {
  {std::to_string(3), OptPassEnum::MatmulBiasaddFusion},
  {std::to_string(9), OptPassEnum::TransposeReshapeFusion},
  {std::to_string(15), OptPassEnum::BnupdateEltwiseEltwiseFusionPass},
  {std::to_string(16), OptPassEnum::BnupdateEltwiseFusionPass},
  {std::to_string(17), OptPassEnum::Conv2DBackpropEltwiseFusionPass},
  {std::to_string(18), OptPassEnum::ConvBnReduceFusionPass},
  {std::to_string(26), OptPassEnum::ReshapeTransposeFusion},
  {std::to_string(27), OptPassEnum::SquareSumFusion},
  {std::to_string(30), OptPassEnum::MatmulEltwiseFusionPass},
  {std::to_string(33), OptPassEnum::BatchMatmulFusedMulAddFusionPass},
  {std::to_string(34), OptPassEnum::EltwiseFusionPass},
  {std::to_string(36), OptPassEnum::MultiOutputFusionPass},
  {std::to_string(37), OptPassEnum::MulAddFusion},
  {std::to_string(39), OptPassEnum::ClipByNormNoDivSquareSumFusion},
  {std::to_string(42), OptPassEnum::MulAddNPass},
  {std::to_string(43), OptPassEnum::Resnet50DbnDwFusionPass},
  {std::to_string(44), OptPassEnum::BatchMatMulDropOutDoMaskV3DFusionPass},
  {std::to_string(45), OptPassEnum::MatmulConfusiontransposeUbFusion},
  {std::to_string(46), OptPassEnum::MatMulDropOutDoMaskV3DFusionPass},
  {std::to_string(47), OptPassEnum::TbeBatchMatmulElementWiseFusionPass},
};

inline std::vector<std::string> SplitStrByRegex(const std::string &str, const std::string &regex) {
  std::regex split(regex);
  return std::vector<std::string>(std::sregex_token_iterator(str.begin(), str.end(), split, -1),
                                  std::sregex_token_iterator());
}
}  // namespace

LicManager::LicManager() { ParseSwitch(); }

LicManager &LicManager::GetInstance() {
  static LicManager lic_manager{};
  return lic_manager;
}

bool LicManager::GetPassSwitch(OptPassEnum pass) const {
  auto iter = pass_switch_.find(pass);
  if (iter == pass_switch_.end()) {
    return true;
  }

  return iter->second;
}

void LicManager::ParseSwitch() {
  std::map<std::string, std::string> opt_info_map;
  auto ret = gelc::GetOptInfo(0, device::ascend::GetSocVersion(), opt_info_map);
  if (ret != 0) {
    MS_LOG(WARNING) << "GetOptInfo failed.";
    return;
  }

  ParseFeSwitch(opt_info_map);
  ParseOpTuneSwitch(opt_info_map);
  ParsePassSwitch(opt_info_map);
  ParseRlSwitch(opt_info_map);
}

void LicManager::ParseFeSwitch(const std::map<std::string, std::string> &options_map) {
  // no fe switch, open all
  auto options_iter = options_map.find(kFeKey);
  if (options_iter == options_map.end()) {
    return;
  }

  // "All" in options means all open, do nothing.
  const auto &options_str = options_iter->second;
  if (options_str.find(kAllOpen) != std::string::npos) {
    return;
  }

  // close all first
  for (auto iter = kPassCodeMap.begin(); iter != kPassCodeMap.end(); ++iter) {
    auto pass = iter->second;
    (void)pass_switch_.emplace(pass, false);
  }

  // then open passes in options
  auto fe_pass = SplitStrByRegex(options_str, ":");
  for (auto &pass_code : fe_pass) {
    auto iter = kPassCodeMap.find(pass_code);
    if (iter != kPassCodeMap.end()) {
      pass_switch_[iter->second] = true;
    }
  }
}

void LicManager::ParseOpTuneSwitch(const std::map<std::string, std::string> &options_map) {
  auto options_iter = options_map.find(kOpTuneKey);
  if (options_iter == options_map.end()) {
    op_tune_switch_ = "null";
    return;
  }

  const auto &op_tune_str = options_iter->second;
  if (op_tune_str.empty()) {
    op_tune_switch_ = "off";
    op_tune_list_.clear();
  } else {
    op_tune_switch_ = "on";
    op_tune_list_ = op_tune_str;
  }
}

void LicManager::ParsePassSwitch(const std::map<std::string, std::string> &options_map) {
  auto options_iter = options_map.find(kPassKey);
  if (options_iter == options_map.end()) {
    pass_list_ = "invalid";
    return;
  }

  const auto &pass_str = options_iter->second;
  if (!pass_str.empty()) {
    pass_list_ = pass_str;
  }
}

void LicManager::ParseRlSwitch(const std::map<std::string, std::string> &options_map) {
  auto options_iter = options_map.find(kRlTuneKey);
  if (options_iter == options_map.end()) {
    rl_tune_switch_ = "null";
    return;
  }
  const auto &rl_tune_str = options_iter->second;
  if (rl_tune_str.empty()) {
    rl_tune_switch_ = "off";
    rl_tune_list_.clear();
  } else {
    rl_tune_switch_ = "on";
    rl_tune_list_ = rl_tune_str;
  }
}
}  // namespace mindspore
