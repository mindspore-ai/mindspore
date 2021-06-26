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
#include "runtime/device/ascend/lic_manager.h"
#include <regex>
#include "utils/ms_context.h"
#include "runtime/dev.h"

namespace gelc {
uint32_t GetOptInfo(uint32_t, const std::string &, std::map<std::string, std::string> &);  // NOLINT
}  // namespace gelc

namespace mindspore {
namespace {
constexpr auto kFeKey = "opt_module.fe";
constexpr auto kOpTuneKey = "opt_module.op_tune";
constexpr auto kAllOpen = "ALL";

static const std::map<std::string, OptPassEnum> kPassCodeMap = {
  {std::to_string(3), OptPassEnum::MatmulBiasaddFusion},
  {std::to_string(8), OptPassEnum::DereluFusion},
  {std::to_string(9), OptPassEnum::TransposeReshapeFusion},
  {std::to_string(10), OptPassEnum::MomentumLossscaleFusion},
  {std::to_string(12), OptPassEnum::FusedBatchNormFusion},
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
  {std::to_string(38), OptPassEnum::SoftmaxGradExtFusion},
  {std::to_string(39), OptPassEnum::ClipByNormNoDivSquareSumFusion},
};

inline std::vector<std::string> SplitStrByRegex(const std::string &str, const std::string &regex) {
  std::regex split(regex);
  return std::vector<std::string>(std::sregex_token_iterator(str.begin(), str.end(), split, -1),
                                  std::sregex_token_iterator());
}

static std::string GetSocVersion() {
  constexpr int kSocVersionLen = 50;
  char soc_version[kSocVersionLen] = {0};
  auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(WARNING) << "rtGetSocVersion failed, ret = " << ret;
    return "Ascend910";
  }

  return soc_version;
}
}  // namespace

LicManager &LicManager::GetInstance() {
  static LicManager lic_manager{};
  return lic_manager;
}

bool LicManager::GetPassSwitch(OptPassEnum pass) {
  if (!init_flag) {
    ParseSwitch();
  }
  auto iter = pass_switch_.find(pass);
  if (iter == pass_switch_.end()) {
    return true;
  }

  return iter->second;
}

void LicManager::ParseSwitch() {
  std::map<std::string, std::string> opt_info_map;
  auto ret = gelc::GetOptInfo(0, GetSocVersion(), opt_info_map);
  if (ret != 0) {
    MS_LOG(WARNING) << "GetOptInfo failed.";
    init_flag = true;
    return;
  }

  auto iter = opt_info_map.find(kFeKey);
  if (iter != opt_info_map.end()) {
    ParseFeSwitch(iter->second);
  }

  init_flag = true;
}

void LicManager::ParseFeSwitch(const std::string &options_str) {
  // invalid options, do nothing.
  if (options_str.empty()) {
    return;
  }
  // "All" in options means all open, do nothing.
  if (options_str.find(kAllOpen) != std::string::npos) {
    return;
  }

  for (auto iter = kPassCodeMap.begin(); iter != kPassCodeMap.end(); ++iter) {
    auto pass = iter->second;
    pass_switch_.emplace(pass, false);
  }
  auto fe_pass = SplitStrByRegex(options_str, ":");
  for (auto &pass_code : fe_pass) {
    auto iter = kPassCodeMap.find(pass_code);
    if (iter != kPassCodeMap.end()) {
      pass_switch_[iter->second] = true;
    }
  }
}
}  // namespace mindspore
