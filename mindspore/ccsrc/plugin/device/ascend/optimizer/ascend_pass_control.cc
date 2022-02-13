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
#include "plugin/device/ascend/optimizer/ascend_pass_control.h"
#include "mindspore/core/utils/ms_utils.h"

namespace mindspore::opt {
namespace {
constexpr char kMsAscendFusionSwitch[] = "MS_DEV_ASCEND_FUSION_SWITCH";
}  // namespace

bool PassWithSwitch::Run(const FuncGraphPtr &func_graph) {
  if (!PassSwitchManager::GetInstance().GetPassSwitch(name())) {
    return false;  // false means no changed
  }

  return RunPass(func_graph);
}

AnfNodePtr PatternProcessPassWithSwitch::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (!PassSwitchManager::GetInstance().GetPassSwitch(name())) {
    return nullptr;  // nullptr means no changed
  }

  return PatternProcessPass::Run(func_graph, node);
}

PassSwitchManager::PassSwitchManager() { SetSwitchFromEnv(); }

PassSwitchManager &PassSwitchManager::GetInstance() {
  static PassSwitchManager instance{};
  return instance;
}

bool PassSwitchManager::GetPassSwitch(const std::string &pass_name) const {
  if (auto iter = env_pass_switch_.find(pass_name); iter != env_pass_switch_.end() && !env_switch_) {
    return false;
  }

  if (!LicManager::GetInstance().GetPassSwitch(GetPassEnum(pass_name))) {
    return false;
  }

  return true;
}

enum OptPassEnum PassSwitchManager::GetPassEnum(const std::string &pass_name) const {
  if (auto iter = pass_enum_map_.find(pass_name); iter != pass_enum_map_.end()) {
    return iter->second;
  }

  return OptPassEnum::Invalid;
}

void PassSwitchManager::SetSwitchFromEnv() {
  auto sw_env = common::GetEnv(kMsAscendFusionSwitch);
  env_switch_ = (sw_env != "OFF" && sw_env != "off" && sw_env != "0");
}
}  // namespace mindspore::opt
