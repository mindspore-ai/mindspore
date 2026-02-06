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
#include "tools/error_handler/error_config.h"
#include <mutex>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <utility>
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "include/utils/callback.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace tools {
namespace {
// training fault tolerance config env var 'MS_ENABLE_TFT'
constexpr char kMsEnableTft[] = "MS_ENABLE_TFT";
constexpr char kTftKeyUce[] = "UCE";
constexpr char kTftKeyHcce[] = "HCCE";
constexpr char kTftKeyArf[] = "ARF";
constexpr char kTftKeyRsc[] = "RSC";
constexpr char kTftKeyTre[] = "TRE";
constexpr char kTftKeyTreSnapShotSteps[] = "TRE_SNAPSHOT_STEPS";
constexpr char kTftValueEnable[] = "1";
constexpr char kTftValueNormalTRE[] = "1";
constexpr char kTftValueStepTRE[] = "2";
constexpr size_t kTftSubItemNumElems = 2;

// Trim from start (in place)
void LTrim(std::string *s, unsigned char extra = '\0') {
  s->erase(s->begin(), std::find_if(s->begin(), s->end(),
                                    [extra](unsigned char ch) { return !std::isspace(ch) && (ch != extra); }));
}

// Trim from end (in place)
void RTrim(std::string *s, unsigned char extra = '\0') {
  s->erase(
    std::find_if(s->rbegin(), s->rend(), [extra](unsigned char ch) { return !std::isspace(ch) && (ch != extra); })
      .base(),
    s->end());
}

// Trim from both ends (in place)
std::string &Trim(std::string *s) {
  LTrim(s);
  RTrim(s);
  return *s;
}

std::vector<std::string> SplitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::istringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}
}  // namespace

inline nlohmann::json ParserJson(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  } else if (py::isinstance<py::bool_>(obj)) {
    return obj.cast<bool>();
  } else if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int64_t>();
  } else if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  } else if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    nlohmann::json j_array = nlohmann::json::array();
    py::list list = py::cast<py::list>(py::reinterpret_borrow<py::list>(obj));
    for (size_t i = 0; i < list.size(); ++i) {
      j_array.push_back(ParserJson(list[i]));
    }
    return j_array;
  } else if (py::isinstance<py::dict>(obj)) {
    nlohmann::json j_dict = nlohmann::json::object();
    for (const auto &item : obj.attr("items")()) {
      auto pair = py::cast<std::pair<py::object, py::object>>(item);
      auto key = pair.first.cast<std::string>();
      j_dict[key] = ParserJson(pair.second);
    }
    return j_dict;
  } else {
    // unsupported type .....
    MS_LOG(EXCEPTION) << "Unsupported data type: " << obj;
  }
}

std::shared_ptr<TftConfig> TftConfig::GetInstance() {
  static std::once_flag parser_init_flag_ = {};
  static std::shared_ptr<TftConfig> inst_parser_ = nullptr;
  std::call_once(parser_init_flag_, [&]() {
    if (inst_parser_ == nullptr) {
      MS_LOG(DEBUG) << "Create new tft parser instance";
      inst_parser_ = std::make_shared<TftConfig>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_parser_);
  return inst_parser_;
}

void TftConfig::RegisterConfig(const py::object &configs) { config_json_ = ParserJson(configs); }

bool TftConfig::IsEnableWatchdog() {
  static auto watchdog = ([this]() -> bool {
    auto context = MsContext::GetInstance();
    if (context != nullptr && !context->get_param<bool>(MS_CTX_ENABLE_HCCL_WATCHDOG)) {
      return false;
    }
    return CheckSupport(kWatchdog, true);
  })();
  return watchdog;
}

bool TftConfig::IsEnableSaveHcclOpStatus() {
  static bool ccae = CheckSupport(kStatusRecord, false);
  return ccae;
}

bool TftConfig::CheckSupport(const std::string &key, bool def_value) {
  if (mark_check_.count(key) != 0) {
    return mark_check_[key];
  }
  auto v = GetConfigValue<std::string>(key, "");
  MS_LOG(INFO) << "Get value of ' " << key << " ' is: " << v;
  auto ret = v == "" ? def_value : v == "1";
  mark_check_[key] = ret;
  return ret;
}

bool TftConfig::IsEnableTRE() {
  static bool enable_tre = []() {
    auto iter = GetConfigMap().find(kTftKeyTre);
    if (iter == GetConfigMap().end()) {
      MS_VLOG(VL_UCE) << "Can find `" << kTftKeyTre << "` in environment var `" << kMsEnableTft << "`";
      return false;
    }
    if (iter->second == kTftValueNormalTRE) {
      return true;
    }
    return IsEnableStepTRE();
  }();
  return enable_tre;
}

bool TftConfig::IsEnableStepTRE() {
  static bool enable_step_tre = []() {
    auto iter = GetConfigMap().find(kTftKeyTre);
    if (iter == GetConfigMap().end()) {
      MS_VLOG(VL_UCE) << "Can find `" << kTftKeyTre << "` in environment var `" << kMsEnableTft << "`";
      return false;
    }
    if (iter->second != kTftValueStepTRE) {
      MS_VLOG(VL_UCE) << "Value of `" << kTftKeyTre << "` is `" << iter->second << "`, step tre is not enabled.";
      return false;
    }
    if (GetSnapShotSteps() <= 0) {
      MS_VLOG(VL_UCE) << "Value of `" << kTftKeyTreSnapShotSteps << "` is `" << GetSnapShotSteps()
                      << "`, step tre is not enabled.";
      return false;
    }
    return true;
  }();
  return enable_step_tre;
}

int TftConfig::GetSnapShotSteps() {
  static int mem_cfg_steps = []() {
    auto iter = GetConfigMap().find(kTftKeyTreSnapShotSteps);
    if (iter == GetConfigMap().end()) {
      MS_VLOG(VL_UCE) << "Can find `" << kTftKeyTreSnapShotSteps << "` in environment var `" << kMsEnableTft << "`";
      return 0;
    }
    try {
      return std::stoi(iter->second);
    } catch (std::invalid_argument const &ex) {
      MS_LOG(WARNING) << "Try to convert value " << iter->second << " for key " << kTftKeyTreSnapShotSteps
                      << " failed, what(): " << ex.what();
      return 0;
    } catch (std::out_of_range const &ex) {
      MS_LOG(WARNING) << "Try to convert value " << iter->second << " for key " << kTftKeyTreSnapShotSteps
                      << " failed, what(): " << ex.what();
      return 0;
    }
  }();
  return mem_cfg_steps;
}

bool TftConfig::IsEnableUCE() {
  static bool is_enable_uce = IsEnableFeature(kTftKeyUce);
  return is_enable_uce && mindspore::IsGraphPipelineCompiled();
}

bool TftConfig::IsEnableHCCE() {
  static bool is_enable_hcce = IsEnableFeature(kTftKeyHcce);
  return is_enable_hcce && mindspore::IsGraphPipelineCompiled();
}

bool TftConfig::IsEnableARF() {
  static bool is_enable_arf = IsEnableFeature(kTftKeyArf);
  return is_enable_arf && mindspore::IsGraphPipelineCompiled();
}

bool TftConfig::IsEnableRsc() {
  static bool is_enable_rsc = IsEnableFeature(kTftKeyArf) || IsEnableFeature(kTftKeyRsc);
  return is_enable_rsc && mindspore::IsGraphPipelineCompiled();
}

std::map<std::string, std::string> &TftConfig::GetConfigMap() {
  static std::map<std::string, std::string> configs;
  static bool config_parsed = false;
  if (config_parsed) {
    return configs;
  }

  // parse value of MS_ENABLE_TFT in format '{key1:value1, key2:value2}'
  static std::once_flag flag;
  std::call_once(
    flag,
    [](std::map<std::string, std::string> &configs, bool &config_parsed) {
      auto tft_config = common::GetEnv(kMsEnableTft);
      MS_LOG(INFO) << "Value of `" << kMsEnableTft << "` is `" << tft_config << "`";
      MS_VLOG(VL_UCE) << "Value of `" << kMsEnableTft << "` is `" << tft_config << "`";
      if (tft_config.empty()) {
        config_parsed = true;
        return;
      }
      // trim left '{' and right '}'
      LTrim(&tft_config, '{');
      RTrim(&tft_config, '}');
      MS_LOG(INFO) << "Trimmed value of `" << kMsEnableTft << "` is `" << tft_config << "`";

      std::vector<std::string> config_items = SplitString(tft_config, ',');
      for (const auto &cfg_item : config_items) {
        auto elems = SplitString(cfg_item, ':');
        if (elems.size() != kTftSubItemNumElems) {
          MS_LOG(WARNING) << "Ignore illegal item `" << cfg_item << "` in " << kMsEnableTft;
        }
        auto &key = Trim(&elems[0]);
        auto &value = Trim(&elems[1]);
        configs[key] = value;
        MS_LOG(INFO) << "Insert key `" << key << "` with value `" << value << "`";
        MS_VLOG(VL_UCE) << "Insert key `" << key << "` with value `" << value << "`";
      }

      config_parsed = true;
    },
    configs, config_parsed);

  return configs;
}

bool TftConfig::IsEnableFeature(const std::string &feature_name) {
  auto iter = GetConfigMap().find(feature_name);
  if (iter == GetConfigMap().end()) {
    return false;
  }
  return iter->second == kTftValueEnable;
}

bool IsEnableUce() { return mindspore::tools::TftConfig::GetInstance()->IsEnableUCE(); }

bool IsEnableArf() { return mindspore::tools::TftConfig::GetInstance()->IsEnableARF(); }

bool IsTftDisableHighPerfMode() {
  auto tft_inst = tools::TftConfig::GetInstance();
  return tft_inst->IsEnableUCE() || tft_inst->IsEnableStepTRE() || tft_inst->IsEnableSaveHcclOpStatus() ||
         tft_inst->IsEnableARF();
}

REGISTER_COMMON_CALLBACK(IsEnableUce);
REGISTER_COMMON_CALLBACK(IsEnableArf);
REGISTER_COMMON_CALLBACK(IsTftDisableHighPerfMode);
}  // namespace tools
}  // namespace mindspore
