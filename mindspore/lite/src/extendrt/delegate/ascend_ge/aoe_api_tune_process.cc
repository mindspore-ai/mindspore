/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/aoe_api_tune_process.h"
#include <cstdio>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/src/extendrt/cxx_api/dlutils.h"
#include "mindspore/ccsrc/utils/dlopen_macro.h"
#include "mindspore/ccsrc/cxx_api/acl_utils.h"

namespace mindspore {
namespace {
constexpr const char *kSubgraphTurning = "subgraph tuning";
constexpr const char *kOperatorTurning = "operator tuning";
constexpr const char *kSubgraphTurningIndex = "1";
constexpr const char *kOperatorTurningIndex = "2";

const std::map<std::string, std::string> kTuneModeMap = {{"1", "subgraph tuning"}, {"2", "operator tuning"}};
}  // namespace

using AoeStatus = int32_t;
constexpr AoeStatus AOE_SUCCESS = 0;

class AoePlugin {
 public:
  static AoePlugin &Instance() {
    static AoePlugin instance;
    return instance;
  }
  AoePlugin() = default;
  ~AoePlugin() { DLSoClose(handle_); }
  bool LoadAoePlugin() {
    if (handle_ != nullptr) {
      return true;
    }
    std::string aoe_so_name = "libaoe_tuning.so";
    std::string aoe_init_func_name = "AoeInitialize";
    std::string aoe_finalize_func_name = "AoeFinalize";
    std::string aoe_create_session_func_name = "AoeCreateSession";
    std::string aoe_destroy_session_func_name = "AoeDestroySession";
    std::string aoe_set_ge_session_func_name = "AoeSetGeSession";
    std::string aoe_set_tuning_graph_func_name = "AoeSetTuningGraph";
    std::string aoe_set_tuning_graph_input_func_name = "AoeSetTuningGraphInput";
    std::string aoe_tuning_graph_func_name = "AoeTuningGraph";

    auto status = DLSoOpen(aoe_so_name, "", &handle_, nullptr);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Dlopen " << aoe_so_name << " failed, result = " << status.ToString();
      return false;
    }
    try {
      aoe_initialize_func_ = DlsymWithCast<AoeInitializeFunc>(handle_, aoe_init_func_name.c_str());
      aoe_finalize_func_ = DlsymWithCast<AoeFinalizeFunc>(handle_, aoe_finalize_func_name.c_str());
      aoe_create_session_func_ = DlsymWithCast<AoeCreateSessionFunc>(handle_, aoe_create_session_func_name.c_str());
      aoe_destroy_session_func_ = DlsymWithCast<AoeDestroySessionFunc>(handle_, aoe_destroy_session_func_name.c_str());
      aoe_set_ge_session_func_ = DlsymWithCast<AoeSetGeSessionFunc>(handle_, aoe_set_ge_session_func_name.c_str());
      aoe_set_tuning_graph_func_ =
        DlsymWithCast<AoeSetTuningGraphFunc>(handle_, aoe_set_tuning_graph_func_name.c_str());
      aoe_set_tuning_graph_input_func_ =
        DlsymWithCast<AoeSetTuningGraphInputFunc>(handle_, aoe_set_tuning_graph_input_func_name.c_str());
      aoe_tuning_graph_func_ = DlsymWithCast<AoeTuningGraphFunc>(handle_, aoe_tuning_graph_func_name.c_str());
    } catch (const std::runtime_error &error) {
      MS_LOG(ERROR) << "Failed to load symbol from " << aoe_so_name;
      return false;
    }
    return true;
  }
  bool AoeInitialize(const std::map<std::string, std::string> &global_options) {
    if (aoe_initialize_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_initialize_func_ is nullptr";
      return false;
    }
    std::map<ge::AscendString, ge::AscendString> options;
    for (auto &item : global_options) {
      MS_LOG(INFO) << "Aoe global option " << item.first << " = " << item.second;
      options[ge::AscendString(item.first.c_str())] = ge::AscendString(item.second.c_str());
    }
    auto aoe_status = aoe_initialize_func_(options);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeInitialize, ret: " << aoe_status;
      return false;
    }
    return true;
  }
  void AoeFinalize() {
    if (aoe_finalize_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_finalize_func_ is nullptr";
      return;
    }
    aoe_finalize_func_();
  }
  bool AoeCreateSession(uint64_t *session_id) {
    if (session_id == nullptr) {
      MS_LOG(ERROR) << "Input parameter session_id cannot be nullptr";
      return false;
    }
    if (aoe_create_session_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_create_session_func_ is nullptr";
      return false;
    }
    auto aoe_status = aoe_create_session_func_(*session_id);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeCreateSession, ret: " << aoe_status;
      return false;
    }
    return true;
  }
  void AoeDestroySession(uint64_t session_id) {
    if (aoe_destroy_session_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_destroy_session_func_ is nullptr";
      return;
    }
    auto aoe_status = aoe_destroy_session_func_(session_id);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeDestroySession, ret: " << aoe_status;
      return;
    }
  }
  bool AoeSetGeSession(uint64_t session_id, ge::Session *ge_session) {
    if (aoe_set_ge_session_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_set_ge_session_func_ is nullptr";
      return false;
    }
    auto aoe_status = aoe_set_ge_session_func_(session_id, ge_session);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeSetGeSession, ret: " << aoe_status;
      return false;
    }
    return true;
  }
  bool AoeSetTuningGraph(uint64_t session_id, const ge::Graph &ge_graph) {
    if (aoe_set_tuning_graph_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_set_tuning_graph_func_ is nullptr";
      return false;
    }
    auto aoe_status = aoe_set_tuning_graph_func_(session_id, ge_graph);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeSetTuningGraph, ret: " << aoe_status;
      return false;
    }
    return true;
  }
  bool AoeSetTuningGraphInput(uint64_t session_id, const std::vector<ge::Tensor> &inputs) {
    if (aoe_set_tuning_graph_input_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_set_tuning_graph_input_func_ is nullptr";
      return false;
    }
    auto aoe_status = aoe_set_tuning_graph_input_func_(session_id, inputs);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeSetTuningGraphInput, ret: " << aoe_status;
      return false;
    }
    return true;
  }
  bool AoeTuningGraph(uint64_t session_id, const std::map<std::string, std::string> &tuning_options) {
    if (aoe_tuning_graph_func_ == nullptr) {
      MS_LOG(ERROR) << "aoe_tuning_graph_func_ is nullptr";
      return false;
    }
    std::map<ge::AscendString, ge::AscendString> options;
    for (auto &item : tuning_options) {
      MS_LOG(INFO) << "Aoe tuning option " << item.first << " = " << item.second;
      options[ge::AscendString(item.first.c_str())] = ge::AscendString(item.second.c_str());
    }
    auto aoe_status = aoe_tuning_graph_func_(session_id, options);
    if (aoe_status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call AoeTuningGraph, ret: " << aoe_status;
      return false;
    }
    return true;
  }

 private:
  using AoeInitializeFunc = AoeStatus (*)(const std::map<ge::AscendString, ge::AscendString> &);
  using AoeFinalizeFunc = AoeStatus (*)();
  using AoeCreateSessionFunc = AoeStatus (*)(uint64_t &);
  using AoeDestroySessionFunc = AoeStatus (*)(uint64_t);
  using AoeSetGeSessionFunc = AoeStatus (*)(uint64_t, ge::Session *);
  using AoeSetTuningGraphFunc = AoeStatus (*)(uint64_t, const ge::Graph &);
  using AoeSetTuningGraphInputFunc = AoeStatus (*)(uint64_t, const std::vector<ge::Tensor> &);
  using AoeTuningGraphFunc = AoeStatus (*)(uint64_t, const std::map<ge::AscendString, ge::AscendString> &);

  AoeInitializeFunc aoe_initialize_func_ = nullptr;
  AoeFinalizeFunc aoe_finalize_func_ = nullptr;
  AoeCreateSessionFunc aoe_create_session_func_ = nullptr;
  AoeDestroySessionFunc aoe_destroy_session_func_ = nullptr;
  AoeSetGeSessionFunc aoe_set_ge_session_func_ = nullptr;
  AoeSetTuningGraphFunc aoe_set_tuning_graph_func_ = nullptr;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input_func_ = nullptr;
  AoeTuningGraphFunc aoe_tuning_graph_func_ = nullptr;

  void *handle_ = nullptr;
};

Status AoeApiTuning::ExecuteAoe(const std::shared_ptr<ge::Session> &session, const transform::DfGraphPtr &graph,
                                const std::vector<ge::Tensor> &inputs, const std::vector<std::string> &job_types,
                                const std::map<std::string, std::string> &global_options,
                                const std::map<std::string, std::string> &tuning_options) {
  MS_LOG(INFO) << "Start to aoe.";
  try {
    auto &aoe_instance = AoePlugin::Instance();
    if (!aoe_instance.LoadAoePlugin()) {
      return kLiteError;
    }
    for (auto &job_type : job_types) {
      std::cout << "Start to " << kTuneModeMap.at(job_type) << std::endl;
      std::map<std::string, std::string> global_options_new = global_options;
      global_options_new["job_type"] = job_type;
      if (!aoe_instance.AoeInitialize(global_options_new)) {
        return kLiteError;
      }
      uint64_t session_id = 0;
      if (!aoe_instance.AoeCreateSession(&session_id)) {
        aoe_instance.AoeFinalize();
        return kLiteError;
      }
      if (session && !aoe_instance.AoeSetGeSession(session_id, session.get())) {
        aoe_instance.AoeDestroySession(session_id);
        aoe_instance.AoeFinalize();
        return kLiteError;
      }
      if (!aoe_instance.AoeSetTuningGraph(session_id, *graph)) {
        aoe_instance.AoeDestroySession(session_id);
        aoe_instance.AoeFinalize();
        return kLiteError;
      }
      if (!inputs.empty() && !aoe_instance.AoeSetTuningGraphInput(session_id, inputs)) {
        aoe_instance.AoeDestroySession(session_id);
        aoe_instance.AoeFinalize();
        return kLiteError;
      }
      if (!aoe_instance.AoeTuningGraph(session_id, tuning_options)) {
        aoe_instance.AoeDestroySession(session_id);
        aoe_instance.AoeFinalize();
        return kLiteError;
      }
      aoe_instance.AoeDestroySession(session_id);
      aoe_instance.AoeFinalize();
      std::cout << "End " << kTuneModeMap.at(job_type) << std::endl;
    }
    return kSuccess;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Execute aoe failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "Execute aoe failed.";
  }
  return kMCFailed;
}

static std::shared_ptr<AscendDeviceInfo> GetAscendDeviceInfo(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    return nullptr;
  }
  auto &device_infos = context->MutableDeviceInfo();
  if (device_infos.size() != 1 || device_infos[0] == nullptr) {
    return nullptr;
  }
  auto ascend_info = device_infos[0]->Cast<AscendDeviceInfo>();
  if (ascend_info == nullptr) {
    return nullptr;
  }
  return ascend_info;
}

static void SetOption(std::map<std::string, std::string> *aoe_options, const std::string &key,
                      const std::map<std::string, std::string> &config_options, std::string get_key = "") {
  if (get_key.empty()) {
    get_key = key;
  }
  auto it = config_options.find(get_key);
  if (it == config_options.end()) {
    return;
  }
  (*aoe_options)[key] = it->second;
}

static void SetOption(std::map<std::string, std::string> *aoe_options, const std::string &key,
                      const std::function<std::string()> &func) {
  auto option = func();
  if (!option.empty()) {
    (*aoe_options)[key] = option;
  }
}

std::map<std::string, std::string> AoeApiTuning::GetAoeGlobalOptions(const std::shared_ptr<Context> &context,
                                                                     const ConfigInfos &config_infos) {
  // framework, device, precision_mode
  std::map<std::string, std::string> aoe_options;
  aoe_options["framework"] = "1";
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("precision_mode");
    if (option_it != options.end()) {
      aoe_options["precision_mode"] = TransforPrecisionToAcl(option_it->second);
    }
  }
  // get options from AscendDeviceInfo: may parse from [ascend_context] & [acl_option_cfg_param]
  auto ascend_info = GetAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(WARNING) << "Failed to get ascend device info from context";
    return {};
  }
  aoe_options["device"] = std::to_string(ascend_info->GetDeviceID());
  auto precision_mode = ascend_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    aoe_options["precision_mode"] = TransforPrecisionToAcl(precision_mode);
  }
  // get options from [ge_session_options]
  section_it = config_infos.find(lite::kGeSessionOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    SetOption(&aoe_options, "precision_mode", options, "ge.exec.precision_mode");
  }
  // get options from [ge_graph_options]
  section_it = config_infos.find(lite::kGeGraphOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    SetOption(&aoe_options, "precision_mode", options, "ge.exec.precision_mode");
  }
  // get options from [aoe_global_options]
  section_it = config_infos.find(lite::kAoeGlobalOptionsSection);
  if (section_it != config_infos.end()) {
    for (auto &option_item : section_it->second) {
      aoe_options[option_item.first] = option_item.second;
      MS_LOG(INFO) << "Update global option " << option_item.first << " to " << option_item.second;
    }
  }
  return aoe_options;
}

std::map<std::string, std::string> AoeApiTuning::GetAoeTuningOptions(const std::shared_ptr<Context> &context,
                                                                     const ConfigInfos &config_infos) {
  // input_shape，dynamic_batch_size, dynamic_image_size, dynamic_dims & input_format
  std::map<std::string, std::string> aoe_options;
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    SetOption(&aoe_options, "input_shape", options);
    SetOption(&aoe_options, "input_format", options);
    SetOption(&aoe_options, "dynamic_batch_size", options);
    SetOption(&aoe_options, "dynamic_image_size", options);
    SetOption(&aoe_options, "dynamic_dims", options);
  }
  // get options for AscendDeviceInfo: may parse from [ascend_context] & [acl_option_cfg_param]
  auto ascend_info = GetAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(WARNING) << "Failed to get ascend device info from context";
    return {};
  }
  SetOption(&aoe_options, "input_format", [ascend_info]() { return ascend_info->GetInputFormat(); });
  SetOption(&aoe_options, "input_shape", [ascend_info]() { return ascend_info->GetInputShape(); });
  SetOption(&aoe_options, "dynamic_batch_size", [ascend_info]() { return ascend_info->GetDynamicBatchSize(); });
  SetOption(&aoe_options, "dynamic_image_size", [ascend_info]() { return ascend_info->GetDynamicImageSize(); });

  // get options from [ge_graph_options]
  section_it = config_infos.find(lite::kGeGraphOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    SetOption(&aoe_options, "input_shape", options, "ge.inputShape");
    SetOption(&aoe_options, "dynamic_dims", options, "ge.dynamicDims");
  }
  // get options from [aoe_tuning_options]
  section_it = config_infos.find(lite::kAoeTuningOptionsSection);
  if (section_it != config_infos.end()) {
    for (auto &option_item : section_it->second) {
      aoe_options[option_item.first] = option_item.second;
      MS_LOG(INFO) << "Update tuning option " << option_item.first << " to " << option_item.second;
    }
  }
  if ((aoe_options.find("dynamic_batch_size") != aoe_options.end() ||
       aoe_options.find("dynamic_dims") != aoe_options.end()) &&
      aoe_options.find("input_format") == aoe_options.end()) {
    aoe_options["input_format"] = "ND";
  }
  return aoe_options;
}

std::vector<std::string> AoeApiTuning::GetAoeJobType(const std::shared_ptr<Context> &context,
                                                     const ConfigInfos &config_infos) {
  std::vector<std::string> job_types;
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("aoe_mode");
    if (option_it != options.end()) {
      auto &option = option_it->second;
      if (option.find(kOperatorTurning) != std::string::npos) {
        job_types.push_back(kOperatorTurningIndex);
      }
      if (option.find(kSubgraphTurning) != std::string::npos) {
        job_types.push_back(kSubgraphTurningIndex);
      }
    }
  }
  // get options from [ascend_context]
  section_it = config_infos.find(lite::kAscendContextSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("aoe_mode");
    if (option_it != options.end()) {
      job_types.clear();
      auto &option = option_it->second;
      if (option.find(kOperatorTurning) != std::string::npos) {
        job_types.push_back(kOperatorTurningIndex);
      }
      if (option.find(kSubgraphTurning) != std::string::npos) {
        job_types.push_back(kSubgraphTurningIndex);
      }
    }
  }
  if (job_types.size() > 1) {
    MS_LOG(ERROR) << "Config aoe_mode should only be " << kOperatorTurning << " or " << kSubgraphTurning
                  << " when provider is ge";
    return {};
  }
  // get options from [aoe_global_options]
  section_it = config_infos.find(lite::kAoeGlobalOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("job_type");
    if (option_it != options.end()) {
      auto option = option_it->second;
      if (option != kSubgraphTurningIndex && option != kOperatorTurningIndex) {
        MS_LOG(ERROR) << "Config job_type should only be " << kOperatorTurningIndex << " or " << kSubgraphTurningIndex
                      << " when provider is ge";
        return {};
      }
      job_types.clear();
      job_types.push_back(option_it->second);
    }
  }
  if (job_types.empty()) {
    MS_LOG(ERROR) << "Option aoe_mode or job_type is not set, option aoe_mode should be in section "
                  << lite::kAclOptionParam << " or " << lite::kAscendContextSection
                  << ", job_type should be in section " << lite::kAoeGlobalOptionsSection;
  }
  return job_types;
}

Status AoeApiTuning::AoeTurningGraph(const std::shared_ptr<ge::Session> &session, const transform::DfGraphPtr &graph,
                                     const std::vector<ge::Tensor> &inputs, const std::shared_ptr<Context> &context,
                                     const ConfigInfos &config_infos) {
  auto global_options = GetAoeGlobalOptions(context, config_infos);
  auto tuning_options = GetAoeTuningOptions(context, config_infos);
  auto job_types = GetAoeJobType(context, config_infos);
  if (job_types.empty()) {
    return kLiteError;
  }
  if (ExecuteAoe(session, graph, inputs, job_types, global_options, tuning_options) != kSuccess) {
    MS_LOG(ERROR) << "Execute aoe failed";
    return kLiteError;
  }
  return kSuccess;
}
}  // namespace mindspore
