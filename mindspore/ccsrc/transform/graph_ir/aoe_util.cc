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
#include <dlfcn.h>
#include <cxxabi.h>
#include <set>
#include <string>
#include "include/common/debug/common.h"
#include "transform/graph_ir/aoe_util.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace transform {
namespace AoeOptions {
const ::ge::AscendString JOB_TYPE = ::ge::AscendString("job_type");
const ::ge::AscendString FRAMEWORK = ::ge::AscendString("framework");
const ::ge::AscendString LOG_LEVEL = ::ge::AscendString("log");
const ::ge::AscendString PRECISION_MODE = ::ge::AscendString("precision_mode");
}  // namespace AoeOptions

bool IsAscendServer() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->ascend_soc_version().find("ascend910") != std::string::npos;
}

AoeUtil::AoeUtil() : initialize_(false) {}

AoeUtil::~AoeUtil() { MS_LOG(INFO) << "release aoeutil success."; }

void AoeUtil::Initialize() {
  if (initialize_) {
    MS_LOG(INFO) << "Aoe already initialized.";
    return;
  }
  if (IsAscendServer()) {
    std::string ascend_path = GetAscendPath();
    auto ld_library_path = common::GetEnv("LD_LIBRARY_PATH");
    ld_library_path = ascend_path + "lib64:" + ld_library_path;
    common::SetEnv("LD_LIBRARY_PATH", ld_library_path.c_str());
    std::string aoe_plugin_path = "lib64/libaoe_tuning.so";
    auto plugin_path = ascend_path + aoe_plugin_path;
    auto ret = access(plugin_path.c_str(), F_OK);
    if (ret != 0) {
      MS_LOG(WARNING) << "plugin " << plugin_path << " not exist";
      return;
    }

    const std::vector<std::string> depend_libs = {"libopat.so", "libaoe_plugin.so", "libparser_common.so"};
    for (const auto &dep_lib : depend_libs) {
      auto dep_lip_path = ascend_path + "lib64/" + dep_lib;
      auto dep_handler = dlopen(dep_lip_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (dep_handler != nullptr) {
        depend_handler_.push_back(dep_handler);
      } else {
        MS_LOG(INFO) << "Cannot dlopen " << dep_lip_path << ", result = " << GetDlErrorMsg()
                     << ", it can be ignored if not use aoe.";
      }
    }

    plugin_handle_ = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (plugin_handle_ == nullptr) {
      MS_LOG(INFO) << "Cannot dlopen " << plugin_path << ", result = " << GetDlErrorMsg()
                   << ", it can be ignored if not use aoe.";
      return;
    }
    MS_LOG(INFO) << "load " << aoe_plugin_path << " success";
    aoe_initialize_ = DlsymFuncObj(AoeInitialize, plugin_handle_);
    aoe_finalize_ = DlsymFuncObj(AoeFinalize, plugin_handle_);
    aoe_create_session_ = DlsymFuncObj(AoeCreateSession, plugin_handle_);
    aoe_set_ge_gession_ = DlsymFuncObj(AoeSetGeSession, plugin_handle_);
    aoe_set_tuning_graph_ = DlsymFuncObj(AoeSetTuningGraph, plugin_handle_);
    aoe_tuning_graph_ = DlsymFuncObj(AoeTuningGraph, plugin_handle_);
    aoe_destroy_session_ = DlsymFuncObj(AoeDestroySession, plugin_handle_);
    auto ms_context = MsContext::GetInstance();
    std::string aoe_job_type = ms_context->get_param<std::string>(MS_CTX_AOE_JOB_TYPE);
    std::map<::ge::AscendString, ::ge::AscendString> globalOptions = {
      {AoeOptions::JOB_TYPE, ::ge::AscendString(aoe_job_type.c_str())}};
    const AoeStatus status = aoe_initialize_(globalOptions);
    if (status != AOE_SUCCESS) {
      MS_LOG(ERROR) << "AoeInitialize failed. Please refer to 'Ascend Optimization Engine' at "
                    << "https://www.mindspore.cn to set environment variables.";
    }
    MS_LOG(INFO) << "AoeInitialize success.";
    initialize_ = true;
  }
}

void AoeUtil::Destroy() {
  if (!initialize_) {
    MS_LOG(WARNING) << "AOE not initialize, stop destroy";
    return;
  }
  if (IsAscendServer()) {
    try {
      const AoeStatus status = aoe_finalize_();
      if (status != AOE_SUCCESS) {
        MS_LOG(ERROR) << "AoeFinalize failed. status is " << status;
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when exec aoe finalize. Error:" << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when  exec aoe finalize. Exception name: " << exName;
    }
  }
  if (plugin_handle_ == nullptr) {
    return;
  }
  aoe_initialize_ = nullptr;
  aoe_finalize_ = nullptr;
  aoe_create_session_ = nullptr;
  aoe_set_ge_gession_ = nullptr;
  aoe_set_tuning_graph_ = nullptr;
  aoe_tuning_graph_ = nullptr;
  aoe_destroy_session_ = nullptr;
  MS_LOG(INFO) << "AoeFinalization success.";
  for (const auto &dep_handler : depend_handler_) {
    (void)dlclose(dep_handler);
  }
  (void)dlclose(plugin_handle_);
  plugin_handle_ = nullptr;
  initialize_ = false;
}

AoeUtil &AoeUtil::GetInstance() {
  static AoeUtil instance{};
  return instance;
}

Status AoeUtil::AoeGeGraph(::ge::Session *ge_session, const transform::DfGraphPtr &graph,
                           const std::map<::ge::AscendString, ::ge::AscendString> &tuningOptions) const {
  uint64_t sessionId = 0;
  AoeStatus status = aoe_create_session_(sessionId);
  if (status != AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeCreateSession failed. error code:" << status;
    return FAILED;
  }
  MS_LOG(DEBUG) << "AoeCreateSession success.";

  status = aoe_set_ge_gession_(sessionId, ge_session);
  if (status != AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeSetGeSession failed. error code:" << status;
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeSetGeSession success.";

  status = aoe_set_tuning_graph_(sessionId, *graph);
  if (status != AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeSetGraph failed. error code:" << status;
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeSetGraph success.";

  status = aoe_tuning_graph_(sessionId, tuningOptions);
  if (status != AOE_SUCCESS && status != AOE_ERROR_NON_OPTIMIZE_GRAPH) {
    MS_LOG(ERROR) << "AoeTuningGraph failed. error code:" << status;
    (void)aoe_destroy_session_(sessionId);
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeTuningGraph success.";

  status = aoe_destroy_session_(sessionId);
  if (status != AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeDestroySession failed. error code:" << status;
    return FAILED;
  }
  return SUCCESS;
}

Status AoeUtil::AoeOnlineGeGraph(const std::shared_ptr<::ge::Session> &ge_session,
                                 const transform::DfGraphPtr &graph) const {
  MS_LOG(DEBUG) << "AoeOnlineGeGraph start.";
  if (!initialize_) {
    MS_LOG(WARNING) << "AOE not initialize";
    return FAILED;
  }
  if (ge_session == nullptr) {
    MS_LOG(ERROR) << "sess is null";
    return FAILED;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  ::ge::AscendString precision_mode = "allow_fp32_to_fp16";
  if (soc_version == "ascend910b" || soc_version == "ascend910c") {
    precision_mode = "must_keep_origin_dtype";
  }

  std::map<::ge::AscendString, ::ge::AscendString> tuneOptions = {
    {AoeOptions::FRAMEWORK, ::ge::AscendString("1")},
    {AoeOptions::PRECISION_MODE, precision_mode},
    {AoeOptions::LOG_LEVEL, ::ge::AscendString("error")},
  };

  if (AoeGeGraph(ge_session.get(), graph, tuneOptions) != SUCCESS) {
    MS_LOG(ERROR) << "Failed to call Aoe online tuning.";
    return FAILED;
  }

  MS_LOG(DEBUG) << "AoeTuningGraph success.";
  return SUCCESS;
}

void AoeUtil::SaveOptimizedGraph(const int32_t &graph_id) { optimized_graphs_id_.insert(graph_id); }

bool AoeUtil::IsSaveOptimizedGraph(const int32_t &graph_id) const {
  auto iter_find = optimized_graphs_id_.find(graph_id);
  if (iter_find != optimized_graphs_id_.end()) {
    return true;
  }
  return false;
}

void AoeUtil::RemoveWaitOptimizedGraph(const std::set<std::string> &optimized_graph_names) {
  for (auto &graph_name : optimized_graph_names) {
    if (auto remove_iter = wait_optimize_graphs_.find(graph_name); remove_iter != wait_optimize_graphs_.end())
      (void)wait_optimize_graphs_.erase(remove_iter);
  }
  if (!wait_optimize_graphs_.empty()) {
    MS_LOG(WARNING) << "optimize_graphs_ is not empty";
  }
}

void AoeUtil::AddOptimizeGraph(const std::string &graph_name) { wait_optimize_graphs_.insert(graph_name); }

std::set<std::string> AoeUtil::GetWaitOptimizeGraph() const { return wait_optimize_graphs_; }

void AoeUtil::SetOfflineEnvDumpGeGraph() {
  auto file_path = GetSaveGraphsPathName("aoe_dump");
  auto real_path = FileUtils::CreateNotExistDirs(file_path, true);
  if (!real_path.has_value()) {
    MS_LOG(WARNING) << "fail to create aoe dump dir " << real_path.value();
    return;
  }
  common::SetEnv("DUMP_GE_GRAPH", "1");
  common::SetEnv("DUMP_GRAPH_LEVEL", "4");
  common::SetEnv("DUMP_GRAPH_PATH", real_path.value().c_str());
}
}  // namespace transform
}  // namespace mindspore
