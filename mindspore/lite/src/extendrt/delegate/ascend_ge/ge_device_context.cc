/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include <cxxabi.h>
#include <utility>
#include <vector>
#include "include/common/utils/scoped_long_running.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/backend/device_type.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/transform/graph_ir/utils.h"
#include "external/ge/ge_api.h"
#include "runtime/config.h"
#include "common/config_infos.h"
#include "common/common.h"
#include "extendrt/delegate/comm_group_info.h"
#include "backend/common/session/executor.h"

namespace mindspore {
constexpr auto kHcclPluginFileName = "libhccl.so";

typedef enum {
  HCCL_SUCCESS = 0,              /**< success */
  HCCL_E_PARA = 1,               /**< parameter error */
  HCCL_E_PTR = 2,                /**< empty pointer */
  HCCL_E_MEMORY = 3,             /**< memory error */
  HCCL_E_INTERNAL = 4,           /**< internal error */
  HCCL_E_NOT_SUPPORT = 5,        /**< not support feature */
  HCCL_E_NOT_FOUND = 6,          /**< not found specific resource */
  HCCL_E_UNAVAIL = 7,            /**< resource unavailable */
  HCCL_E_SYSCALL = 8,            /**< call system interface error */
  HCCL_E_TIMEOUT = 9,            /**< timeout */
  HCCL_E_OPEN_FILE_FAILURE = 10, /**< open file fail */
  HCCL_E_TCP_CONNECT = 11,       /**< tcp connect fail */
  HCCL_E_ROCE_CONNECT = 12,      /**< roce connect fail */
  HCCL_E_TCP_TRANSFER = 13,      /**< tcp transfer fail */
  HCCL_E_ROCE_TRANSFER = 14,     /**< roce transfer fail */
  HCCL_E_RUNTIME = 15,           /**< call runtime api fail */
  HCCL_E_DRV = 16,               /**< call driver api fail */
  HCCL_E_PROFILING = 17,         /**< call profiling api fail */
  HCCL_E_CCE = 18,               /**< call cce api fail */
  HCCL_E_NETWORK = 19,           /**< call network api fail */
  HCCL_E_AGAIN = 20,             /**< try again */
  HCCL_E_RESERVED                /**< reserved */
} HcclResult;

using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

extern "C" {
HcclResult HcomCreateGroup(const char *, uint32_t, uint32_t *);
HcclResult HcomInitByFile(const char *, const char *);
HcclResult HcomDestroy();
}

constexpr const char *kHcomCreateGroupName = "HcomCreateGroup";
constexpr const char *kHcomInitByFileName = "HcomInitByFile";
constexpr const char *kHcomDestroyName = "HcomDestroy";

using HcomCreateGroupFunObj = std::function<HcclResult(const char *, uint32_t, uint32_t *)>;
using HcomInitByFileFunObj = std::function<HcclResult(const char *, const char *)>;
using HcomDestroyFunObj = std::function<HcclResult()>;
using HcomCreateGroupFunPtr = HcclResult (*)(const char *, uint32_t, uint32_t *);
using HcomInitByFileFunPtr = HcclResult (*)(const char *, const char *);
using HcomDestroyFunPtr = HcclResult (*)();

HcomCreateGroupFunObj HcomCreateGroup_;
HcomInitByFileFunObj HcomInitByFile_;
HcomDestroyFunObj HcomDestroy_;

bool ge_initialize_ = true;
bool init_hccl_exec_ = false;

bool do_hccl_sym_load() {
  void *libhccl = dlopen(kHcclPluginFileName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
  if (libhccl == nullptr) {
    MS_LOG(ERROR) << "Dlopen libhccl" << kHcclPluginFileName << " failed, result = " << GetDlErrorMsg();
    return false;
  }
  HcomCreateGroup_ = DlsymWithCast<HcomCreateGroupFunPtr>(libhccl, kHcomCreateGroupName);
  HcomInitByFile_ = DlsymWithCast<HcomInitByFileFunPtr>(libhccl, kHcomInitByFileName);
  HcomDestroy_ = DlsymWithCast<HcomDestroyFunPtr>(libhccl, kHcomDestroyName);
  if (HcomCreateGroup_ == nullptr || HcomInitByFile_ == nullptr || HcomDestroy_ == nullptr) {
    MS_LOG(ERROR) << "Dlsys libhccl failed, result = " << GetDlErrorMsg();
    return false;
  }
  return true;
}

bool load_hccl_symbols() {
  static std::once_flag g_flag;
  static bool ret = false;
  std::call_once(g_flag, [] { ret = do_hccl_sym_load(); });
  return ret;
}

bool InitHcclExec(const std::string &rank_table_path, const std::string &identify) {
  if (ge_initialize_) {
    return true;
  }
  MS_LOG(INFO) << "Start init hccl exec.";
  MS_EXCEPTION_IF_NULL(HcomInitByFile_);
  HcclResult hccl_ret = HcomInitByFile_(rank_table_path.c_str(), identify.c_str());
  if (hccl_ret == HCCL_E_PTR) {
    MS_LOG(WARNING) << "Hccl comm is null, hcom executor initialize is not required";
  } else if (hccl_ret == HCCL_SUCCESS) {
    MS_LOG(INFO) << "Hcom DynamicKernel Initialize success";
  } else {
    MS_LOG(ERROR) << "Hcom DynamicKernel Initialize failed";
    return false;
  }
  init_hccl_exec_ = true;
  MS_LOG(INFO) << "InitHcclExec success";
  return true;
}

bool FinalizeHcclExec() {
  if (!init_hccl_exec_) {
    return true;
  }
  MS_LOG(INFO) << "Start finalize hccl exec.";
  MS_EXCEPTION_IF_NULL(HcomDestroy_);
  HcclResult hccl_ret = HcomDestroy_();
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Hcom DynamicKernel Finalize failed";
    return false;
  }
  init_hccl_exec_ = false;
  MS_LOG(INFO) << "HcclExec destroy success";
  return true;
}

std::weak_ptr<GeDeviceContext> GeDeviceContext::global_ge_context_;
std::mutex GeDeviceContext::global_ge_context_mutex_;

GeDeviceContext::GeDeviceContext() = default;

GeDeviceContext::~GeDeviceContext() { Destroy(); }

std::shared_ptr<GeDeviceContext> GeDeviceContext::InitGlobalContext(const std::shared_ptr<Context> &context,
                                                                    const ConfigInfos &config_info) {
  std::lock_guard<std::mutex> lock(global_ge_context_mutex_);
  auto ge_context = global_ge_context_.lock();
  if (ge_context != nullptr) {
    MS_LOG(INFO) << "GE Context has been initialized, skip.";
  } else {
    ge_context = std::make_shared<GeDeviceContext>();
    if (!ge_context) {
      MS_LOG(ERROR) << "Failed to create GeDeviceContext";
      return nullptr;
    }
    auto status = ge_context->Initialize(context, config_info);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Failed to initialize GeDeviceContext";
      return nullptr;
    }
    global_ge_context_ = ge_context;
    MS_LOG(INFO) << "Init global ge context success.";
  }
  return ge_context;
}

std::shared_ptr<AscendDeviceInfo> GeDeviceContext::GetGeAscendDeviceInfo(const std::shared_ptr<Context> &context) {
  auto device_list = context->MutableDeviceInfo();
  auto ascend_info_iter = std::find_if(
    device_list.begin(), device_list.end(), [&](std::shared_ptr<mindspore::DeviceInfoContext> &device_info) {
      return (device_info && device_info->GetDeviceType() == kAscend && device_info->GetProvider() == "ge");
    });
  if (ascend_info_iter == device_list.end()) {
    MS_LOG(ERROR) << "AscendDeviceInfo is not set. If using distributed inference, make sure device_id "
                     "and rank_id are set in AscendDeviceInfo";
    return nullptr;
  }
  auto device_info = *(ascend_info_iter);
  return device_info->Cast<mindspore::AscendDeviceInfo>();
}

Status GeDeviceContext::Initialize(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MsContext::GetInstance()->set_backend_policy("ge");
  auto status = InitGe(MsContext::GetInstance(), context, config_info);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to Init GE";
    return status;
  }
  status = InitHccl(context, config_info);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to Init HCCL";
    return status;
  }
  return kSuccess;
}

void GeDeviceContext::Destroy() {
  (void)FinalizeGe(MsContext::GetInstance());
  FinalizeHcclExec();
}

Status GeDeviceContext::InitHccl(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  if (!load_hccl_symbols()) {
    return kCoreFailed;
  }
  auto ascend_info = GetGeAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Failed to Get AscendDeviceInfo from context";
    return kCoreFailed;
  }
  uint32_t device_id = ascend_info->GetDeviceID();
  std::string rank_table_file = "";
  if (config_info.empty() || config_info.find(lite::kAscendContextSection) == config_info.end()) {
    MS_LOG(INFO) << "There is no ascend context info in config file.";
  } else {
    auto config_info_ascend = config_info.at(lite::kAscendContextSection);
    if (config_info_ascend.find(lite::kRankTableFilePathKey) == config_info_ascend.end()) {
      MS_LOG(INFO)
        << "There is no rank table file in Ascend section of config file, distributed inference is not enabled."
        << " If using distributed inference, make sure rank_table_file in the config file,"
        << " device_id and rank_id are set in AscendDeviceInfo.";
    } else {
      rank_table_file = config_info_ascend[lite::kRankTableFilePathKey];
      MS_LOG(INFO) << "Distributed inference is enabled, rank table file: " << rank_table_file;
    }
  }
  auto device_id_s = std::to_string(device_id);
  InitHcclExec(rank_table_file, device_id_s);

  auto group_info_file = context->GetGroupInfoFile();
  if (!group_info_file.empty()) {
    MS_LOG(INFO) << "Get env group_info"
                 << " success: " << group_info_file;
    GroupInfoMap group_info_map;
    lite::CommGroupInfo comm_group_info;

    if (!comm_group_info.LoadGroupInfo(group_info_file, &group_info_map)) {
      MS_LOG(ERROR) << "LoadGroupInfo failed.";
      return kMEInvalidInput;
    }
    for (const auto &[group_name, rank_ids] : group_info_map) {
      MS_LOG(INFO) << "group_name" << group_name << "rank_ids" << rank_ids;
      auto rank_size = rank_ids.size();
      auto res = HcomCreateGroup_(reinterpret_cast<const char *>(group_name.c_str()), UlongToUint(rank_size),
                                  std::vector<unsigned int>(rank_ids).data());
      if (res != HCCL_SUCCESS) {
        MS_LOG(ERROR) << "Create group " << group_name << " rank ids " << rank_ids << " failed.";
        return kMEInvalidInput;
      }
    }
    MS_LOG(INFO) << "Create groups by checkpoint file success ";
  }
  return kSuccess;
}

Status GeDeviceContext::InitGe(const std::shared_ptr<MsContext> &inst_context, const std::shared_ptr<Context> &context,
                               const ConfigInfos &config_info) {
  MS_EXCEPTION_IF_NULL(inst_context);
  int32_t is_heterogeneous = 0;
  (void)rtGetIsHeterogenous(&is_heterogeneous);
  inst_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, is_heterogeneous == 1);
  if (inst_context->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return kSuccess;
  }

  if (static_cast<bool>(inst_context->get_param<uint32_t>(MS_CTX_GE_REF))) {
    inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
    return kSuccess;
  }

  std::map<std::string, std::string> ge_options;
  GetGeOptions(inst_context, context, &ge_options, config_info);
  for (auto &option : ge_options) {
    MS_LOG(INFO) << "GE Global option " << option.first << " = " << option.second;
  }
  if (ge::GEInitialize(ge_options) != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Initialize GE failed: " << ge::GEGetErrorMsg();
    return kLiteError;
  }
  inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  return kSuccess;
}

void GeDeviceContext::SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(WARNING) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
  }
}

void GeDeviceContext::GetGeOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                   const std::shared_ptr<Context> &context,
                                   std::map<std::string, std::string> *ge_options, const ConfigInfos &config_info) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(ge_options);

  (*ge_options)["device_id"] = "0";
  (*ge_options)["rank_table_file"] = "";
  (*ge_options)["ge.DDK_version"] = "1.60.T17.B830";
  (*ge_options)["graphType"] = "1";
  (*ge_options)["ge.graphRunMode"] = "0";
  (*ge_options)["ge.exec.disableReuseMemory"] = "0";
  (*ge_options)["ge.exec.jobId"] = "0";
  (*ge_options)["ge.exec.precision_mode"] = "force_fp16";
  SetHcclOptions(context, ge_options, config_info);

  // Disable the global variable acc, only enable it while adding training graph in pipeline
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // 0: False, dynamic and static graph compile with cann opp_kernel*.run，GE default for pytorch
  // 1: True, dynamic and static graph online compiler op
  // 2: Auto, dynamic compile with cann opp_kernel*.run, static graph online compiler op，GE default for others
  (*ge_options)["ge.jit_compile"] = "2";
  auto config_it = config_info.find(lite::kGeGlobalOptionsSection);
  if (config_it != config_info.end()) {
    for (auto &item : config_it->second) {
      (*ge_options)[item.first] = item.second;
      MS_LOG(INFO) << "Set ge global option " << item.first << " to " << item.second;
    }
  }
}

void GeDeviceContext::SetHcclOptions(const std::shared_ptr<Context> &context,
                                     std::map<std::string, std::string> *ge_options, const ConfigInfos &config_info) {
  auto ascend_info = GetGeAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Failed to Get AscendDeviceInfo from context";
    return;
  }
  std::string rank_table_file = "";
  uint32_t device_id = ascend_info->GetDeviceID();
  uint32_t rank_id = ascend_info->GetRankID();

  if (config_info.empty() || config_info.find(lite::kAscendContextSection) == config_info.end()) {
    MS_LOG(INFO) << "There is no ascend context info in config file.";
  } else {
    auto config_info_ascend = config_info.at(lite::kAscendContextSection);
    if (config_info_ascend.find(lite::kRankTableFilePathKey) == config_info_ascend.end()) {
      MS_LOG(INFO)
        << "There is no rank table file in Ascend section of config file, distributed inference is not enabled."
        << " If using distributed inference, make sure rank_table_file in the config file,"
        << " device_id and rank_id are set in AscendDeviceInfo.";
    } else {
      rank_table_file = config_info_ascend[lite::kRankTableFilePathKey];
      MS_LOG(INFO) << "Distributed inference is enabled, rank table file: " << rank_table_file;
    }
  }

  MS_LOG(INFO) << "Set ge_options for rank table file " << rank_table_file << " device id " << device_id << " rank id "
               << rank_id;
  if (!rank_table_file.empty()) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
    (*ge_options)["ge.exec.rankTableFile"] = rank_table_file;
    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = std::to_string(device_id);
    (*ge_options)["ge.exec.rankId"] = std::to_string(rank_id);
    (*ge_options)["ge.exec.podName"] = std::to_string(rank_id);
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = std::to_string(device_id);
    MS_LOG(INFO) << "No hccl mode. "
                 << "If use hccl, make sure that the rank table file path is set in config file, "
                 << "rank id and device id are set in AscendDeviceInfo.";
  }
  (*ge_options)["ge.exec.deployMode"] = "0";
}

bool GeDeviceContext::FinalizeGe(const std::shared_ptr<MsContext> &inst_context) {
  MS_EXCEPTION_IF_NULL(inst_context);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    return true;
  }
  inst_context->decrease_param<uint32_t>(MS_CTX_GE_REF);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    inst_context->set_param<uint32_t>(MS_CTX_GE_REF, 0);
    try {
      transform::ClearGeSessionAndRunner();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (ge::GEFinalize() != ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    inst_context->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = "
                 << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  }
  return true;
}
}  // namespace mindspore
