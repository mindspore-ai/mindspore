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

#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"

#include <unistd.h>
#include <fstream>
#include <string>
#include "include/utils/common.h"
#include "acl/acl_rt.h"
#include "utils/distributed_meta.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "utils/temp_file_manager.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";

std::string GenerateAclInitJsonPath() {
  const pid_t pid = getpid();
  std::string rankid_str = common::GetEnv("RANK_ID");
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rankid_str = std::to_string(mindspore::DistributedMeta::GetInstance()->global_rank_id());
  }
  constexpr size_t random_len = 12;
  auto rand_str = Common::GetRandomStr(random_len);
  return "/tmp/aclinit_" + rankid_str + "_" + std::to_string(pid) + "_" + rand_str + ".json";
}
}  // namespace
static thread_local aclrtContext thread_local_rt_context{nullptr};

AscendHalManager AscendHalManager::instance_{};
AscendHalManager &AscendHalManager::GetInstance() { return instance_; }

void AscendHalManager::InitDevice(uint32_t device_id) {
  MS_LOG(INFO) << "Enter SetRtDevice, current initialize device number:" << initialized_device_set_.size();
  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    MS_LOG(INFO) << "Device " << device_id << " has been set";
    return;
  }

  auto ret = CALL_ASCEND_API(aclrtSetDevice, UintToInt(device_id));
  if (ret != ACL_SUCCESS) {
    auto device_count = GetDeviceCount();
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetDevice failed, ret[" << static_cast<int>(ret)
                                     << "]. Got device count[" << device_count << "] and device id[" << device_id
                                     << "], please check if device id is valid.";
  }

  aclrtContext rt_context;
  ret = CALL_ASCEND_API(aclrtGetCurrentContext, &rt_context);
  if (ret != ACL_SUCCESS || rt_context == nullptr) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtGetCurrentContext failed, ret[" << ret << "]";
    return;
  }

  default_device_context_map_[device_id] = rt_context;
  (void)initialized_device_set_.insert(device_id);
}

void AscendHalManager::ResetDevice(uint32_t device_id) {
  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    auto ret = CALL_ASCEND_API(aclrtResetDevice, UintToInt(device_id));
    if (ret != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "Call aclrtResetDevice, ret[" << ret << "]";
    }
    default_device_context_map_[device_id] = nullptr;
    (void)initialized_device_set_.erase(device_id);
  }
}

uint32_t AscendHalManager::GetDeviceCount() {
  uint32_t device_count = 0;
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &device_count);
  if (ret != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }
  return device_count;
}

void AscendHalManager::SetDeviceSatMode(const aclrtFloatOverflowMode &overflow_mode) {
  auto overflow_mode_str =
    (overflow_mode == aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION) ? kSaturationMode : kINFNANMode;
  MS_LOG(INFO) << "The current overflow detection mode is " << overflow_mode_str << ".";
  auto ret = CALL_ASCEND_API(aclrtSetDeviceSatMode, overflow_mode);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set " << overflow_mode_str << " mode failed.";
  }
}

void AscendHalManager::SetOpWaitTimeout(uint32_t op_wait_timeout) {
  MS_LOG(INFO) << "Set op wait timeout: " << op_wait_timeout << " s";
  auto acl_ret = CALL_ASCEND_API(aclrtSetOpWaitTimeout, op_wait_timeout);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set op wait timeout failed, error: " << acl_ret;
  }
}

void AscendHalManager::SetOpExecuteTimeOut(uint32_t op_execute_timeout) {
  MS_LOG(INFO) << "Set op execute timeout: " << op_execute_timeout << " s";
  auto acl_ret = CALL_ASCEND_API(aclrtSetOpExecuteTimeOut, op_execute_timeout);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set op execute timeout failed, error: " << acl_ret;
  }
}

aclrtContext AscendHalManager::CreateContext(uint32_t device_id) {
  aclrtContext rt_context;
  auto ret = CALL_ASCEND_API(aclrtCreateContext, &rt_context, device_id);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtCreateContext failed, ret: " << ret;
  }
  rt_contexts_.insert(rt_context);
  return rt_context;
}

void AscendHalManager::ResetContext(uint32_t device_id) {
  aclrtContext rt_context = CreateContext(device_id);
  default_device_context_map_[device_id] = rt_context;
}

void AscendHalManager::DestroyContext(aclrtContext context) {
  auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to destroy context, ret = " << ret << ".";
  }
  rt_contexts_.erase(context);
}

void AscendHalManager::DestroyAllContext() {
  for (auto context : rt_contexts_) {
    auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Failed to destroy context, ret = " << ret << ".";
    }
  }
  rt_contexts_.clear();
}

void AscendHalManager::SetContextForce(uint32_t device_id) {
  if (default_device_context_map_[device_id] == nullptr) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, default_device_context_map_[device_id]);
  if (ret != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
}

void AscendHalManager::SetContext(uint32_t device_id) {
  if (default_device_context_map_[device_id] == nullptr) {
    return;
  }
  if (thread_local_rt_context == default_device_context_map_[device_id]) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, default_device_context_map_[device_id]);
  if (ret != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
  thread_local_rt_context = default_device_context_map_[device_id];
}

void AscendHalManager::InitializeAcl() {
  std::lock_guard<std::mutex> lock(acl_init_mutex_);
  if (acl_initialized_) {
    return;
  }
  acl_initialized_ = true;
  std::string file_name = GenerateAclInitJsonPath();
  std::string json_str;
  auto realpath = Common::CreatePrefixPath(file_name);
  if (!realpath.has_value()) {
    MS_LOG(WARNING) << "Failed to get real path: [" << file_name << "] in generate aclInit json file path.";
    return;
  }
  if (!OpDebugConf::GetInstance()->GenerateAclInitJson(realpath.value(), &json_str)) {
    MS_LOG(WARNING) << "Failed to generate aclinit json, the file path is " << realpath.value() << ".";
    return;
  }
  aclError ret = CALL_ASCEND_API(aclInit, realpath.value().c_str());
  TempFileManager::GetInstance().RemoveFile(realpath.value());
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Call aclInit failed, the error number is " << ret << ", json is " << json_str;
  } else {
    MS_LOG(INFO) << "Call aclInit successfully, json is " << json_str;
  }
}

bool AscendHalManager::EnableLccl() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b" && ascend_soc_version != "ascend910_93") {
    return false;
  }
  auto enable_infer_boost = MsContext::GetInstance()->IsEnableInferBoost();
  if (enable_infer_boost) {
    static bool disable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "off";
    if (disable_lccl) {
      return false;
    }
    return true;
  } else {
    static bool enable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "on";
    if (enable_lccl) {
      MS_LOG(EXCEPTION)
        << "Currently, the LCCL communication library is only supported in the inference boost scenario. "
           "Please remove the environment variable MS_ENABLE_LCCL=on.";
    }
    return false;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
