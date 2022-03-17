/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/recovery/recovery_context.h"

#include <dirent.h>
#include <algorithm>
#include <utility>

#include "nlohmann/json.hpp"
#include "ps/ps_context.h"
#include "ps/constants.h"
#include "utils/file_utils.h"
#include "distributed/constants.h"
#include "distributed/cluster/topology/common.h"
#include "distributed/init.h"
#include "runtime/hardware/device_context.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace runtime {
namespace recovery {
constexpr char kEnvEnableRecovery[] = "MS_ENABLE_RECOVERY";
constexpr char kEnvRecoveryPath[] = "MS_RECOVERY_PATH";
constexpr char kEnvRecoveryInterval[] = "MS_RECOVERY_INTERVAL";

constexpr char kCkptSuffix[] = ".ckpt";
constexpr char kCkptPath[] = "ckpt_path";
constexpr char kJsonSuffix[] = ".json";
constexpr char kConfigJson[] = "/config.json";

const uint32_t kSendBufferLen = 2;

namespace {
std::pair<int, int> ParseCkptEpochStep(const std::string &checkpoint) {
  size_t suffix_pos = checkpoint.rfind('.');
  if (suffix_pos == std::string::npos || checkpoint.substr(suffix_pos) != kCkptSuffix) {
    MS_LOG(WARNING) << "The file : " << checkpoint << "is not a checkpoint";
    return {};
  }

  size_t epoch_begin_pos = checkpoint.rfind('-');
  size_t step_begin_pos = checkpoint.rfind('_');
  if (epoch_begin_pos == std::string::npos || step_begin_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "The checkpoint file name is not valid: " << checkpoint;
  }

  return std::make_pair(std::stoi(checkpoint.substr(epoch_begin_pos + 1, (step_begin_pos - epoch_begin_pos) - 1)),
                        std::stoi(checkpoint.substr(step_begin_pos + 1, (suffix_pos - step_begin_pos) - 1)));
}

void RemoveAllCkptFiles(const std::string &directory, const std::vector<std::string> &files_list) {
  for (size_t i = 0; i < files_list.size(); i++) {
    const auto &ckpt_name = files_list[i];
    const auto &ckpt_file = directory + "/" + ckpt_name;
    (void)remove(ckpt_file.c_str());
  }
}
}  // namespace

void RecoveryContext::Initialize() {
  if (initialized_) {
    return;
  }

  // 1. Read environment variable.
  enable_recovery_ = (common::GetEnv(kEnvEnableRecovery) == std::string("1"));
  if (!enable_recovery_) {
    return;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_RECOVERY, true);

  recovery_path_ = common::GetEnv(kEnvRecoveryPath);
  if (recovery_path_.empty()) {
    MS_LOG(EXCEPTION) << "The recovery path is empty, please export MS_RECOVERY_PATH correctly.";
  }

  auto env_recovery_interval = common::GetEnv(kEnvRecoveryInterval);
  if (!env_recovery_interval.empty()) {
    recovery_interval_ = std::stoi(env_recovery_interval);
  }

  node_role_ = common::GetEnv(distributed::kEnvRole);
  if (distributed::kValidRoleName.count(node_role_) == 0) {
    MS_LOG(EXCEPTION) << "Role name '" << node_role_ << "' is invalid. ";
  }

  // 2. Create config json file.
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    if (!FileIOUtils::IsFileOrDirExist(recovery_path_)) {
      FileIOUtils::CreateDirRecursive(recovery_path_);
    }

    auto ret = FileUtils::GetRealPath(recovery_path_.c_str());
    if (!ret.has_value()) {
      MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage path: " << recovery_path_;
    }
    recovery_path_ = ret.value();
    if (!FileIOUtils::IsFileOrDirExist(recovery_path_ + kConfigJson)) {
      nlohmann::json config_js;
      config_js[std::string(ps::kStoreType)] = 1;
      config_js[std::string(ps::kStoreFilePath)] = recovery_path_ + "/" + ps::kStoreFilePath + kJsonSuffix;
      config_js[std::string(ps::kSchedulerStoreFilePath)] =
        recovery_path_ + "/" + ps::kSchedulerStoreFilePath + kJsonSuffix;

      nlohmann::json recovery_js;
      recovery_js[std::string(ps::kKeyRecovery)] = config_js;
      std::ofstream config_file(recovery_path_ + kConfigJson);
      config_file << recovery_js.dump();
      config_file.close();
    }
  }

  // 3. Worker or Server need to wait the recovery config json file to be created.
  while (!FileIOUtils::IsFileOrDirExist(recovery_path_ + kConfigJson)) {
    // Wait duration: 200ms.
    const int kWaitDuration = 200;
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitDuration));
  }

  // 4. Set config content to PSContext.
  ps::PSContext::instance()->set_config_file_path(recovery_path_ + kConfigJson);
  ps::PSContext::instance()->set_node_id(common::GetEnv(distributed::cluster::topology::kEnvNodeId));

  initialized_ = true;
}

bool RecoveryContext::ReInitializeCollective() {
  auto ret = distributed::Initialize();
  if (ret) {
    recovery_status_ = RecoveryErrCode::kUnKnownError;
    set_need_reset(true);
    set_need_sync_weight_to_device(true);
    return true;
  }

  if (recovery_status_ == RecoveryErrCode::kBroadcastUniqueIDFailed ||
      recovery_status_ == RecoveryErrCode::kAllGatherHostNameFailed) {
    MS_LOG(WARNING) << "Prepare to initialize NCCL failed, retrying.";
    // Retry duration: 30s.
    const int kRetryDuration = 30;
    std::this_thread::sleep_for(std::chrono::seconds(kRetryDuration));
    return ReInitializeCollective();
  } else if (recovery_status_ == RecoveryErrCode::kInitNcclFailed) {
    MS_LOG(EXCEPTION) << "Initialize NCCL failed.";
  }

  MS_LOG(EXCEPTION) << "ReInitialize collective failed.";
  return false;
}

void RecoveryContext::ObtainGlobalLatestCkptInfo() {
  // 1. Obtain the step corresponding to the local latest checkpoint.
  ObtainLocalLatestCkptInfo();

  // For standalone training.
  if (global_rank_size_ == 0) {
    return;
  }

  // 2. AllGather the latest checkpoint info of all nodes.
  device::DeviceContextKey host_key = {"CPU", 0};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  device::CollectiveCommunicationLib *host_comm_lib_instance = host_context->collective_comm_lib();
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance);

  if (global_rank_id_ >= global_rank_size_) {
    MS_LOG(EXCEPTION) << "The global rank id " << global_rank_id_ << " should be less than global rank size "
                      << global_rank_size_;
  }

  const uint32_t kRecvBufferLen = kSendBufferLen * global_rank_size_;

  int send_buffer[kSendBufferLen] = {latest_ckpt_epoch_, latest_ckpt_step_};
  int recv_buffer[kRecvBufferLen];
  (void)std::fill_n(recv_buffer, kRecvBufferLen, 0);
  recv_buffer[kSendBufferLen * global_rank_id_] = latest_ckpt_epoch_;
  recv_buffer[kSendBufferLen * global_rank_id_ + 1] = latest_ckpt_step_;

  const std::string &host_global_group_name = host_comm_lib_instance->global_group_name();
  if (!host_comm_lib_instance->AllGather(send_buffer, recv_buffer, kSendBufferLen, TypeId::kNumberTypeInt,
                                         host_global_group_name)) {
    MS_LOG(EXCEPTION) << "AllGather latest ckpt step failed";
  }

  // 3. Check whether save checkpoint successfully on every workers.
  uint32_t save_ckpt_success_num = 0;
  uint32_t save_ckpt_failed_num = 0;
  for (uint32_t i = 0; i < kRecvBufferLen; i += kSendBufferLen) {
    if (recv_buffer[i] < 0) {
      save_ckpt_failed_num++;
    } else {
      save_ckpt_success_num++;
    }
  }

  if (save_ckpt_success_num > 0 && save_ckpt_failed_num > 0) {
    RemoveAllCkptFiles(GetCkptPath(), ckpt_files_);
    MS_LOG(EXCEPTION) << "Can not find checkpoint for same step, the workers quits and training should start over.";
  }
  if (save_ckpt_success_num == 0 && save_ckpt_failed_num == global_rank_size_) {
    return;
  }

  // 4. Parse latest epoch and step info.
  ParseLatestCkptInfo(recv_buffer, kRecvBufferLen);

  // 5. Remove useless ckpt
  for (int i = SizeToInt(ckpt_files_.size()) - 1; i >= 0; i--) {
    const auto &last_ckpt_name = ckpt_files_[i];
    const auto &last_ckpt_file = GetCkptPath() + "/" + last_ckpt_name;
    if (last_ckpt_file != latest_ckpt_file_) {
      (void)remove(last_ckpt_file.c_str());
    } else {
      break;
    }
  }
}

void RecoveryContext::ObtainLocalLatestCkptInfo() {
  std::string ckpt_save_dir = GetCkptPath();
  if (ckpt_save_dir.empty()) {
    MS_LOG(INFO) << "The ckpt file path is empty";
    return;
  }

  DIR *dir = opendir(ckpt_save_dir.c_str());
  if (dir == nullptr) {
    MS_LOG(EXCEPTION) << "The file path [" << ckpt_save_dir << "] is not exist";
    return;
  }

  if (!ckpt_files_.empty()) {
    ckpt_files_.clear();
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string file_name = entry->d_name;
    size_t suffix_pos = file_name.rfind('.');
    if (suffix_pos == std::string::npos || file_name.substr(suffix_pos) != kCkptSuffix) {
      continue;
    }

    ckpt_files_.push_back(file_name);
  }
  (void)closedir(dir);

  if (ckpt_files_.empty()) {
    MS_LOG(INFO) << "There is no checkpoint file in dir: " << ckpt_save_dir;
    return;
  }

  sort(ckpt_files_.begin(), ckpt_files_.end(), [](const std::string &a, const std::string &b) {
    auto ckpt_epoch_step_a = ParseCkptEpochStep(a);
    auto ckpt_epoch_step_b = ParseCkptEpochStep(b);
    if (ckpt_epoch_step_a.first < ckpt_epoch_step_b.first) {
      return true;
    } else if (ckpt_epoch_step_a.first == ckpt_epoch_step_b.first) {
      return ckpt_epoch_step_a.second < ckpt_epoch_step_b.second;
    } else {
      return false;
    }
  });

  const auto &latest_ckpt_name = ckpt_files_.back();
  latest_ckpt_file_ = ckpt_save_dir + "/" + latest_ckpt_name;

  auto ckpt_epoch_step = ParseCkptEpochStep(latest_ckpt_name);
  latest_ckpt_epoch_ = ckpt_epoch_step.first;
  latest_ckpt_step_ = ckpt_epoch_step.second;
}

void RecoveryContext::ParseLatestCkptInfo(const int *recv_buffer, const uint32_t buffer_len) {
  std::vector<std::pair<int, int>> ckpts_epoch_step;
  for (uint32_t i = 0; i < buffer_len; i += kSendBufferLen) {
    ckpts_epoch_step.emplace_back(recv_buffer[i], recv_buffer[i + 1]);
  }
  sort(ckpts_epoch_step.begin(), ckpts_epoch_step.end(),
       [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
         if (a.first < b.first) {
           return true;
         } else if (a.first == b.first) {
           return a.second < b.second;
         } else {
           return false;
         }
       });

  const std::pair<int, int> &latest_epoch_step = ckpts_epoch_step.front();
  latest_ckpt_epoch_ = latest_epoch_step.first;
  latest_ckpt_step_ = latest_epoch_step.second;

  const std::string latest_epoch_step_suffix =
    std::to_string(latest_epoch_step.first) + "_" + std::to_string(latest_epoch_step.second) + kCkptSuffix;
  auto iter = std::find_if(ckpt_files_.rbegin(), ckpt_files_.rend(), [&](const std::string &file_name) {
    if (file_name.size() <= latest_epoch_step_suffix.size()) {
      return false;
    }
    return file_name.rfind(latest_epoch_step_suffix) == (file_name.size() - latest_epoch_step_suffix.size());
  });
  if (iter == ckpt_files_.rend()) {
    RemoveAllCkptFiles(GetCkptPath(), ckpt_files_);
    MS_LOG(EXCEPTION) << "Can not find checkpoint for same step, the workers quits and training should start over.";
  }

  latest_ckpt_file_ = GetCkptPath() + "/" + *iter;
}

void RecoveryContext::CreatePersistentFile() {
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    return;
  }

  if (persistent_json_ != nullptr) {
    return;
  }

  // Need to get real path of recovry path for worker or server.
  auto ret = FileUtils::GetRealPath(recovery_path_.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage path: " << recovery_path_;
  }
  recovery_path_ = ret.value();

  // The directory used to save ckpt is persisted to json file.
  std::string persistent_file_path =
    recovery_path_ + "/" + node_role_ + "_" + std::to_string(global_rank_id_) + "_persistent.json";
  persistent_json_ = std::make_unique<JsonUtils>(persistent_file_path);
  if (!persistent_json_->Initialize()) {
    MS_LOG(EXCEPTION) << "Initialize json failed, file path: " << persistent_file_path;
  }
}

void RecoveryContext::SetCkptPath(const std::string &path) {
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    return;
  }

  if (!FileIOUtils::IsFileOrDirExist(path)) {
    FileIOUtils::CreateDirRecursive(path);
  }

  auto ret = FileUtils::GetRealPath(path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path for save checkpoint, path: " << path;
  }

  if (persistent_json_ == nullptr) {
    CreatePersistentFile();
  }

  MS_EXCEPTION_IF_NULL(persistent_json_);
  persistent_json_->Insert(kCkptPath, ret.value());
}

std::string RecoveryContext::GetCkptPath() {
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    return std::string();
  }

  if (persistent_json_ == nullptr) {
    CreatePersistentFile();
  }

  MS_EXCEPTION_IF_NULL(persistent_json_);
  if (!persistent_json_->Exists(kCkptPath)) {
    return std::string();
  }

  return persistent_json_->Get<std::string>(kCkptPath);
}
}  // namespace recovery
}  // namespace runtime
}  // namespace mindspore
