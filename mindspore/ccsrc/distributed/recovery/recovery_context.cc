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

#include "include/backend/distributed/recovery/recovery_context.h"

#include <dirent.h>
#include <algorithm>
#include <utility>
#include <map>

#include "nlohmann/json.hpp"
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/ps/constants.h"
#include "utils/file_utils.h"
#include "include/backend/distributed/constants.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/persistent/storage/json_utils.h"
#include "include/backend/distributed/cluster/topology/common.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#endif
#include "runtime/hardware/device_context_manager.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace distributed {
namespace recovery {
constexpr char kCkptSuffix[] = ".ckpt";
constexpr char kCkptPath[] = "ckpt_path";
constexpr char kJsonSuffix[] = ".json";
constexpr char kConfigJson[] = "/config.json";

const uint32_t kSendBufferLen = 2;

constexpr char kCkptEpochInfoPrefix[] = "ckpt_epoch_rank_";
constexpr char kCkptStepInfoPrefix[] = "ckpt_step_rank_";

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

bool IsEnableRecovery() { return common::GetEnv(kEnvEnableRecovery) == std::string("1"); }

std::string RecoveryPath() { return common::GetEnv(kEnvRecoveryPath); }

void RecoveryContext::Initialize() {
  if (initialized_) {
    return;
  }

  // 1. Read environment variable.
  enable_recovery_ = IsEnableRecovery();
  if (!enable_recovery_) {
    return;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_RECOVERY, true);

  recovery_path_ = RecoveryPath();
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

  // 2. Get real recovery path and create config file.
  if (!storage::FileIOUtils::IsFileOrDirExist(recovery_path_)) {
    storage::FileIOUtils::CreateDirRecursive(recovery_path_);
  }

  auto ret = FileUtils::GetRealPath(recovery_path_.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage path: " << recovery_path_;
  }
  recovery_path_ = ret.value();

  std::string config_file_path = recovery_path_ + kConfigJson;
  if (!storage::FileIOUtils::IsFileOrDirExist(config_file_path)) {
    CreateConfigFile(config_file_path);
  }

  // 3. Set config content to PSContext.
  ps::PSContext::instance()->set_config_file_path(config_file_path);
  ps::PSContext::instance()->set_node_id(common::GetEnv(distributed::cluster::topology::kEnvNodeId));

  initialized_ = true;
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
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  device::CollectiveCommunicationLib *host_comm_lib_instance = host_context->device_res_manager_->collective_comm_lib();
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance);

  if (global_rank_id_ >= global_rank_size_) {
    MS_LOG(EXCEPTION) << "The global rank id " << global_rank_id_ << " should be less than global rank size "
                      << global_rank_size_;
  }

  const std::size_t kRecvBufferLen = kSendBufferLen * global_rank_size_;

  std::vector<int> recv_buffer(kRecvBufferLen, 0);

#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  // Synchronize the checkpoint information between all the other nodes to ensure the accuracy of training.
  auto node = cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(node);
  MS_EXCEPTION_IF_NULL(cgn);

  // Start the ckpt file info exchange process.
  std::map<std::string, std::string> results;
  const std::string biz = "sync_ckpt";

  std::vector<std::string> names_prefix;
  (void)names_prefix.emplace_back(kCkptEpochInfoPrefix);
  (void)names_prefix.emplace_back(kCkptStepInfoPrefix);

  std::vector<std::string> values;
  values.push_back(std::to_string(latest_ckpt_epoch_));
  values.push_back(std::to_string(latest_ckpt_step_));

  if (cgn->ExchangeMetadata(biz, global_rank_size_, names_prefix, values, &results, INT_MAX)) {
    for (uint32_t i = 0; i < global_rank_size_; ++i) {
      auto epoch_key = kCkptEpochInfoPrefix + std::to_string(i);
      auto step_key = kCkptStepInfoPrefix + std::to_string(i);
      auto ckpt_epoch = results[epoch_key];
      auto ckpt_step = results[step_key];
      if (ckpt_epoch.length() > 0 && ckpt_step.length() > 0) {
        recv_buffer[kSendBufferLen * i] = std::stoi(ckpt_epoch);
        recv_buffer[kSendBufferLen * i + 1] = std::stoi(ckpt_step);
        MS_LOG(INFO) << "The latest checkpoint for rank " << i << "is that epoch: " << ckpt_epoch
                     << ", step: " << ckpt_step;
      }
    }
    MS_LOG(INFO) << "The checkpoint information of all the ranks have been synchronized.";
  }
#endif

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
  ParseLatestCkptInfo(recv_buffer);

  // 5. Remove useless ckpt
  for (int i = SizeToInt(ckpt_files_.size()) - 1; i >= 0; i--) {
    const auto &last_ckpt_name = ckpt_files_[IntToSize(i)];
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

void RecoveryContext::ParseLatestCkptInfo(const std::vector<int> &recv_buffer) {
  std::vector<std::pair<int, int>> ckpts_epoch_step;
  for (std::size_t i = 0; i < recv_buffer.size(); i += kSendBufferLen) {
    (void)ckpts_epoch_step.emplace_back(recv_buffer[i], recv_buffer[i + 1]);
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

void RecoveryContext::CreateConfigFile(const std::string &config_file_path) {
  if (storage::FileIOUtils::IsFileOrDirExist(config_file_path)) {
    MS_LOG(WARNING) << "The config file exists, file path: " << config_file_path;
    return;
  }

  int fd = open(config_file_path.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd == -1) {
    if (errno != EEXIST) {
      MS_LOG(EXCEPTION) << "Create config file: [" << config_file_path << "] failed, errno: " << errno << ", "
                        << strerror(errno);
    }
    MS_LOG(INFO) << "The config file is already created, file path: " << config_file_path;
  } else {
    // Create config file.
    nlohmann::json config_js;
    config_js[std::string(ps::kStoreType)] = 1;
    config_js[std::string(ps::kStoreFilePath)] = recovery_path_ + "/" + ps::kStoreFilePath + kJsonSuffix;
    config_js[std::string(ps::kSchedulerStoreFilePath)] =
      recovery_path_ + "/" + ps::kSchedulerStoreFilePath + kJsonSuffix;

    nlohmann::json recovery_js;
    recovery_js[std::string(ps::kKeyRecovery)] = config_js;

    std::string config_content = recovery_js.dump();
    auto ret_size = write(fd, config_content.c_str(), config_content.size());
    if (ret_size != SizeToLong(config_content.size())) {
      (void)close(fd);
      errno_t err = (ret_size == 0) ? EOF : errno;
      MS_LOG(EXCEPTION) << "Write config file: [" << config_file_path << "] failed, errno: " << err << ", "
                        << strerror(err);
    }
    (void)close(fd);
  }
}

void RecoveryContext::CreatePersistentFile() {
  std::unique_lock<std::mutex> lock(create_persist_json_mtx_);
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
  persistent_json_ = std::make_shared<storage::JsonUtils>(persistent_file_path);
  if (!persistent_json_->Initialize()) {
    MS_LOG(EXCEPTION) << "Initialize json failed, file path: " << persistent_file_path;
  }
}

void RecoveryContext::SetCkptPath(const std::string &path) {
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    return;
  }

  if (!storage::FileIOUtils::IsFileOrDirExist(path)) {
    storage::FileIOUtils::CreateDirRecursive(path);
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

const std::shared_ptr<storage::JsonUtils> &RecoveryContext::persistent_json() {
  if (persistent_json_ == nullptr) {
    CreatePersistentFile();
  }

  MS_EXCEPTION_IF_NULL(persistent_json_);
  return persistent_json_;
}

std::string RecoveryContext::latest_ckpt_file() {
  // For standalone training.
  if (enable_recovery_ && global_rank_size_ == 0 && latest_ckpt_file_.empty()) {
    ObtainLocalLatestCkptInfo();
  }

  return latest_ckpt_file_;
}
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
