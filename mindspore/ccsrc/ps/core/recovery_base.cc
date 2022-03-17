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

#include "ps/core/recovery_base.h"

namespace mindspore {
namespace ps {
namespace core {
bool RecoveryBase::Initialize(const std::string &config_json) {
  std::unique_lock<std::mutex> lock(recovery_mtx_);
  nlohmann::json recovery_config;
  try {
    recovery_config = nlohmann::json::parse(config_json);
  } catch (nlohmann::json::exception &e) {
    MS_LOG(ERROR) << "Parse the json:" << config_json;
    return false;
  }

  MS_LOG(INFO) << "The node is support recovery.";
  if (!recovery_config.contains(kStoreType)) {
    MS_LOG(WARNING) << "The " << kStoreType << " is not existed.";
    return false;
  }
  std::string storage_file_path = "";
  std::string type = recovery_config.at(kStoreType).dump();
  if (type == kFileStorage) {
    storage_type_ = StorageType::kFileStorage;

    if (!recovery_config.contains(kStoreFilePath)) {
      MS_LOG(WARNING) << "The " << kStoreFilePath << " is not existed.";
      return false;
    }
    storage_file_path = recovery_config.at(kStoreFilePath);
    if (storage_file_path == "") {
      MS_LOG(EXCEPTION) << "If the scheduler support recovery, and if the persistent storage is a file, the path of "
                           "the file must be configured";
    }
    recovery_storage_ = std::make_unique<FileConfiguration>(storage_file_path);
    MS_EXCEPTION_IF_NULL(recovery_storage_);
    if (recovery_storage_->Initialize()) {
      MS_LOG(INFO) << "The storage file path " << storage_file_path << " initialize success.";
    } else {
      return false;
    }
  }

  MS_LOG(INFO) << "The storage type is:" << storage_type_ << ", the storage file path is:" << storage_file_path;
  return true;
}

bool RecoveryBase::InitializeNodes(const std::string &config_json) {
  nlohmann::json recovery_config;
  try {
    recovery_config = nlohmann::json::parse(config_json);
  } catch (nlohmann::json::exception &e) {
    MS_LOG(ERROR) << "Parse the json:" << config_json;
    return false;
  }

  if (!recovery_config.contains(kSchedulerStoreFilePath)) {
    MS_LOG(WARNING) << "The " << kStoreFilePath << " is not existed.";
    return false;
  }

  // this is only for scheduler
  std::string scheduler_storage_file_path = recovery_config.at(kSchedulerStoreFilePath);
  if (scheduler_storage_file_path == "") {
    MS_LOG(WARNING) << "scheduler storage file path is not exist!";
  }
  scheduler_recovery_storage_ = std::make_unique<FileConfiguration>(scheduler_storage_file_path);
  MS_EXCEPTION_IF_NULL(scheduler_recovery_storage_);
  if (scheduler_recovery_storage_->Initialize()) {
    MS_LOG(INFO) << "The scheduler storage file path " << scheduler_storage_file_path << " initialize success.";
  } else {
    return false;
  }

  MS_LOG(INFO) << "the scheduler storage file path is:" << scheduler_storage_file_path;
  return true;
}

void RecoveryBase::Persist(const core::ClusterConfig &clusterConfig) {
  std::unique_lock<std::mutex> lock(recovery_mtx_);
  if (recovery_storage_ == nullptr) {
    MS_LOG(WARNING) << "recovery storage is null, so don't persist meta data";
    return;
  }
  recovery_storage_->PersistFile(clusterConfig);
}

void RecoveryBase::PersistNodesInfo(const core::ClusterConfig &clusterConfig) {
  std::unique_lock<std::mutex> lock(recovery_mtx_);
  if (scheduler_recovery_storage_ == nullptr) {
    MS_LOG(WARNING) << "scheduler recovery  storage is null, so don't persist nodes meta data";
    return;
  }
  scheduler_recovery_storage_->PersistNodes(clusterConfig);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
