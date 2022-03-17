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

#include "fl/server/server_recovery.h"
#include "fl/server/local_meta_store.h"
#include "debug/common.h"

namespace mindspore {
namespace fl {
namespace server {
bool ServerRecovery::Initialize(const std::string &config_file) {
  std::unique_lock<std::mutex> lock(server_recovery_file_mtx_);
  config_ = std::make_unique<ps::core::FileConfiguration>(config_file);
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(EXCEPTION) << "Initializing for server recovery failed. Config file path " << config_file
                      << " may be invalid or not exist.";
    return false;
  }

  // Read the server recovery file path.
  if (!config_->Exists(kServerRecovery)) {
    MS_LOG(WARNING) << "Server recovery config is not set. This node doesn't support recovery.";
    return true;
  } else {
    std::string value = config_->Get(kServerRecovery, "");
    nlohmann::json value_json;
    try {
      value_json = nlohmann::json::parse(value);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "The data is not in json format.";
      return false;
    }

    // Parse the storage type.
    uint32_t storage_type = JsonGetKeyWithException<uint32_t>(value_json, ps::kStoreType);
    if (std::to_string(storage_type) != ps::kFileStorage) {
      MS_LOG(EXCEPTION) << "Storage type " << storage_type << " is not supported.";
      return false;
    }

    // Parse storage file path.
    server_recovery_file_path_ = JsonGetKeyWithException<std::string>(value_json, ps::kStoreFilePath);
    MS_LOG(INFO) << "Server recovery file path is " << server_recovery_file_path_;
  }
  return true;
}

bool ServerRecovery::Recover() {
  std::unique_lock<std::mutex> lock(server_recovery_file_mtx_);
  server_recovery_file_.open(server_recovery_file_path_, std::ios::in);
  if (!server_recovery_file_.good() || !server_recovery_file_.is_open()) {
    MS_LOG(WARNING) << "Can't open server recovery file " << server_recovery_file_path_;
    return false;
  }

  nlohmann::json server_recovery_json;
  try {
    server_recovery_json = nlohmann::json::parse(server_recovery_file_);
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "The server recovery file is not in json format.";
    return false;
  }
  uint64_t current_iter = JsonGetKeyWithException<uint64_t>(server_recovery_json, kCurrentIteration);
  std::string instance_state = JsonGetKeyWithException<std::string>(server_recovery_json, kInstanceState);

  LocalMetaStore::GetInstance().set_curr_iter_num(current_iter);
  LocalMetaStore::GetInstance().set_curr_instance_state(GetInstanceState(instance_state));

  MS_LOG(INFO) << "Recover from persistent storage: current iteration number is " << current_iter
               << ", instance state is " << instance_state;
  server_recovery_file_.close();
  return true;
}

bool ServerRecovery::Save(uint64_t current_iter, InstanceState instance_state) {
  std::unique_lock<std::mutex> lock(server_recovery_file_mtx_);
  server_recovery_file_.open(server_recovery_file_path_, std::ios::out | std::ios::ate);
  if (!server_recovery_file_.good() || !server_recovery_file_.is_open()) {
    MS_LOG(WARNING) << "Can't save data to recovery file " << server_recovery_file_path_
                    << ". This file path is invalid or does not exit.";
    return false;
  }

  nlohmann::json server_metadata_json;
  server_metadata_json[kCurrentIteration] = current_iter;
  server_metadata_json[kInstanceState] = GetInstanceStateStr(instance_state);
  server_recovery_file_ << server_metadata_json;
  server_recovery_file_.close();
  return true;
}

bool ServerRecovery::SyncAfterRecovery(const std::shared_ptr<ps::core::TcpCommunicator> &communicator,
                                       uint32_t rank_id) {
  std::unique_lock<std::mutex> lock(server_recovery_file_mtx_);
  // If this server is follower server, notify leader server that this server has recovered.
  if (rank_id != kLeaderServerRank) {
    MS_ERROR_IF_NULL_W_RET_VAL(communicator, false);
    SyncAfterRecover sync_after_recover_req;
    sync_after_recover_req.set_current_iter_num(LocalMetaStore::GetInstance().curr_iter_num());
    if (!communicator->SendPbRequest(sync_after_recover_req, kLeaderServerRank,
                                     ps::core::TcpUserCommand::kSyncAfterRecover)) {
      MS_LOG(ERROR) << "Sending sync after recovery message to leader server failed.";
      return false;
    }
  }
  return true;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
