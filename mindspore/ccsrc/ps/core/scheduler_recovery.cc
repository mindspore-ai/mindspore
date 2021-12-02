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

#include "ps/core/scheduler_recovery.h"

namespace mindspore {
namespace ps {
namespace core {
std::string SchedulerRecovery::GetMetadata(const std::string &key) {
  MS_EXCEPTION_IF_NULL(recovery_storage_);
  return recovery_storage_->Get(key, "");
}

bool SchedulerRecovery::Recover() {
  if (recovery_storage_ == nullptr) {
    return false;
  }
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();

  // 1. recover worker num
  if (recovery_storage_->Exists(kRecoveryWorkerNum)) {
    clusterConfig.initial_worker_num =
      std::strtol(recovery_storage_->Get(kRecoveryWorkerNum, "").c_str(), nullptr, kBase);
  } else {
    clusterConfig.initial_worker_num = PSContext::instance()->initial_worker_num();
  }

  // 2. recover server num
  if (recovery_storage_->Exists(kRecoveryServerNum)) {
    clusterConfig.initial_server_num =
      std::strtol(recovery_storage_->Get(kRecoveryServerNum, "").c_str(), nullptr, kBase);
  } else {
    clusterConfig.initial_server_num = PSContext::instance()->initial_server_num();
  }

  // 3. recover scheduler ip
  if (recovery_storage_->Exists(kRecoverySchedulerIp)) {
    clusterConfig.scheduler_host = recovery_storage_->GetString(kRecoverySchedulerIp, "");
  } else {
    clusterConfig.scheduler_host = PSContext::instance()->scheduler_host();
  }

  // 4. recover scheduler port
  if (recovery_storage_->Exists(kRecoverySchedulerPort)) {
    clusterConfig.scheduler_port =
      std::strtol(recovery_storage_->Get(kRecoverySchedulerPort, "").c_str(), nullptr, kBase);
  } else {
    clusterConfig.scheduler_port = PSContext::instance()->scheduler_port();
  }

  MS_LOG(INFO) << "The worker num:" << clusterConfig.initial_worker_num
               << ", the server num:" << clusterConfig.initial_server_num
               << ", the scheduler ip:" << clusterConfig.scheduler_host
               << ", the scheduler port:" << clusterConfig.scheduler_port;

  if (scheduler_recovery_storage_ == nullptr) {
    MS_LOG(WARNING) << "scheduler recovery storage is null. return false";
    return false;
  }
  // 5. recover total node num
  if (scheduler_recovery_storage_->Exists(kRecoveryTotalNodeNum)) {
    clusterConfig.initial_total_node_num =
      std::strtol(scheduler_recovery_storage_->Get(kRecoveryTotalNodeNum, "").c_str(), nullptr, kBase);
  }

  // 6. recover next worker rank id
  if (scheduler_recovery_storage_->Exists(kRecoveryNextWorkerRankId)) {
    clusterConfig.initial_next_worker_rank_id =
      std::strtol(scheduler_recovery_storage_->Get(kRecoveryNextWorkerRankId, "").c_str(), nullptr, kBase);
  }

  // 7. recover next server rank id
  if (scheduler_recovery_storage_->Exists(kRecoveryNextServerRankId)) {
    clusterConfig.initial_next_server_rank_id =
      std::strtol(scheduler_recovery_storage_->Get(kRecoveryNextServerRankId, "").c_str(), nullptr, kBase);
  }

  // 8. recover register nodes info
  if (scheduler_recovery_storage_->Exists(kRecoveryRegisteredNodesInfos)) {
    auto node_ids = scheduler_recovery_storage_->GetVector(kRecoveryRegisteredNodesInfos);
    std::unordered_map<std::string, NodeInfo> nodes_infos;
    for (auto elem : node_ids) {
      std::string port = elem.at("port");
      std::string rank_id = elem.at("rank_id");

      NodeInfo node_info;
      node_info.ip_ = elem.at("ip");
      node_info.port_ = std::strtol(port.c_str(), nullptr, kBase);
      node_info.node_id_ = elem.at("node_id");
      node_info.rank_id_ = std::strtol(rank_id.c_str(), nullptr, kBase);
      node_info.is_alive = CommUtil::StringToBool(elem.at("alive"));
      node_info.node_role_ = CommUtil::StringToNodeRole(elem.at("role"));

      nodes_infos[node_info.node_id_] = node_info;
    }
    clusterConfig.initial_registered_nodes_infos = nodes_infos;
  }

  MS_LOG(INFO) << "The worker num:" << clusterConfig.initial_worker_num
               << ", the server num:" << clusterConfig.initial_server_num
               << ", the scheduler ip:" << clusterConfig.scheduler_host
               << ", the scheduler port:" << clusterConfig.scheduler_port
               << ", the initial total node num:" << clusterConfig.initial_total_node_num
               << ", the initial next worker rank id:" << clusterConfig.initial_next_worker_rank_id
               << ", the initial next server rank id:" << clusterConfig.initial_next_server_rank_id;

  if (!clusterConfig.initial_registered_nodes_infos.empty()) {
    for (const auto kvs : clusterConfig.initial_registered_nodes_infos) {
      MS_LOG(INFO) << "The ip:" << kvs.second.ip_ << ", the port:" << kvs.second.port_
                   << ", the node_id:" << kvs.second.node_id_
                   << ", the node_role:" << CommUtil::NodeRoleToString(kvs.second.node_role_)
                   << ", the rank_id_:" << kvs.second.rank_id_
                   << ", the is_alive:" << CommUtil::BoolToString(kvs.second.is_alive);
    }
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
