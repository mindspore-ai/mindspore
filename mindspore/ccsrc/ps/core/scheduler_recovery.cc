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
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
std::string SchedulerRecovery::GetMetadata(const std::string &key) {
  std::unique_lock<std::mutex> lock(recovery_mtx_);
  MS_EXCEPTION_IF_NULL(recovery_storage_);
  return recovery_storage_->Get(key, "");
}

bool SchedulerRecovery::Recover() {
  std::unique_lock<std::mutex> lock(recovery_mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recovery_storage_, false);
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();

  // 1. recover worker num
  if (recovery_storage_->Exists(kRecoveryWorkerNum)) {
    uint32_t initial_worker_num =
      UlongToUint(std::strtoul(recovery_storage_->Get(kRecoveryWorkerNum, "").c_str(), nullptr, kBase));
    clusterConfig.initial_worker_num = initial_worker_num;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryWorkerNum << " is not contained in " << recovery_storage_->file_path();
  }

  // 2. recover server num
  if (recovery_storage_->Exists(kRecoveryServerNum)) {
    uint32_t initial_server_num =
      UlongToUint(std::strtoul(recovery_storage_->Get(kRecoveryServerNum, "").c_str(), nullptr, kBase));
    clusterConfig.initial_server_num = initial_server_num;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryServerNum << " is not contained in " << recovery_storage_->file_path();
  }

  // 3. recover scheduler ip
  if (recovery_storage_->Exists(kRecoverySchedulerIp)) {
    clusterConfig.scheduler_host = recovery_storage_->GetString(kRecoverySchedulerIp, "");
  } else {
    MS_LOG(EXCEPTION) << kRecoverySchedulerIp << " is not contained in " << recovery_storage_->file_path();
  }

  // 4. recover scheduler port
  if (recovery_storage_->Exists(kRecoverySchedulerPort)) {
    uint16_t scheduler_port = std::strtol(recovery_storage_->Get(kRecoverySchedulerPort, "").c_str(), nullptr, kBase);
    clusterConfig.scheduler_port = scheduler_port;
  } else {
    MS_LOG(EXCEPTION) << kRecoverySchedulerPort << " is not contained in " << recovery_storage_->file_path();
  }

  MS_ERROR_IF_NULL_W_RET_VAL(scheduler_recovery_storage_, false);
  // 5. recover total node num
  if (scheduler_recovery_storage_->Exists(kRecoveryTotalNodeNum)) {
    uint32_t initial_total_node_num =
      UlongToUint(std::strtoul(scheduler_recovery_storage_->Get(kRecoveryTotalNodeNum, "").c_str(), nullptr, kBase));
    clusterConfig.initial_total_node_num = initial_total_node_num;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryTotalNodeNum << " is not contained in " << recovery_storage_->file_path();
  }

  // 6. recover next worker rank id
  if (scheduler_recovery_storage_->Exists(kRecoveryNextWorkerRankId)) {
    uint32_t initial_next_worker_rank_id = UlongToUint(
      std::strtoul(scheduler_recovery_storage_->Get(kRecoveryNextWorkerRankId, "").c_str(), nullptr, kBase));
    clusterConfig.initial_next_worker_rank_id = initial_next_worker_rank_id;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryNextWorkerRankId << " is not contained in " << recovery_storage_->file_path();
  }

  // 7. recover next server rank id
  if (scheduler_recovery_storage_->Exists(kRecoveryNextServerRankId)) {
    uint32_t initial_next_server_rank_id = UlongToUint(
      std::strtoul(scheduler_recovery_storage_->Get(kRecoveryNextServerRankId, "").c_str(), nullptr, kBase));
    clusterConfig.initial_next_server_rank_id = initial_next_server_rank_id;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryNextServerRankId << " is not contained in " << recovery_storage_->file_path();
  }

  // 8. recover register nodes info
  if (scheduler_recovery_storage_->Exists(kRecoveryRegisteredNodesInfos)) {
    auto node_ids = scheduler_recovery_storage_->GetVector(kRecoveryRegisteredNodesInfos);
    std::unordered_map<std::string, NodeInfo> nodes_infos;
    uint32_t recovery_server_num = 0;
    for (auto elem : node_ids) {
      std::string port = elem.at("port");
      std::string rank_id = elem.at("rank_id");

      NodeInfo node_info;
      node_info.ip_ = elem.at("ip");
      node_info.port_ = static_cast<uint16_t>(std::strtol(port.c_str(), nullptr, kBase));
      node_info.node_id_ = elem.at("node_id");
      node_info.rank_id_ = UlongToUint(std::strtoul(rank_id.c_str(), nullptr, kBase));
      node_info.node_role_ = CommUtil::StringToNodeRole(elem.at("role"));

      nodes_infos[node_info.node_id_] = node_info;
      if (elem.at("role") == "SERVER") {
        recovery_server_num += 1;
      }
    }
    if (recovery_server_num != clusterConfig.initial_server_num) {
      MS_LOG(EXCEPTION) << "Server nodes list size is not equal with initial server num. server nodes list size is:"
                        << recovery_server_num << " initial server num is:" << clusterConfig.initial_server_num;
    }
    clusterConfig.initial_registered_nodes_infos = nodes_infos;
  } else {
    MS_LOG(EXCEPTION) << kRecoveryRegisteredNodesInfos << " is not contained in " << recovery_storage_->file_path();
  }

  // 9. recover cluster state
  if (recovery_storage_->Exists(kRecoveryClusterState)) {
    clusterConfig.initial_cluster_state = kClusterStateMap.at(recovery_storage_->GetString(kRecoveryClusterState, ""));
  } else {
    MS_LOG(EXCEPTION) << kRecoveryClusterState << " is not contained in " << recovery_storage_->file_path();
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
