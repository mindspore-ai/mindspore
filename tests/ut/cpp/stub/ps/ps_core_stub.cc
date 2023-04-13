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

#include "ps/core/abstract_node.h"
#include "ps/core/abstract_ps_node.h"
#include "ps/core/node.h"
#include "ps/core/scheduler_node.h"
#include "ps/core/file_configuration.h"
#include "ps/core/ps_scheduler_node.h"
#include "ps/core/ps_worker_node.h"
#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
namespace mindspore {
namespace ps {
void PsDataPrefetch::CreateDataChannel(const std::string &channel_name, size_t step_num) {}
namespace core {
bool AbstractPSNode::InitClientToScheduler() { return true; }
void AbstractPSNode::RegisterInitCollectCommResphandler() {}
void AbstractPSNode::RegisterRecoveryRespHandler() {}

AbstractNode::~AbstractNode() {}
void AbstractNode::Register(const std::shared_ptr<TcpClient> &client) {}
bool AbstractNode::InitClientToScheduler() { return true; }
bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                                   const Protos &protos, const void *data, size_t size, const uint32_t &timeout) {
  return true;
}
bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                                   const uint32_t &timeout) {
  return true;
}
void AbstractNode::NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta) {}

void PSSchedulerNode::RunRecovery() {}
void PSSchedulerNode::RegisterInitCollectCommServiceHandler() {}
void PSSchedulerNode::RegisterRecoveryServiceHandler() {}
void PSSchedulerNode::HandleNodeTimeoutForRecovery(
  const std::unordered_map<std::string, NodeInfo> &timeout_nodes_infos) {}
void PSSchedulerNode::HandleNodeRecoverByHeartBeat(uint32_t rank_id) {}
void PSSchedulerNode::RecoverFromPersistence() {}

SchedulerNode::~SchedulerNode() {}
bool SchedulerNode::Start(const uint32_t &timeout) { return true; }
bool SchedulerNode::Stop() { return true; }
bool SchedulerNode::Finish(const uint32_t &timeout) { return true; }
void SchedulerNode::RunRecovery() {}
bool SchedulerNode::SendPrepareBuildingNetwork(const std::unordered_map<std::string, NodeInfo> &node_infos) {
  return true;
}
void SchedulerNode::RecordSchedulerRestartInfo() {}
void SchedulerNode::InitEventTxtFile() {}

void PSWorkerNode::Register(const std::shared_ptr<TcpClient> &client) {}
bool PSWorkerNode::Start(const uint32_t &timeout) { return true; }
bool PSWorkerNode::Stop() { return true; }
bool PSWorkerNode::Finish(const uint32_t &timeout) { return true; }

bool FileConfiguration::Initialize() { return true; }
bool FileConfiguration::IsInitialized() const { return true; }
std::string FileConfiguration::Get(const std::string &key, const std::string &defaultvalue) const { return ""; }
std::string FileConfiguration::GetString(const std::string &key, const std::string &defaultvalue) const { return ""; }
std::vector<nlohmann::json> FileConfiguration::GetVector(const std::string &key) const { return {}; }
int64_t FileConfiguration::GetInt(const std::string &key, int64_t default_value) const { return 0; }
void FileConfiguration::Put(const std::string &key, const std::string &value) {}
bool FileConfiguration::Exists(const std::string &key) const { return true; }

void FileConfiguration::PersistFile(const core::ClusterConfig &clusterConfig) const {}

void FileConfiguration::PersistNodes(const core::ClusterConfig &clusterConfig) const {}

std::string FileConfiguration::file_path() const { return ""; }
uint32_t Node::rank_id() const { return 0; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
